
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import pickle 
import torchaudio 
import torch.distributed as dist 
import utils 
import scipy.sparse as sp 
import os
import json
import torch.nn.functional as F


class WaveformDataset(Dataset):
    def __init__(self, args, config, split=None):
        self.split = split
        self.dataset_path = f'{args.dataset_path}{args.dataset.lower()}/trunc_mp3/'
        self.audio_model = args.audio_model
        self.audio_conf = config['MODELS']['AUDIO_MODELS'][self.audio_model]
        self.data_conf = config['DATASETS'][args.dataset][f'{args.dataset_size}_SET']
        self.dataset_name = args.dataset 
        self.dataset_org_path = self.data_conf['org_filepath'] if split == None else self.data_conf[split]
        if split == 'test' and args.session: 
            self.dataset_org_path = self.data_conf['session']
        if split == 'test' and args.trunc: 
            self.dataset_org_path = self.data_conf['trunc']
        self.data = pickle.load(open(self.dataset_org_path, 'rb')) 
        self.caption_mode = config['cmdline']['caption_mode'] if args.dataset != 'MUSIC_CAPS' else ''
        self.audio_mode = config['cmdline']['audio_mode']
        self.audio_key = self.audio_mode 
        self.id_key = config['DATASETS'][args.dataset]['id_key']
        self.verbose = args.verbose

        with open(os.path.join(config['EVALUATION']['BUNDLE'][args.dataset]['train'], f'train_datasize')) as f:
            self.num_playlists, self.num_tracks = [eval(i.strip()) for i in f.read().split(",")]
        
    def __getitem__(self, index):
        datum = self.data.loc[index] 
        mp3_file = datum[self.audio_key]
        if self.audio_mode == 'short_path': 
            mp3_file = f'/mnt/ssd1/rebecca/lfm/trunc_mp3/{mp3_file}'
            # mp3_file = f'{self.dataset_path}{mp3_file}'
        try: 
            waveform, sr = torchaudio.load(mp3_file)
        except: 
            print(mp3_file)
        # waveform, sr = torchaudio.load(mp3_file)
        text = datum[f'{self.caption_mode}_caption']
        uri = datum[self.id_key]
        return waveform, text, uri 

    def __len__(self):
        return len(self.data)

class CF_WaveformDataset(Dataset):
    def __init__(self, args, config, split):
        self.mode = args.mode  
         
        #Load configs 
        self.audio_conf = config['MODELS']['AUDIO_MODELS'][args.audio_model]
        self.data_conf = config['DATASETS'][args.dataset][f'{args.dataset_size}_SET']
        self.dataset_path = f'{args.dataset_path}{args.dataset.lower()}/trunc_mp3/'

        #Load settings 
        self.keep_ego_pairs = True if args.mode == 'TTC' or args.mode == 'TPC' else False 
        self.split = split
        self.caption_mode = config['cmdline']['caption_mode'] if args.dataset != 'MUSIC_CAPS' else ''
        self.audio_mode = config['cmdline']['audio_mode'] 
        self.id_key = config['DATASETS'][args.dataset]['id_key']
        self.verbose = args.verbose

        #Load data
        if self.mode == 'TTC' or self.mode == 'TPC': 
            mode = {"TTC":"CIC", "TPC":"CIP"}[self.mode]
            self.pairs = pickle.load(open(self.data_conf[mode]['pair_dataset_path'], 'rb')) # pandas df with pair sets 
            self.num_pairs = len(self.pairs) 
            self.audio_A = self.pairs[f'{self.audio_mode}_A'].tolist()
            self.audio_B = self.pairs[f'{self.audio_mode}_B'].tolist()
            self.caption_A = self.pairs[f'{self.caption_mode}_caption_A'].tolist()
            self.caption_B = self.pairs[f'{self.caption_mode}_caption_B'].tolist()
            self.uri_A = self.pairs[f'{self.id_key}_A'].tolist()
            self.uri_B = self.pairs[f'{self.id_key}_B'].tolist()
            path = config['EVALUATION']['BUNDLE'][args.dataset]['train'] 
            with open(os.path.join(path, f'train_datasize')) as f:
                self.num_playlists, self.num_tracks = [eval(i.strip()) for i in f.read().split(",")]

        if self.mode == 'TPC': 
            self.idx = self.pairs[f'idx'].tolist()
            self.pidx = self.pairs[f'pidx'].tolist()
            path = config['EVALUATION']['BUNDLE'][args.dataset]['train']   
            # with open(os.path.join(path, f'train_datasize')) as f:
            #     self.num_bundles, self.num_items = [eval(i.strip()) for i in f.read().split(",")]
            # load uri2id:
            with open(os.path.join(path, f'train_uri2id'), "r") as f:
                self.uri2id = json.loads(f.read())

            b_i_pairs = np.load(os.path.join(path, f'bi_train.npy'))
            self.b_i_graph = pairs2csr(b_i_pairs, (self.num_playlists, self.num_tracks))
            self.len_max = int(self.b_i_graph.sum(axis=1).max())
            self.num_truncate = args.num_truncate
            # sequences_dataset_path
            self.sequences = torch.load(self.data_conf[mode]['sequences_dataset_path'])

        if args.verbose: 
            print(f"loaded a {self.mode} dataset with {self.num_pairs} entries")
            print(f"ego flag set to {self.keep_ego_pairs}")
    
    def __len__(self):
        return self.num_pairs
        # return 100

    def __getitem__(self, index):
        if self.keep_ego_pairs:
            item_pairs = [                
                (self.audio_A[index], self.caption_A[index], self.uri_A[index]),
                (self.audio_B[index], self.caption_B[index], self.uri_B[index]),

                (self.audio_A[index], self.caption_B[index], self.uri_A[index]),
                (self.audio_B[index], self.caption_A[index], self.uri_B[index]),              
            ]
        else:
            item_pairs = [
                (self.audio_A[index], self.caption_B[index]),
                (self.audio_B[index], self.caption_A[index])
            ]            

        for i, audio_caption_pair in enumerate(item_pairs):
            if self.keep_ego_pairs :
                item_pairs[i] = self.process_audio(audio_caption_pair[0], audio_caption_pair[1], audio_caption_pair[2])
            else: 
                item_pairs[i] = self.process_audio(audio_caption_pair[0], audio_caption_pair[1])
        if self.mode == 'TPC': 
            pidx = self.pidx[index]
            # b_i_i = torch.from_numpy(self.b_i_graph[pidx].toarray()).squeeze()
            # indices = torch.argwhere(b_i_i)[:,0]
            # pad_token = self.num_tracks
            # seq_b_i = F.pad(indices, (0, self.len_max-len(indices)), value=pad_token)
            seq_b_i = self.sequences[pidx]
            pad_token = self.num_tracks

            mask = (seq_b_i == pad_token).detach()
            length = torch.sum(1-mask.float(), dim=-1)

            # reorder:
            order = torch.randperm(int(length))
            seq_b_i[:int(length)] = seq_b_i[order]

            # truncate:
            seq_b_i = seq_b_i[:self.num_truncate]
            mask = (seq_b_i == pad_token).detach()
            seq_b_i = seq_b_i.masked_fill(mask, 0)
            length = torch.sum(1-mask.float(), dim=-1)

            item_pairs.append((self.idx[index], seq_b_i, mask, length))
            
        return item_pairs

    def process_audio(self, audio, caption, uri=None): 
        if self.audio_mode == 'short_path': 
            audio = f'/mnt/ssd1/rebecca/lfm/trunc_mp3/{audio}'
            # audio = f'{self.dataset_path}{audio}'
        waveform, sr = torchaudio.load(audio)
        if uri != None: 
            return waveform, caption, uri 
        return waveform, caption

    # For cross-item pairs, the diagonal of image-text similarity matrix indicates the cross-item matching pairs
    # The upper left / lower right half diagonal indicates the original image-text pair (ego item pair)
    # We need to mask both cross-item pairs and ego-item pairs.
    def generate_batch_cross_item_mask(self, num_items):
        mask = np.diag(np.ones(num_items * 2))
        mask += np.diag(np.ones(num_items), num_items)
        mask += np.diag(np.ones(num_items), -num_items)
        mask = 1 - mask
        return mask
    

    def collate_CF_pair_finetune_batch(self, batch):
        # if self.keep_ego_pairs:
        if self.mode == 'TTC': 
            (ego_item_pair_0, ego_item_pair_1, cross_item_pair_0, cross_item_pair_1) = zip(*batch)
            ego_pairs = ego_item_pair_0 + ego_item_pair_1
            ego_pairs = list(zip(*ego_pairs))
            cross_pairs = cross_item_pair_0 + cross_item_pair_1
            cross_pairs = list(zip(*cross_pairs))

            cross_item_mask = self.generate_batch_cross_item_mask(len(cross_item_pair_0))
            return ego_pairs, cross_pairs, cross_item_mask

        if self.mode == 'TPC':
            (ego_item_pair_0, ego_item_pair_1, cross_item_pair_0, cross_item_pair_1, idxes) = zip(*batch)
            ego_pairs = ego_item_pair_0 + ego_item_pair_1
            ego_pairs = list(zip(*ego_pairs))
            cross_pairs = cross_item_pair_0 + cross_item_pair_1
            cross_pairs = list(zip(*cross_pairs))

            cross_item_mask = self.generate_batch_cross_item_mask(len(cross_item_pair_0))
            t_idx, seq, mask, length = zip(*idxes)

            return ego_pairs, cross_pairs, cross_item_mask, t_idx, seq, mask, length
        

class Datasets(): 
    def __init__(self, args, config, split=True): 
        self.config = config['MODELS']['BASE_MODELS']['LARP'] 
        self.audio_config = config['MODELS']['AUDIO_MODELS'][args.audio_model.upper()]
        batch_size = config['cmdline']['batch_size']
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()  
        

        if args.mode == 'train': #Training LARP from scratch 
            self.train_dataset = WaveformDataset(args, config, split='train')
            self.valid_dataset = WaveformDataset(args, config, split='valid')
            self.test_dataset  = WaveformDataset(args, config, split='test')
            self.train_sampler = torch.utils.data.DistributedSampler(self.train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            self.valid_sampler = torch.utils.data.DistributedSampler(self.valid_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            self.test_sampler = torch.utils.data.DistributedSampler(self.test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=config['cmdline']['batch_size'], num_workers=self.config['n_worker'], pin_memory=False, sampler=self.train_sampler, shuffle=False, collate_fn=None, drop_last=True)              
            self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=config['cmdline']['batch_size'], num_workers=self.config['n_worker'], pin_memory=True,  shuffle=False, collate_fn=None,drop_last=False)   
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=config['cmdline']['batch_size'], num_workers=self.config['n_worker'], pin_memory=True, shuffle=False, collate_fn=None,drop_last=False)

        elif args.mode == "TTC" or args.mode == "TPC": 
            self.train_dataset = CF_WaveformDataset(args, config, split='train')
            self.valid_dataset = WaveformDataset(args, config, split='valid')
            self.test_dataset  = WaveformDataset(args, config, split='test')
            self.train_sampler = torch.utils.data.DistributedSampler(self.train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            self.valid_sampler = torch.utils.data.DistributedSampler(self.valid_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            self.test_sampler = torch.utils.data.DistributedSampler(self.test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=config['cmdline']['batch_size'], num_workers=self.config['n_worker'], pin_memory=True, sampler=self.train_sampler, shuffle=False, collate_fn=self.train_dataset.collate_CF_pair_finetune_batch, drop_last=True)              
            self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=config['cmdline']['batch_size'], num_workers=self.config['n_worker'], pin_memory=True,  shuffle=False, collate_fn=None,drop_last=False)   
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=config['cmdline']['batch_size'], num_workers=self.config['n_worker'], pin_memory=True, shuffle=False, collate_fn=None,drop_last=False)


def pairs2csr(pairs, shape):
    indice = np.array(pairs, dtype=np.int32)
    values = np.ones(len(pairs), dtype=np.float32)
    return sp.csr_matrix(
        (values, (indice[:, 0], indice[:, 1])), shape=shape)

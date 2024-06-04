import math
import numpy as np
import io
import os
import time
from functools import partial
from collections import defaultdict, deque
from datetime import datetime
import torch
import torch.distributed as dist
import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import random
import scipy.sparse as sp 
import pickle
import ipdb
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
# import laion_clap
import pandas as pd 
import webdataset as wds
import torch.nn.functional as F
import torch.nn as nn 
from glob import glob 
import random
import warnings
warnings.filterwarnings("ignore")

#BLAP utils 
def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    

def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):        
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    
      
def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg_for_log(self): 
        loss_dict = {name: meter.global_avg for name, meter in self.meters.items()}
        return loss_dict

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)    
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        
def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)
    
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, word {}): {}'.format(
        args.rank, args.world_size, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)        


#My Utils 

#HOUSEKEEPING // CHECKS 

#RUN SETUP UTILS 
def setup_ddp(rank, world_size, cuda_id, master_port):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost" #ip address of the gpu
    os.environ["MASTER_PORT"] = master_port #free port on the gpu 
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(int(cuda_id)) 

def get_gpu_id(rank, args): 
    if args.gpu == "-1": 
        return rank, rank 
    gpu_ids = args.gpu.split(',')
    return rank, gpu_ids[rank]

def setup_seed(seed): 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

def setup_logging(args): 
    log_base, dataset_name, audio_model, text_model, settings, loss, embedding, batch_size, mode, caption_mode, base_model = args['log_base'],args['dataset'], args['audio_model'], args['text_model'], args['info'], args['ablation_loss'], args['embed_dim'], args['batch_size'], args['mode'], args['caption_mode'], args['base_model']
    
    date = datetime.now().strftime("%m_%d__%H:%M")
    settings = "_".join(settings)
    prefix = f'{date}_{loss}_e{embedding}_b{batch_size}_{caption_mode}_{settings}'
    log_path = f"{log_base}/{mode}/log/{dataset_name}/{base_model}_{audio_model}_{text_model}/{prefix}"
    run_path = f"{log_base}/{mode}/runs/{dataset_name}/{base_model}_{audio_model}_{text_model}/{prefix}" 
    checkpoint_model_path = f"{log_base}/{mode}/checkpoints/{dataset_name}/{base_model}_{audio_model}_{text_model}/model/{prefix}" 
    checkpoint_conf_path = f"{log_base}/{mode}/checkpoints/{dataset_name}/{base_model}_{audio_model}_{text_model}/conf/{prefix}" 
    best_results_path = f"{log_base}/{mode}/checkpoints/{dataset_name}/{base_model}_{audio_model}_{text_model}//best_performance/{prefix}"
    summary_path = f"{log_base}/{mode}/summary/{prefix}"
    all_experiments_path = f"/home/rebecca/BLAP_test/all_experiments.csv" 

    if not os.path.isdir(run_path):
        os.makedirs(run_path, exist_ok=True)
    if not os.path.isdir(log_path):
        os.makedirs(log_path, exist_ok=True)
    if not os.path.isdir(checkpoint_model_path):
        os.makedirs(checkpoint_model_path, exist_ok=True)
    if not os.path.isdir(checkpoint_conf_path):
        os.makedirs(checkpoint_conf_path, exist_ok=True)
    if not os.path.isdir(best_results_path): 
        os.makedirs(best_results_path, exist_ok=True)
    if not os.path.isdir(summary_path): 
        os.makedirs(summary_path, exist_ok=True)
    
    return log_path, run_path, checkpoint_model_path, checkpoint_conf_path, best_results_path, summary_path, all_experiments_path, prefix 

#LOGGING 
def log_stats(stats, mode, epoch, log_path, eval_mode, rounding_int=4):
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    log_stats = {'mode': mode.upper(), 'epoch': epoch, 'current_time': curr_time, **{f'{mode}_{k}': v for k, v in stats.items()}} 
    log = open(os.path.join(log_path, f'{eval_mode}_stats.txt'), "a")
    log.write("%s\n" %(log_stats))
    log.close()               
      
def save_checkpoint(model, optimizer, config, epoch, checkpoint_path): 
    save_obj = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config, 
        'epoch': epoch 
    }
    if config['cmdline']['fusion_method'] == "self_attn" and config['cmdline']['mode'] == "CIP":
        if hasattr(model, 'module'):
            save_obj['playlist_encoder'] = [i.state_dict() for i in model.module.playlist_constructor.transformer_encoder]
        else:
            save_obj['playlist_encoder'] = [i.state_dict() for i in model.playlist_constructor.transformer_encoder]

    torch.save(save_obj, os.path.join(checkpoint_path, 'checkpoint_%02d.pth'%epoch))

class EarlyStopping(nn.Module):
    def __init__(self, epoch,decision_metric_name,decision_index, verbose=True):
        super(EarlyStopping, self)
        self.best_epoch = 0
        self.best_score = 0
        self.stop = False
        self.epochs = epoch
        self.distance = 0
        self.verbose = verbose
        self.decision_metric_name = decision_metric_name
        self.decision_index = decision_index
        self.start_time = datetime.now() 
        self.end_time = None 
        
    def update(self, config, decision_metric, model, optimizer, results_path, all_results, epoch, checkpoint=True, features=None):
       
        if decision_metric >= self.best_score:
            print(f"ES | found a better/equal {self.decision_metric_name}@{self.decision_index}: current:{np.round(decision_metric, 4)} -- epoch:{epoch}, previously:{np.round(self.best_score, 4)} -- epoch:{self.best_epoch}")
            self.best_epoch = epoch
            self.best_score = decision_metric
            self.distance = 0
            self.stop=False
            
            # print(f"find a better: {self.best_epoch} - {self.best_score}")
            log_stats(stats =all_results, mode='VALID', epoch=epoch, log_path=results_path, eval_mode='best_performance')
            self.end_time = datetime.now() 
            if checkpoint: 
                save_checkpoint(model=model, optimizer=optimizer, config=config, epoch=epoch, checkpoint_path=results_path)
                if features != None: 
                    print('hi')
        else:
            log_stats(stats =all_results, mode='VALID', epoch=epoch, log_path=results_path, eval_mode='best_performance')
            print(f'ES | no improvement: best:{np.round(self.best_score, 4)} - current:{np.round(decision_metric, 4)}, increasing patience:{self.distance}/{self.epochs}')
            self.distance += 1
            if self.distance > self.epochs:
                self.stop = True
                print(f"ES | early stopping: {self.best_epoch} - {self.best_score}")

# RUN UTILS 
def add_layer_to_opt(model): 
    def exclude(n,p):
        for i in ['playlist_constructor', 'audio_features_all', 'text_features_all']:
            if i in n:
                return True
        return False
    # exclude = (lambda n, p: 'playlist_constructor' in n)
    og_params = {'params': [p for n,p in model.named_parameters() if not exclude(n,p)]}
    new_params = {'params': [p for n,p in model.named_parameters() if exclude(n,p)]}
    return og_params, new_params


# EVAL UTILS 
def order_emb_by_uri(uri2id, uris, features):
    print("reorder features >>>")
    orders = np.zeros(len(uri2id))
    for idx, uri in enumerate(uris):
        id = int(uri2id[uri])
        orders[id] = idx
    features = features[orders]
    uris = np.array(uris)[orders.astype(int)]
    print("reorder features <<<")
    return uris, features 


#BENCHMARK UTILS 
class FileDataset(Dataset): 
    def __init__(self, args, config, split): 
        self.split = split
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
        
    def __getitem__(self, index): 
        datum = self.data.loc[index] 
        mp3_file = datum[self.audio_key]
        if self.audio_mode == 'short_path': 
            mp3_file = f'/mnt/ssd1/rebecca/lfm/trunc_mp3/{mp3_file}'
        # waveform, sr = torchaudio.load(mp3_file)
        text = datum[f'{self.caption_mode}_caption']
        uri = datum[self.id_key]
        return mp3_file, text, uri 
    
    def __len__(self): 
        return len(self.data)

class BenchmarkDatasets(): 
    def __init__(self, args, config): 
        self.config = config 

        self.dataset = FileDataset(args, config, 'test')
        if args.dataset == 'MusicCaps': 
            self.dataset = MusicCapsDataset(args)
        if args.distributed: 
            num_tasks = dist.get_world_size()
            global_rank = dist.get_rank() 
            self.sampler = torch.utils.data.DistributedSampler(self.dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            self.dataloader = DataLoader(self.dataset, batch_size=self.config['MODELS']['AUDIO_MODELS'][args.audio_model.upper()]['batch_size'], num_workers=self.config['MODELS']['BASE_MODELS'][args.base_model]['n_worker'], pin_memory=True, sampler=self.sampler, shuffle=False, collate_fn=None,drop_last=False)   
        else: 
            self.dataloader = DataLoader(self.dataset, batch_size=self.config['MODELS']['AUDIO_MODELS'][args.audio_model.upper()]['batch_size'], num_workers=self.config['MODELS']['BASE_MODELS'][args.base_model]['n_worker'], pin_memory=True, shuffle=False, collate_fn=None,drop_last=False)







##### DEPRECATED BELOW THIS LINE 

# def add_playlist_layer_to_opt(model, playlist_constructor): 
#     all_params = {'params': [p for n,p in model.named_parameters()] + [p for n,p in playlist_constructor.transformer_encoder[0].named_parameters()]}
#     return all_params
    
# def normalize_interaction_graph(pt_graph): 
#     row_sum = sp.diags(1/(pt_graph.sum(axis=1).A.ravel() + 1e-9))
#     pt_graph = row_sum @ pt_graph
#     pt_graph = pt_graph.tocoo()
#     values = pt_graph.data
#     indices = np.vstack((pt_graph.row, pt_graph.col))
#     pt_graph = torch.sparse.FloatTensor(torch.LongTensor(
#         indices), torch.FloatTensor(values), torch.Size(pt_graph.shape))
#     return pt_graph 


# #AUDIO-TEXT DATASET 

# class MusicCapsDataset(Dataset): 
#     def __init__(self, args):
#         self.folder_path = args.input_path 
#         self.data = pd.read_csv('/ssd3/rebecca/music_caps/clean_musiccaps.csv')

#     def __getitem__(self, index): 
#         datum = self.data.loc[index]
#         yid, caption = datum['ytid'], datum['caption']
#         mp3_fp = os.join(self.folder_path, f'{yid}.mp3')
#         return yid, mp3_fp, caption
            
#     def __len__(self): 
#         return len(self.data)

# class Mp3Dataset(Dataset): 
#     def __init__(self, args, config): 
#         self.data_conf = config[args.dataset]
#         self.caption_mode = config['cmdline']['caption_mode']
#         self.audio_mode = config['cmdline']['audio_mode']
#         self.audio_key = 'slices' if 'slice' in self.audio_mode else 'full_mp3'
#         self.data = pickle.load(open(self.data_conf['test'], 'rb')) 
        
#     def __getmp3__(self, slices): 
#         if self.audio_mode == 'random_slice':
#             mp3_path = np.random.choice(slices)
#             uri = mp3_path.split('_')[0]
#             return mp3_path, uri
#         if self.audio_mode == 'first_slice': 
#             mp3_path = slice[0]
#             return mp3_path, uri
#         if self.audio_mode == 'full_mp3': 
#             return slices, uri
#     def __getitem__(self, index): 
#         datum = self.data.loc[index]
#         file_path, uri = self.__getmp3__(datum[self.audio_key])
#         caption = datum[f'{self.caption_mode}caption']
#         return uri, file_path, caption 
#     def __len__(self): 
#         return len(self.data)

# class WaveformDataset(Dataset):
#     def __init__(self, args, config, split=None):
#         """
#         Dataset that manages audio recordings
#         Based on dataloader from AST implementation: https://github.com/YuanGongND/ast/blob/master/src/dataloader.py
#         :param args: commandline arguments that specify parameters for run 
#         :param config: Dictionary containing the audio loading and preprocessing settings
#         :param split: specifying the dataset split, if None return entire dataset 
#         """
#         self.split = split
#         self.audio_model = args.audio_model
#         self.audio_conf = config['MODELS']['AUDIO_MODELS'][self.audio_model]
#         self.data_conf = config['DATASETS'][args.dataset][f'{args.dataset_size}_SET']
#         self.dataset_name = args.dataset 
#         self.dataset_org_path = self.data_conf['org_filepath'] if split == None else self.data_conf[split]
#         self.data = pickle.load(open(self.dataset_org_path, 'rb')) 
#         self.caption_mode = config['cmdline']['caption_mode'] if args.dataset != 'MUSIC_CAPS' else ''
#         self.audio_mode = config['cmdline']['audio_mode']
#         self.audio_key = self.audio_mode 
#         self.id_key = config['DATASETS'][args.dataset]['id_key']
#         self.melbins = self.audio_conf['num_mel_bins']
#         self.freqm = self.audio_conf['freqm']
#         self.timem = self.audio_conf['timem']
#         self.mixup = self.audio_conf['mixup']
#         self.max_length_ms = self.audio_conf['max_length_ms']
#         self.verbose = args.verbose
#         if self.verbose:
#             print(f'---------------Loading the {self.dataset_name} dataset with {self.split} split---------------') 
#             print(f"now using following mask: {self.audio_conf.get('freqm'):d} freq, {self.audio_conf.get('timem'):d} time")
#             print(f'now using mix-up with rate {self.mixup:f}')
#         # dataset spectrogram mean and std, used to normalize the input
#         self.norm_mean = self.audio_conf['mean']
#         self.norm_std = self.audio_conf['std']
#         # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
#         # set it as True ONLY when you are getting the normalization stats.
#         self.skip_norm = self.audio_conf['skip_norm'] if self.audio_conf['skip_norm'] else False
#         if self.skip_norm and self.verbose:
#             print('now skip normalization (use it ONLY when you are computing the normalization stats).')
#         else:
#             if self.verbose: 
#                 print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
#         # if add noise for data augmentation
#         self.noise = self.audio_conf.get('noise')
#         if self.noise == True and self.verbose:
#             print('now use noise augmentation')

#     def _wav2fbank(self, filename, filename2=None):
#         # mixup
#         if filename2 == None:
#             waveform, sr = torchaudio.load(filename)
#             waveform = waveform - waveform.mean()
#         # mixup
#         else:
#             waveform1, sr1 = torchaudio.load(filename)
#             waveform2, sr2 = torchaudio.load(filename2)
#             sr = sr1 

#             waveform1 = waveform1 - waveform1.mean()
#             waveform2 = waveform2 - waveform2.mean()
#             if self.verbose: 
#                 print(waveform1.shape, waveform2.shape, sr1, sr2)
        
#             if waveform1.shape[1] != waveform2.shape[1]:
#                 min_size = min(waveform1.shape[1], waveform2.shape[1])
#                 waveform1 = waveform1[:, 0:min_size]
#                 waveform2 = waveform2[:, 0:min_size]
#                 if self.verbose: 
#                     print('reshaping')
#                     print(waveform1.shape, waveform2.shape, sr1, sr2)

            
#             # sample lambda from beta distribtion
#             mix_lambda = np.random.beta(10, 10)

#             mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
#             waveform = mix_waveform - mix_waveform.mean()
#         # if len(waveform) > self.max_length_ms: 
#         #     crop_pos = random.randint(0, len(x) - self.max_length_ms - 1)
#         #     waveform =  waveform[crop_pos: crop_pos + crop_size]

#         fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
#                                                   window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

#         target_length = self.audio_conf.get('target_length')
#         n_frames = fbank.shape[0]

#         p = target_length - n_frames

#         # cut and pad
#         if p > 0:
#             m = torch.nn.ZeroPad2d((0, 0, 0, p))
#             fbank = m(fbank)
#         elif p < 0:
#             fbank = fbank[0:target_length, :]

#         if filename2 == None:
#             return fbank, 0
#         else:
#             return fbank, mix_lambda
#     def __getslice__(self, slices): 
#         if self.audio_mode == 'random_slice':
#             return np.random.choice(slices)
#         if self.audio_mode == 'first_slice': 
#             return slice[0]
        
#     def __getitem__(self, index):
#         """
#         returns: image, audio, nframes
#         where image is a FloatTensor of size (3, H, W)
#         audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
#         nframes is an integer
#         """
#         if self.audio_model == 'AST': 
#             # do mix-up for this sample (controlled by the given mixup rate)
#             if random.random() < self.mixup:
#                 datum = self.data.loc[index] 
#                 # find another sample to mix, also do balance sampling
#                 # sample the other sample from the uniform distribution
#                 mix_sample_idx = random.randint(0, len(self.data)-1)
#                 mix_datum = self.data.loc[mix_sample_idx]
#                 # get the mixed fbank
#                 # if self.dataset_audio_folder != "": 
#                 #     f1,f2 = os.path.join(self.dataset_audio_folder, datum['file_path'].split('/')[-1]), os.path.join(self.dataset_audio_folder, mix_datum['file_path'].split('/')[-1])
                
#                 f1 = self.__getslice__(datum[self.audio_key])
#                 f2 = self.__getslice__(mix_datum[self.audio_key])
#                 # f1, f2 = datum['file_path'].split('/')[-1], mix_datum['file_path'].split('/')[-1]

#                 # if self.verbose: print(f1,f2)

#                 fbank, mix_lambda = self._wav2fbank(f1, f2) 
#                 text = datum[f'{self.caption_mode}_caption']
#                 uri = datum[self.id_key]
#             # if not do mixup
#             else:
#                 datum = self.data.loc[index] #self.data[index]
#                 # if self.dataset_audio_folder != "": 
#                 #     f1 = os.path.join(self.dataset_audio_folder, datum['file_path'].split('/')[-1])
#                 # else: 
                
#                 f1 = self.__getslice__(datum[self.audio_key])
#                 fbank, mix_lambda = self._wav2fbank(f1) #self._wav2fbank(datum['wav'])
#                 text = datum[f'{self.caption_mode}_caption']
#                 uri = datum[self.id_key]

#             # SpecAug, not do for eval set
#             freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
#             timem = torchaudio.transforms.TimeMasking(self.timem)
#             fbank = torch.transpose(fbank, 0, 1)
#             # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
#             fbank = fbank.unsqueeze(0)
#             if self.freqm != 0:
#                 fbank = freqm(fbank)
#             if self.timem != 0:
#                 fbank = timem(fbank)
#             # squeeze it back, it is just a trick to satisfy new torchaudio version
#             fbank = fbank.squeeze(0)
#             fbank = torch.transpose(fbank, 0, 1)

#             # normalize the input for both training and test
#             if not self.skip_norm:
#                 fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
#             # skip normalization the input if you are trying to get the normalization stats.
#             else:
#                 pass

#             if self.noise == True:
#                 fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
#                 fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

#             mix_ratio = min(mix_lambda, 1-mix_lambda) / max(mix_lambda, 1-mix_lambda)

#             # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
#             return fbank, text, uri 
#         if self.audio_model == 'HTSAT': 
#             datum = self.data.loc[index] 
#             mp3_file = datum[self.audio_key]
#             waveform, sr = torchaudio.load(mp3_file)
#             text = datum[f'{self.caption_mode}_caption']
#             uri = datum[self.id_key]
#             # print('shapes', waveform.shape, uri)
#             if waveform.shape != torch.Size([1, 480000]): 
#                 print("ERROR", uri)
#                 pickle.dump(mp3_file, open("bad_files.pkl", 'wb'))
#             return waveform, text, uri 

#     def __len__(self):
#         # return 10 if self.split == "train" else len(self.data)
#         return len(self.data)

# class BenchmarkDatasets(): 
#     def __init__(self, args, config): 
#         self.config = config 
#         if args.dataset == 'MPD': 
#             self.dataset = Mp3Dataset(args, config)
#         if args.dataset == 'MusicCaps': 
#             self.dataset = MusicCapsDataset(args)
#         if args.distributed: 
#             num_tasks = get_world_size()
#             global_rank = get_rank() 
#             self.sampler = torch.utils.data.DistributedSampler(self.dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
#             self.dataloader = DataLoader(self.dataset, batch_size=self.config[args.audio_model]['batch_size'], num_workers=self.config[args.base_model]['n_worker'], pin_memory=True, sampler=self.sampler, shuffle=False, collate_fn=None,drop_last=False)   
#         else: 
#             self.dataloader = DataLoader(self.dataset, batch_size=self.config[args.base_audio]['batch_size'], num_workers=self.config[args.base_model]['n_worker'], pin_memory=True, shuffle=False, collate_fn=None,drop_last=False)

# class Datasets(): 
#     def __init__(self, args, config, split=True): 
#         self.train_dataset = WaveformDataset(args, config, split='train')
#         self.valid_dataset = WaveformDataset(args, config, split='valid')
#         self.test_dataset  = WaveformDataset(args, config, split='test')
#         self.config = config['MODELS']['BASE_MODELS']['BLAP'] 
#         self.audio_config = config['MODELS']['AUDIO_MODELS'][args.audio_model.upper()]
#         batch_size = min(self.audio_config['batch_size'], config['cmdline']['batch_size'])
#         if args.distributed: 
#             if args.verbose: 
#                 print("loading distributed datalaoders")
#             num_tasks = get_world_size()
#             global_rank = get_rank()  
#             self.train_sampler = torch.utils.data.DistributedSampler(self.train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
#             self.valid_sampler = torch.utils.data.DistributedSampler(self.valid_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
#             self.test_sampler = torch.utils.data.DistributedSampler(self.test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
#             self.train_dataloader = DataLoader(self.train_dataset, batch_size=config['cmdline']['batch_size'], num_workers=self.config['n_worker'], pin_memory=False, sampler=self.train_sampler, shuffle=False, collate_fn=None,drop_last=True)              
#             self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=config['cmdline']['batch_size'], num_workers=self.config['n_worker'], pin_memory=True,  shuffle=False, collate_fn=None,drop_last=False)   
#             self.test_dataloader = DataLoader(self.test_dataset, batch_size=config['cmdline']['batch_size'], num_workers=self.config['n_worker'], pin_memory=True, shuffle=False, collate_fn=None,drop_last=False)


#         else: 
#             self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.audio_config['batch_size'], num_workers=self.config['n_worker'], pin_memory=True, shuffle=True, collate_fn=None,drop_last=True)              
#             self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.audio_config['batch_size'], num_workers=self.config['n_worker'], pin_memory=True,  shuffle=False, collate_fn=None,drop_last=False)   
#             self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.audio_config['batch_size'], num_workers=self.config['n_worker'], pin_memory=True, shuffle=False, collate_fn=None,drop_last=False)

  

# #LOADING MODEL/FEATURES --DEPRECATED
# def load_model(args, config, checkpoint=None): 
#     if args.base_model == 'CLAP': 
#         model = laion_clap.CLAP_Module(enable_fusion=False)
#         model.load_ckpt()
#         model = model.to(args.device)
#         return model 


def generate_features(args, config, model): 
    if args.base_model == 'CLAP': 
        dataset_path = config[args.dataset]['org_filepath']
        feature_set = benchmarks.clap.clap_features(args, model, dataset_path)
        return feature_set


class EarlyStopping_OLD: #### DEPRECATED #### 
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Taken from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, checkpoint_path=None, patience=7, verbose=False, delta=0, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None 
        self.early_stop = False
        self.val_score = np.Inf
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.trace_func = trace_func

    def __call__(self, score, model, optimizer, epoch):
        if self.best_score is None: #INIT
            self.best_score = score
            self.best_epoch = epoch
            self.score_min = score
        elif score < self.best_score + self.delta: #INCREASE PATIENCE
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch 
            self.score_min = score
            self.counter = 0

def crop_wav(self, x):
        crop_size = self.audio_cfg["crop_size"]
        crop_pos = random.randint(0, len(x) - crop_size - 1)
        return x[crop_pos: crop_pos + crop_size]

class WDSDatasets():
    def __init__(self, args, config):  
        self.config = config['MODELS']['BASE_MODELS']['CLAP']
        self.train_dataloader = WDSDataset(args.batch_size, self.config['n_workers']).get_dataloader(args.seed, self.config['wds_dataset_path'], split='train')
        self.test_dataloader = WDSDataset(args.batch_size, self.config['n_workers']).get_dataloader(args.seed, self.config['wds_dataset_path'], split='test')

#CLAP dataloaders & its helper functions 
_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

def int16_to_float32_torch(x):
    return (x / 32767.0).type(torch.float32)

def float32_to_int16_torch(x):
    x = torch.clamp(x, min=-1., max=1.)
    return (x * 32767.).type(torch.int16)

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True

def get_dataset_size(shards, sizefilepath_, is_local=True):
    sizes = json.load(open(sizefilepath_, "r"))
    total_size = sum([int(v) for k,v in sizes.items()])
    num_shards = len(shards)
    return total_size, num_shards

class WDSDataset(): 
    def __init__(self, batch_size, n_workers): 
            self.num_batches = 0
            self.batch_size = 50
            self.workers = 8
            self.max_len = 480000 
            self.world_size = dist.get_world_size()
    
    def get_audio_features(self, audio_data): 
        if len(audio_data) > self.max_len:
            # random crop to max_len (for compatibility)
            overflow = len(audio_data) - self.max_len
            idx = np.random.randint(0, overflow + 1)
            audio_data = audio_data[idx: idx + self.max_len]

        else:  # padding if too short
            if len(audio_data) < self.max_len:  # do nothing if equal
                n_repeat = int(self.max_len / len(audio_data))
                audio_data = audio_data.repeat(n_repeat)
                audio_data = F.pad(
                    audio_data,
                    (0, self.max_len - len(audio_data)),
                    mode="constant",
                    value=0,
                )
        return audio_data 

    def preprocess(self, sample): 
        raw_audio, orig_sr = sample['flac']
        raw_audio = int16_to_float32_torch(float32_to_int16_torch(raw_audio[0]))
        waveform = self.get_audio_features(raw_audio).unsqueeze(dim=0)
        uri = sample['json']['uri']
        text = random.choice(sample['json']["text"])
        return waveform, text, uri

    def collate(self, batch):
        waveforms, texts, uris = [], [], [] 
        for sample in batch:
            waveform, text, uri = self.preprocess(sample)
            waveforms.append(waveform)
            texts.append(text)
            uris.extend(uri)
        waveforms = torch.concat(waveforms)
        return waveforms, texts, uris 
    
    def get_dataloader(self, seed, path, split):
        world_size = dist.get_world_size()

        input_shards = glob(f'{path}{split}/*.tar')
        sizefilepath_ = f'{path}{split}/sizes.json'
        num_samples, num_shards = get_dataset_size(input_shards, sizefilepath_)
        pipeline = [wds.SimpleShardList(input_shards)]
        if split == 'train': 
            pipeline.extend(
            [   
                wds.detshuffle(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=seed,
                ),
                wds.split_by_node,
                wds.split_by_worker,
                
                wds.tarfile_to_samples(handler=log_and_continue),
                wds.shuffle(
                    bufsize=_SAMPLE_SHUFFLE_SIZE,
                    initial=_SAMPLE_SHUFFLE_INITIAL,
                    rng=random.Random(seed),
                ),
                # wds.repeatedly,  # FIXME determine if this is beneficial
            ])
        else:
            pipeline.extend(
                [
                    wds.split_by_worker,
                    # at this point, we have an iterator over the shards assigned to each worker
                    wds.tarfile_to_samples(handler=log_and_continue),
                ])

        pipeline.append(wds.decode(wds.torch_audio)) 
        pipeline.append(
            wds.batched(
                self.batch_size,
                partial=not(split == 'train'),
                collation_fn=partial(self.collate))) 

        dataset = wds.DataPipeline(*pipeline)
        if split=='train':
            global_batch_size = self.batch_size * self.world_size
            num_batches = math.ceil(num_samples / global_batch_size)
            num_workers = max(1, self.workers)
            num_worker_batches = math.ceil(
                num_batches / num_workers
            )  # per dataloader worker
            num_batches = num_worker_batches * num_workers
            num_samples = num_batches * global_batch_size
            dataset = dataset.with_epoch(
                num_worker_batches
            ) 
            prefetch_factor = max(2, self.batch_size // self.workers)
        else: 
            num_batches = math.ceil(num_samples / self.batch_size)
            prefetch_factor = 2
        
        dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=None,
        )
        self.num_batches = num_batches
        dataloader.num_batches = num_batches
        dataloader.num_samples = num_samples
        dataloader.length = num_batches
        print("# batches = ", num_batches)
        print("# samples = ", num_samples)

        return dataloader 



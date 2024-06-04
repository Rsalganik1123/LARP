# 1. datasets; 2. model (a complete life cycle): train -> valid -> test 

# DATASET

import torch 
from datetime import datetime 
from torch import nn
import torch.nn.functional as F
import scipy.sparse as sp 
import os 
import pickle 
import json
import numpy as np 
from tqdm import tqdm
import ipdb 
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import ruamel.yaml as yaml


#PATHS FOR FEATURES 
base_path = {
        "MPD": {
            "LARP": "/mnt/ssd1/rebecca/spotify_clean/final_blap_feat/BLAP/",
            "jukebox": "/mnt/ssd1/rebecca/spotify_clean/benchmark_feat/juke_feat/",
            "Resnet": "/mnt/ssd1/rebecca/spotify_clean/benchmark_feat/resnet_feat/",
            "MULE": "/mnt/ssd1/rebecca/spotify_clean/benchmark_feat/mule_feat/",
            "CLAP_PT": "/mnt/ssd1/rebecca/spotify_clean/benchmark_feat/clap_pt_feat/",
            "CLAP_FT": "/mnt/ssd1/rebecca/spotify_clean/benchmark_feat/clap_ft_feat/",
            "LARP_wtc": "/mnt/ssd1/rebecca/spotify_clean/final_blap_feat/BLAP/",
            "LARP_fusion": "/mnt/ssd1/rebecca/spotify_clean/final_blap_feat/BLAP_CIP_F/",
            "BLAP_CIC": "/mnt/ssd1/rebecca/spotify_clean/final_blap_feat/BLAP_CIC/"
        },
        "LFM":{
            "jukebox": "/mnt/ssd1/rebecca/lfm/benchmark_feat/juke_feat/",
            "Resnet": "/mnt/ssd1/rebecca/lfm/benchmark_feat/resnet_feat/",
            "MULE": "/mnt/ssd1/rebecca/lfm/benchmark_feat/mule_feat/",
            "CLAP_PT": "/mnt/ssd1/rebecca/lfm/benchmark_feat/clap_pt_feat/",
            "CLAP_FT": "/mnt/ssd1/rebecca/lfm/benchmark_feat/clap_ft_feat/",
            "LARP_fusion": "/mnt/ssd1/rebecca/lfm/final_blap_feat/BLAP_CIP/A5000/",
            # "LARP_fusion": "/mnt/ssd1/rebecca/lfm/final_blap_feat/BLAP_CIP/",
            "BLAP_CIC": "/mnt/ssd1/xhliu/lfm/",
        }
    }

def pairs2csr(pairs, shape):
    indice = np.array(pairs, dtype=np.int32)
    values = np.ones(len(pairs), dtype=np.float32)
    return sp.csr_matrix(
        (values, (indice[:, 0], indice[:, 1])), shape=shape)

def pairwise_sim(feat): 
    feat = feat / feat.norm(dim=1)[:, None]
    res = torch.mm(feat, feat.transpose(0,1))
    return torch.mean(res.cpu()) 
 
class FeatureSetBuilder(): 
    def __init__(self, args, config, audio_emb, caption_emb): 
        self.fusion_mode = args.feat_fusion
        self.audio_emb = audio_emb
        self.caption_emb = caption_emb
    def average_modalities(self, audio_emb, caption_emb): 
        return (self.audio_emb + self.caption_emb)/2 
    def __call__(self, audio_emb, caption_emb): 
        if self.fusion_mode == 'avg': 
            return (self.audio_emb + self.caption_emb)/2 
        if self.fusion_mode == 'concat': 
            return torch.stack([audio_emb, caption_emb], dim=1)
    
class SpotifyDataset():
    def __init__(self, args, config, mode):
        self.batch_size_test = config['EVALUATION']['BUNDLE']['batch_size']
        self.batch_size_train = config['EVALUATION']['BUNDLE']['batch_size_train']

        self.path = config['EVALUATION']['BUNDLE'][args.dataset]['train']        
        if mode == "test":
            session_name = args.session_name
        else:
            session_name = ""
        
        with open(os.path.join(self.path, f'{mode}_datasize{session_name}')) as f:
            self.num_bundles, self.num_items = [eval(i.strip()) for i in f.read().split(",")]
        # load uri2id:
        with open(os.path.join(self.path, f'{mode}_uri2id{session_name}'), "r") as f:
            self.uri2id = json.loads(f.read())

        # load data:
        if mode != 'train':
            self.b_i_pairs_i = np.load(os.path.join(self.path, f'bi_{mode}_input{session_name}.npy'))
            self.b_i_pairs_gt = np.load(os.path.join(self.path, f'bi_{mode}_gt{session_name}.npy'))
            self.b_i_graph_i = pairs2csr(self.b_i_pairs_i, (self.num_bundles, self.num_items))
            self.b_i_graph_gt = pairs2csr(self.b_i_pairs_gt, (self.num_bundles, self.num_items))
        else:
            self.b_i_pairs = np.load(os.path.join(self.path, f'bi_{mode}.npy'))
            self.b_i_graph = pairs2csr(self.b_i_pairs, (self.num_bundles, self.num_items))


        self.artist_mask=None
        self.num_artists = 0 
        self.num_genres = 0 
        
        if mode in ('test', 'valid'): 
            if mode == "test":
                b_a_pairs_gt = np.load(os.path.join(self.path, f'ba_test_gt.npy'))
                b_g_pairs_gt = np.load(os.path.join(self.path, f'bg_test_gt.npy'))

                self.num_artists = np.max(b_a_pairs_gt)
                self.num_genres = np.max(b_g_pairs_gt)

                b_a_graph_gt = pairs2csr(b_a_pairs_gt, (self.num_bundles, self.num_artists+1))
                b_g_graph_gt = pairs2csr(b_g_pairs_gt, (self.num_bundles, self.num_genres+1))
                
                self.artist_mask = np.load(os.path.join(self.path,'artist_mask_test.npy')) 
                self.genre_mask = np.load(os.path.join(self.path,'genre_mask_test.npy')) 
            else:
                self.artist_mask = self.genre_mask = b_g_graph_gt = b_a_graph_gt = None,
            
            # build dataloader
            self.bundle_test_data = BundleTestDataset(self.b_i_pairs_i, self.b_i_graph_i, self.b_i_pairs_gt, self.b_i_graph_gt,
                                                    self.num_bundles, self.num_items, 
                                                    b_a_graph_gt = b_a_graph_gt, artist_mask = self.artist_mask, 
                                                    b_g_graph_gt = b_g_graph_gt, genre_mask = self.genre_mask)
            self.test_loader = DataLoader(self.bundle_test_data, batch_size=self.batch_size_test, shuffle=False, num_workers=20)
        
        if mode == 'train':
            self.bundle_train_data = BundleTrainDataset(self.b_i_pairs, self.b_i_graph)
            self.train_loader = DataLoader(self.bundle_train_data, batch_size=self.batch_size_train, shuffle=True, num_workers=20)
    

class BundleTrainDataset(Dataset):
    def __init__(self, b_i_pairs, b_i_graph, neg_sample=1):
        self.b_i_pairs = b_i_pairs
        self.b_i_graph = b_i_graph
        self.num_items = b_i_graph.shape[1]
        self.neg_sample = neg_sample

    def __getitem__(self, index):
        bundle, pos_items = self.b_i_pairs[index]
        all_items = [pos_items]
        while True:
            i = np.random.randint(self.num_items)
            if self.b_i_graph[bundle, i] == 0 and not i in all_items:                                                          
                all_items.append(i)                                                                                                   
                if len(all_items) == self.neg_sample+1:                                                                               
                    break                                                                                                        

        return torch.LongTensor([bundle]), torch.LongTensor(all_items)

    def __len__(self):
        return len(self.b_i_pairs)

class BundleTestDataset(Dataset):
    def __init__(self, b_i_pairs_i, b_i_graph_i, b_i_pairs_gt, b_i_graph_gt, num_bundles, num_items, b_a_graph_gt = None, artist_mask=None, b_g_graph_gt = None, genre_mask=None, num_trunc=None, output_P=False):
        self.b_i_pairs_i = b_i_pairs_i
        self.b_i_graph_i = b_i_graph_i
        
        self.bundles_map = np.argwhere(self.b_i_graph_i.sum(axis=1)>0)[:,0].reshape(-1)
        self.b_i_pairs_gt = b_i_pairs_gt
        self.b_i_graph_gt = b_i_graph_gt
        self.b_a_graph_gt = b_a_graph_gt
        self.b_g_graph_gt = b_g_graph_gt
        
        self.num_bundles = num_bundles
        self.num_items = num_items
        
        self.len_max = int(self.b_i_graph_i.sum(axis=1).max())
        self.artist_mask = artist_mask
        self.genre_mask = genre_mask 

        self.num_trunc = num_trunc
        self.output_P = output_P

    def __getitem__(self, index):
        graph_index = self.bundles_map[index]
        b_i_i = torch.from_numpy(self.b_i_graph_i[graph_index].toarray()).squeeze()
        b_i_gt = torch.from_numpy(self.b_i_graph_gt[graph_index].toarray()).squeeze()    

        b_a_gt = b_g_gt = 0 
        if type(self.b_a_graph_gt) != None: 
            b_a_gt = torch.from_numpy(self.b_a_graph_gt[graph_index].toarray()).squeeze()        
        if type(self.b_g_graph_gt) != None: 
            b_g_gt = torch.from_numpy(self.b_g_graph_gt[graph_index].toarray()).squeeze()        

        indices = torch.argwhere(b_i_i)[:,0]
        pad_token = self.num_items
        seq_b_i_i = F.pad(indices, (0, self.len_max-len(indices)), value=pad_token)
        if self.num_trunc is not None:
            seq_b_i_i = seq_b_i_i[:, :self.num_trunc]
        if self.output_P:
            mask = (seq_b_i_i == pad_token).detach()
            seq_b_i_i = seq_b_i_i.masked_fill(mask, 0)
            length = torch.sum(1-mask.float(), dim=-1)
            N_max = mask.shape[1]
            return self.bundles_map[index], b_i_i, seq_b_i_i, mask, length, N_max, b_i_gt, b_a_gt, b_g_gt 

        return self.bundles_map[index], b_i_i, seq_b_i_i, b_i_gt, b_a_gt, b_g_gt 

    def __len__(self):
        return len(self.bundles_map)


class BundleTestDatasetP_Beccas(Dataset):
    def __init__(self, args, config, mode='test'):
        self.path = config['EVALUATION']['BUNDLE'][args.dataset][mode] 
        with open(os.path.join(self.path, f'{mode}_datasize_trunc')) as f:
            self.num_bundles, self.num_items = [eval(i.strip()) for i in f.read().split(",")]
        # load uri2id:
        with open(os.path.join(self.path, f'{mode}_uri2id_trunc'), "r") as f:
            self.uri2id = json.loads(f.read())
        self.b_i_pairs_i = np.load(os.path.join(self.path, f'bi_{mode}_input_trunc.npy'))
        self.b_i_graph_i = pairs2csr(self.b_i_pairs_i, (self.num_bundles, self.num_items))
        self.b_i_pairs_gt = np.load(os.path.join(self.path, f'bi_{mode}_gt_trunc.npy'))
        self.b_i_graph_gt = pairs2csr(self.b_i_pairs_gt, (self.num_bundles, self.num_items))
    
        self.bundles_map = np.argwhere(self.b_i_graph_i.sum(axis=1)>0)[:,0].reshape(-1)
        
        self.len_max = int(self.b_i_graph_i.sum(axis=1).max())
        
        self.num_trunc = args.num_truncate
        self.num_tracks = self.num_items
        self.num_playlists = self.num_bundles

    def __getitem__(self, index):
        # ipdb.set_trace() 
        graph_index = self.bundles_map[index]
        b_i_i = torch.from_numpy(self.b_i_graph_i[graph_index].toarray()).squeeze()
        b_i_gt = torch.from_numpy(self.b_i_graph_gt[graph_index].toarray()).squeeze()        

        indices = torch.argwhere(b_i_i)[:,0]
        pad_token = self.num_items
        seq_b_i_i = F.pad(indices, (0, self.len_max-len(indices)), value=pad_token)
        if self.num_trunc is not None:
            seq_b_i_i = seq_b_i_i[:self.num_trunc]
        
        mask = (seq_b_i_i == pad_token).detach()
        seq_b_i_i = seq_b_i_i.masked_fill(mask, 0)
        length = torch.sum(1-mask.float(), dim=-1)
        # N_max = mask.shape[1]
        return self.bundles_map[index], seq_b_i_i, mask, length

        
    def __len__(self):
        return len(self.bundles_map)



# MODEL: MF For DropoutNet #########################################

class SpotifyDataset4MF():
    def __init__(self, args, config, mode):
        self.batch_size_test = config['EVALUATION']['BUNDLE']['batch_size']
        self.batch_size_train = config['EVALUATION']['BUNDLE']['batch_size_train']

        self.path = config['EVALUATION']['BUNDLE'][args.dataset][mode]   

        if mode == "test":
            session_name = args.session_name
        else:
            session_name = ""     
        
        with open(os.path.join(self.path, f'{mode}_datasize{session_name}')) as f:
            self.num_bundles, self.num_items = [eval(i.strip()) for i in f.read().split(",")]
        # load uri2id:
        with open(os.path.join(self.path, f'{mode}_uri2id{session_name}'), "r") as f:
            self.uri2id = json.loads(f.read())

        # load data:
        if mode != 'train':
            b_i_pairs = np.load(os.path.join(self.path, f'bi_{mode}_input{session_name}.npy'))
            b_i_graph = pairs2csr(b_i_pairs, (self.num_bundles, self.num_items))
        else:
            b_i_pairs = np.load(os.path.join(self.path, f'bi_{mode}.npy'))
            b_i_graph = pairs2csr(b_i_pairs, (self.num_bundles, self.num_items))

        self.bundle_train_data = BundleTrainDataset(b_i_pairs, b_i_graph)
        self.train_loader = DataLoader(self.bundle_train_data, batch_size=self.batch_size_train, shuffle=True, num_workers=20)
    

def train_MF(args, config, device, mode): # train, valid, test
    # the pipline
    num_epoch = 10
    _lr = 0.005
    _decay_lr_every = 50
    _lr_decay = 0.1

    train_dataset = SpotifyDataset4MF(args, config, mode)
    # train:
    config["num_bundle"] = train_dataset.num_bundles
    config["num_items"] = train_dataset.num_items
    config["device"] = device
    model = MF(args, config).to(device)
    optimizer = torch.optim.SGD(model.parameters(), _lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=_decay_lr_every, gamma=_lr_decay)
    model.train()
    best_metrics = {}
    best_metrics = {}
    metrics_name = config['EVALUATION']['BUNDLE']['metrics']
    topks = config['EVALUATION']["topk"]
    for m in metrics_name: 
        best_metrics[m] = {}
        for topk in topks:
            best_metrics[m][topk] = [0, 0]
    for epoch in range(num_epoch):
        pbar = tqdm(enumerate(train_dataset.train_loader), total=len(train_dataset.train_loader))
        for batch_i, batch in pbar:
            optimizer.zero_grad()
            batch = [x.to(device) for x in batch]
            loss = model(batch)
            loss.backward()
            optimizer.step()
            pbar.set_description(
                " epoch: %d, %.5s" %(epoch, loss.detach().cpu().item())
                )
        scheduler.step()
    
    # save Preference Features
    path = "/mnt/ssd1/xhliu/preference_features_for_DropoutNet/"
    torch.save(model.U, f"{path}U_{mode}_{args.dataset}.pt")
    torch.save(model.V, f"{path}V_{mode}_{args.dataset}.pt")
    

class MF(nn.Module):
    def __init__(self, args, config):
        super(MF, self).__init__()
        self.emb_size = 64
        num_users = config['num_bundle']
        num_items = config['num_items']
        self.device = config['device']
        self.U = nn.Parameter(
            torch.FloatTensor(num_users, self.emb_size).to(self.device))
        nn.init.xavier_uniform_(self.U, gain=1)
        self.V = nn.Parameter(
            torch.FloatTensor(num_items, self.emb_size).to(self.device))
        nn.init.xavier_uniform_(self.V, gain=1)

    def forward(self, batch):
        uid, items = batch
        target = torch.tensor([1., 0.] * uid.shape[0]).to(self.device)
        uid = uid.repeat_interleave(2)
        vid = items.view(-1)
        u_emb = self.U[uid]
        v_emb = self.V[vid]
        
        preds = nn.functional.normalize(u_emb, dim=-1) * nn.functional.normalize(v_emb, dim=-1) 
        # preds = u_emb * v_emb
        preds = torch.sum(preds, 1)
        return torch.nn.MSELoss()(preds, target)

    # do not need evaluate

# MDDEL: DropoutNet ################################################

def eval_DropoutNet(args, config, device):
    # the pipline
    num_epoch = 20
    _lr = 0.005
    _decay_lr_every = 50
    _lr_decay = 0.1

    if args.base_model == "BLAP_CIC" and args.dataset == "LFM":
        _lr = 0.001
    if args.base_model == "LARP_fusion" and args.dataset == "LFM":
        _lr = 0.001

    train_dataset = SpotifyDataset(args, config, 'train')
    valid_dataset = SpotifyDataset(args, config, 'valid')
    test_dataset = SpotifyDataset(args, config, 'test')

    # train:
    config["num_bundle"] = train_dataset.num_bundles
    config["num_items"] = train_dataset.num_items
    config["device"] = device

    # load preference features:
    path = "/mnt/ssd1/xhliu/preference_features_for_DropoutNet/"
    p_feature = {}
    for mode in ['train','valid','test']:
        p_feature[mode] = {
            "U": torch.load(f"{path}U_{mode}_{args.dataset}.pt", map_location=torch.device('cpu')),
            "V": torch.load(f"{path}V_{mode}_{args.dataset}.pt", map_location=torch.device('cpu'))
        }
    # load content features:
    c_feature = {}
    datasets = {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset}
    datasets['test'+args.session_name] = datasets['test']
    # for mode in ['train','valid','test']:
    global base_path
    modes = ['train','test'+args.session_name]
    for mode in modes:
        path = base_path[args.dataset][args.base_model] + mode

        features = torch.load(f"{path}/features.pt", map_location=torch.device('cpu'))
        if os.path.exists(f"{path}/uris.pkl"):
            uris = pickle.load(open(f"{path}/uris.pkl", "rb"))
        else:
            uris = pickle.load(open(f"{path}/all_uri.pkl", "rb"))[0]
        uri2id = datasets[mode].uri2id
        orders = np.zeros(len(uri2id))
        for idx, uri in enumerate(uris):
            id = int(uri2id[uri])
            orders[id] = idx

        features = features[orders]
        c_feature[mode] = features

    model = DropoutNet(args, config).to(device)
    optimizer = torch.optim.SGD(model.parameters(), _lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=_decay_lr_every, gamma=_lr_decay)
    best_metrics = {}
    best_metrics = {}
    metrics_name = config['EVALUATION']['BUNDLE']['metrics']
    topks = config['EVALUATION']["topk"]
    for m in metrics_name: 
        best_metrics[m] = {}
        for topk in topks:
            best_metrics[m][topk] = 0
    best_epoch = 0
    early_stop = 10
    cold_item_features = p_feature['train']['U'].mean(dim=0).expand(test_dataset.num_items,-1)
    cold_user_faetures = p_feature['train']['V'].mean(dim=0).expand(test_dataset.num_bundles,-1)
    for epoch in range(num_epoch):
        model.train()
        model.set_features(p_feature['train']['U'], p_feature['train']['V'], c_feature['train'], ui_graph = train_dataset.b_i_graph)
        pbar = tqdm(enumerate(train_dataset.train_loader), total=len(train_dataset.train_loader))
        for batch_i, batch in pbar:
            optimizer.zero_grad()
            batch = [x.to(device) for x in batch]
            loss = model(batch)
            loss.backward()
            optimizer.step()
            pbar.set_description(
                " epoch: %d, %s" %(epoch, loss.detach().cpu())
                )
        scheduler.step()
        # valid:
        model.eval()

        model.set_features(cold_user_faetures, cold_item_features, c_feature[modes[-1]], ui_graph = test_dataset.b_i_graph_i, update_cold_start=False)
        metrics = test(args, config, model, test_dataset.test_loader, device)

        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(args.save_path, curr_time)
        log(args.save_path, metrics)

        if metrics['recall'][10] >= best_metrics['recall'][10]:
            best_metrics = metrics
            best_epoch = epoch
            print(f"[!] Find a better one: {metrics['recall'][10]:.5f}@10,{metrics['recall'][20]:.5f}@20,{metrics['recall'][40]:.5f}@40")
            print(metrics)
        # elif epoch - best_epoch >= early_stop:
        #     # Test:
        #     model.set_features(cold_user_faetures, cold_item_features, c_feature['test'], ui_graph = test_dataset.b_i_graph_i, update_cold_start=False)
        #     test_metrics = test(args, config, model, test_dataset.test_loader, device)
        #     print("[TEST RESULT]:")
        #     curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #     # for topk in topks: 
        #     #     str_ = "%s, TOP %d: REC_T=%.5f, NDCG_T=%.5f" %(curr_time, topk, test_metrics["recall"][topk], test_metrics["ndcg"][topk])
        #     #     print(str_)
        #     print(test_metrics)

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

@torch.no_grad()
def init_weights(net):
    if type(net) == nn.Linear:
        #torch.nn.init.normal_(net.weight, mean=0, std=0.01)
        truncated_normal_(net.weight, std=0.01)
        if net.bias is not None:
            torch.nn.init.constant_(net.bias, 0)

class DNN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(DNN, self).__init__()
        self.layer = nn.Linear(dim_in, dim_out)
        self.bn = nn.BatchNorm1d(
                num_features=dim_out,
                momentum=0.01,
                eps=0.001
                )

    def forward(self, x):
        out = self.layer(x)
        out = self.bn(out)
        out = torch.tanh(out)
        return out


class DropoutNet(nn.Module):

    def __init__(self, args, config):
        super(DropoutNet, self).__init__()
        self.dropout = 0.2
        self.rank_in = 64 # the dim of cf features
        self.phi_u_dim = self.phi_v_dim = {
           "jukebox": 4800,
           "Resnet": 2048,
           "MULE": 8640, 
           "LARP": 256,
           "CLAP_PT": 512,
           "CLAP_FT": 512,
           "LARP_wtc": 256,
           "LARP_fusion": 256,
           "BLAP_CIC": 256,
        }[args.base_model]
        self.model_select = [512, 256, 256]
        self.rank_out = 256
        self.device = config["device"]

        u_dim = self.rank_in + self.phi_u_dim if self.phi_u_dim > 0 else self.rank_in
        v_dim = self.rank_in + self.phi_v_dim if self.phi_v_dim > 0 else self.rank_in
        
        # u_dim = self.phi_u_dim
        # v_dim = self.phi_v_dim

        u_dims = [u_dim] + self.model_select
        v_dims = [v_dim] + self.model_select
        
        self.p_f = nn.Linear(self.rank_in, self.rank_in)
        self.c_f = nn.Linear(self.phi_u_dim, self.phi_u_dim)

        self.u_layers = nn.ModuleList(DNN(u_dims[i], u_dims[i + 1]) for i in range(len(u_dims) - 1))
        self.v_layers = nn.ModuleList(DNN(v_dims[i], v_dims[i + 1]) for i in range(len(v_dims) - 1))
        
        self.u_emb = nn.Linear(u_dims[-1], self.rank_out)
        self.v_emb = nn.Linear(v_dims[-1], self.rank_out)

        self.apply(init_weights)

    def set_features(self, u_preference, v_preference, content_features, ui_graph = None, update_cold_start = False):
        user_preference_features = u_preference.float().detach().to(self.device)
        item_preference_features = v_preference.float().detach().to(self.device)
        self.item_content_features = content_features.detach().to(self.device)
        if ui_graph is not None:
            rowsum_sqrt = sp.diags(1/(ui_graph.sum(axis=1).A.ravel() + 1e-8))
            graph = rowsum_sqrt @ ui_graph
            graph = graph.tocoo()
            values = graph.data
            indices = np.vstack((graph.row, graph.col))
            graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape)).to(self.device)
            self.user_content_features = torch.spmm(graph, self.item_content_features)
        if update_cold_start:
            self.user_preference_features = torch.spmm(graph, item_preference_features)

            colsum_sqrt = sp.diags(1/(ui_graph.sum(axis=0).A.ravel() + 1e-8))
            graph2 = ui_graph @ colsum_sqrt
            graph2 = graph2.tocoo()
            values = graph2.data
            indices = np.vstack((graph2.row, graph2.col))
            graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph2.shape)).to(self.device)
            self.item_preference_features = torch.spmm(graph.T, user_preference_features)
        else:
            self.item_preference_features = item_preference_features
            self.user_preference_features = user_preference_features

    def encode(self, Uin, Vin, Ucontent, Vcontent):
        return self.encode_users(Uin, Ucontent), self.encode_items(Vin, Vcontent)

    def encode_items(self, Vin, Vcontent):
        # v_out = torch.cat((self.p_f(Vin), self.c_f(Vcontent)), 1)

        v_out = torch.cat((Vin, Vcontent), 1)
        # v_out = self.c_f(Vcontent)

        for layer in self.v_layers:
            v_out = layer(v_out)
        V_embedding = self.v_emb(v_out)
        return V_embedding

    def encode_users(self, Uin, Ucontent):
        # u_out = torch.cat((self.p_f(Uin), self.c_f(Ucontent)), 1)

        u_out = torch.cat((Uin, Ucontent), 1)

        # u_out = self.c_f(Ucontent)
        
        for layer in self.u_layers:
            u_out = layer(u_out)
        U_embedding = self.u_emb(u_out)
        return U_embedding
        
    def forward(self, batch):
        self.V_embedding = None
        uid, items = batch
        target = torch.tensor([1., 0.] * uid.shape[0]).to(device)
        uid = uid.repeat_interleave(2)
        vid = items.view(-1) # only positive
        Uin = self.user_preference_features[uid]
        Vin = self.item_preference_features[vid]

        Ucontent = self.user_content_features[uid]
        Vcontent = self.item_content_features[vid]
        if self.dropout > 0:
            indices = torch.arange(Uin.shape[0]).long().to(device)
            num_to_select = int(indices.shape[0] * self.dropout)
            u_indices = torch.randperm(indices.numel())[:num_to_select]
            v_indices = torch.randperm(indices.numel())[:num_to_select]
            Uin[u_indices].zero_()
            Vin[v_indices].zero_()
        
        U_embedding, V_embedding = self.encode(Uin, Vin, Ucontent, Vcontent)
        
        preds = U_embedding * V_embedding
        # preds = nn.functional.normalize(U_embedding, dim=-1) * nn.functional.normalize(V_embedding, dim=-1) 

        preds = torch.sum(preds, 1)
        return torch.nn.MSELoss()(preds, target)

    @torch.no_grad()
    def evaluate(self, batch):
        if self.V_embedding is None:
            self.features = self.V_embedding = self.encode_items(self.item_preference_features, self.item_content_features)
            
        b_id, seed_tracks, _ = batch # [bs, #tracks], e.g., [[0,1,0,0,1,...,0], ... ]]
        
        Uin = self.user_preference_features[b_id]
        Ucontent = self.user_content_features[b_id]
        
        # summation of tracks'features
        users_feat = self.encode_users(Uin, Ucontent)
        # retrival
        scores = users_feat @ self.V_embedding.T # [bs, d] x [d, #tracks] -> [bs, #tracks]

        return scores


# MDDEL: CLCRec ################################################


def eval_CLCRec(args, config, device):
    # the pipline
    num_epoch = 20
    _lr = 0.001
    if args.base_model == "CLAP_PT":
        _lr = 0.01
    if args.base_model == "BLAP_CIC" and args.dataset == "LFM":
        _lr = 0.0001
    if args.base_model == "LARP_fusion" and args.dataset == "LFM":
        _lr = 0.0002

    train_dataset = SpotifyDataset(args, config, 'train')
    valid_dataset = SpotifyDataset(args, config, 'valid')
    test_dataset = SpotifyDataset(args, config, 'test')

    # train:
    config["num_bundles"] = train_dataset.num_bundles
    config["num_items"] = train_dataset.num_items
    config["device"] = device

    # load preference features:
    path = "/mnt/ssd1/xhliu/preference_features_for_DropoutNet/"
    p_feature = {}
    for mode in ['train','valid','test']:
        p_feature[mode] = {
            "U": torch.load(f"{path}U_{mode}_{args.dataset}.pt"),
            "V": torch.load(f"{path}V_{mode}_{args.dataset}.pt")
        }
    # load content features:
    c_feature = {}
    datasets = {'train': train_dataset, 'valid': valid_dataset, 'test'+args.session_name: test_dataset}
    global base_path 
    modes = ['train','test'+args.session_name]
    for mode in modes:
        path = base_path[args.dataset][args.base_model] + mode

        features = torch.load(f"{path}/features.pt", map_location=torch.device('cpu'))
        if os.path.exists(f"{path}/uris.pkl"):
            uris = pickle.load(open(f"{path}/uris.pkl", "rb"))
        else:
            uris = pickle.load(open(f"{path}/all_uri.pkl", "rb"))[0]
        uri2id = datasets[mode].uri2id
        orders = np.zeros(len(uri2id))
        for idx, uri in enumerate(uris):
            id = int(uri2id[uri])
            orders[id] = idx

        features = features[orders]
        c_feature[mode] = features

    model = CLCRec(args, config).to(device)
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': _lr}])
    best_metrics = {}
    best_metrics = {}
    metrics_name = config['EVALUATION']['BUNDLE']['metrics']
    topks = config['EVALUATION']["topk"]
    for m in metrics_name: 
        best_metrics[m] = {}
        for topk in topks:
            best_metrics[m][topk] = 0
    best_epoch = 0
    early_stop = 10
    cold_item_features = p_feature['train']['U'].mean(dim=0).expand(test_dataset.num_items,-1)
    cold_user_faetures = p_feature['train']['V'].mean(dim=0).expand(test_dataset.num_bundles,-1)
    for epoch in range(num_epoch):
        model.train()
        model.set_features(p_feature['train']['U'], p_feature['train']['V'], c_feature['train'], ui_graph = train_dataset.b_i_graph)
        pbar = tqdm(enumerate(train_dataset.train_loader), total=len(train_dataset.train_loader))
        for batch_i, batch in pbar:
            optimizer.zero_grad()
            batch = [x.to(device) for x in batch]
            loss = model(batch)
            loss.backward()
            optimizer.step()
            pbar.set_description(
                " epoch: %d, %s" %(epoch, loss.detach().cpu())
                )
        # valid:
        model.eval()

        model.set_features(cold_user_faetures, cold_item_features, c_feature[modes[-1]], ui_graph = test_dataset.b_i_graph_i, update_cold_start=False)
        metrics = test(args, config, model, test_dataset.test_loader, device)

        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(args.save_path, curr_time)
        log(args.save_path, metrics)

        if metrics['recall'][10] >= best_metrics['recall'][10]:
            best_metrics = metrics
            best_epoch = epoch
            print(f"[!] Find a better one: {metrics['recall'][10]:.5f}@10,{metrics['recall'][20]:.5f}@20,{metrics['recall'][40]:.5f}@40")
            print(metrics)
        # elif epoch - best_epoch >= early_stop:
        #     # Test:
        #     model.set_features(cold_user_faetures, cold_item_features, c_feature['test'], ui_graph = test_dataset.b_i_graph_i, update_cold_start=False)
        #     test_metrics = test(args, config, model, test_dataset.test_loader, device)
        #     print("[TEST RESULT]:")
        #     curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #     print(test_metrics)

infonce_criterion = nn.CrossEntropyLoss()

class CLCRec(nn.Module):
    def __init__(self, args, configs):
        super(CLCRec, self).__init__()
        self.num_user = configs["num_bundles"]
        self.num_item = configs["num_items"]
        self.num_warm_item = configs["num_items"]
        self.device = configs["device"]
        self.num_neg = 128
        self.lr_lambda = 0.5
        self.reg_weight = 0.1
        self.temp_value = 2.0
        self.dim_E = 64
        self.user_id_embedding = nn.Parameter(nn.init.xavier_normal_(torch.rand((self.num_user, self.dim_E))))
        self.item_id_embedding = nn.Parameter(nn.init.xavier_normal_(torch.rand((self.num_item, self.dim_E))))
        self.dim_feat = {
           "jukebox": 4800,
           "Resnet": 2048,
           "MULE": 8640,
           "LARP": 256,
           "CLAP_PT": 512,
           "CLAP_FT": 512,
           "LARP_wtc": 256,
           "LARP_fusion": 256,
           "BLAP_CIC": 256,
        }[args.base_model]
        self.num_sample = 0.5

        if args.base_model == "LARP_fusion" and args.dataset == "LFM":
            self.reg_weight = 0.02
            self.temp_value = 3

        if args.base_model == "CLAP_PT":
            self.reg_weight = 0.1
            self.temp_value = 0.2
        
        self.MLP = nn.Linear(self.dim_E, self.dim_E)

        self.encoder_layer1 = nn.Linear(self.dim_feat, 256)
        self.encoder_layer2 = nn.Linear(256, self.dim_E)
        
        self.att_weight_1 = nn.Parameter(nn.init.kaiming_normal_(torch.rand((self.dim_E, self.dim_E))))
        self.att_weight_2 = nn.Parameter(nn.init.kaiming_normal_(torch.rand((self.dim_E, self.dim_E))))
        self.bias = nn.Parameter(nn.init.kaiming_normal_(torch.rand((self.dim_E, 1))))
        self.att_sum_layer = nn.Linear(self.dim_E, self.dim_E)

        # self.result = nn.init.xavier_normal_(torch.rand((self.num_user+self.num_item, self.dim_E))).cuda()

    def set_features(self, u_preference, v_preference, content_features, ui_graph = None, update_cold_start = False):
        user_preference_features = u_preference.float().detach().to(self.device)
        item_preference_features = v_preference.float().detach().to(self.device)
        self.item_content_features = content_features.detach().to(self.device)
        self.ui_graph = ui_graph
        # if ui_graph is not None:
        #     rowsum_sqrt = sp.diags(1/(ui_graph.sum(axis=1).A.ravel() + 1e-8))
        #     graph = rowsum_sqrt @ ui_graph
        #     graph = graph.tocoo()
        #     values = graph.data
        #     indices = np.vstack((graph.row, graph.col))
        #     graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape)).to(self.device)
        #     self.user_content_features = torch.spmm(graph, self.item_content_features)

    def encoder(self, mask=None):
        feature = F.normalize(self.item_content_features, dim=-1)

        feature = F.leaky_relu(self.encoder_layer1(feature))
        feature = self.encoder_layer2(feature)
        return feature

    def loss_contrastive(self, a, b, temp_value):      
        # a = nn.functional.normalize(a, dim=-1)
        # b = nn.functional.normalize(b, dim=-1)
        logits = torch.mm(a, b.T)
        logits /= temp_value
        labels = torch.arange(a.shape[0]).to(a.device)
        return infonce_criterion(logits, labels)

    def forward(self, batch):
        uid, items = batch
        uid = uid[:, 0]
        items = items[:, 0]

        self.user_cold_features = self.item_cold_features = None

        feature = self.encoder()
        all_item_feat = feature[items]

        user_embedding = self.user_id_embedding[uid]
        all_item_embedding = self.item_id_embedding[items]
        
        head_feat = F.normalize(all_item_feat, dim=1)
        head_embed = F.normalize(all_item_embedding, dim=1)

        all_item_input = all_item_embedding.clone()
        rand_index = torch.randint(all_item_embedding.size(0), (int(all_item_embedding.size(0)*self.num_sample), )).cuda()
        all_item_input[rand_index] = all_item_feat[rand_index].clone()

        self.contrastive_loss_1 = self.loss_contrastive(head_embed, head_feat, self.temp_value)
        self.contrastive_loss_2 = self.loss_contrastive(user_embedding, all_item_input, self.temp_value)

        reg_loss = ((torch.sqrt((user_embedding**2).sum(1))).mean()+(torch.sqrt((all_item_embedding**2).sum(1))).mean())/2

        return self.contrastive_loss_1*self.lr_lambda+(self.contrastive_loss_2)*(1-self.lr_lambda) + self.reg_weight * reg_loss

        # self.result = torch.cat((self.id_embedding[:self.num_user+self.num_warm_item], feature[self.num_warm_item:]), dim=0)

    @torch.no_grad()
    def evaluate(self, batch):
        if self.user_cold_features is None:
            self.item_cold_features = self.features = self.encoder()
            rowsum_sqrt = sp.diags(1/(self.ui_graph.sum(axis=1).A.ravel() + 1e-8))
            graph = rowsum_sqrt @ self.ui_graph
            graph = graph.tocoo()
            values = graph.data
            indices = np.vstack((graph.row, graph.col))
            graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape)).to(self.device)
            self.user_cold_features = torch.spmm(graph, self.item_cold_features)

        b_id, seed_tracks, _ = batch # [bs, #tracks], e.g., [[0,1,0,0,1,...,0], ... ]]
        
        users_feat = self.user_cold_features[b_id]
        
        scores = users_feat @ self.item_cold_features.T # [bs, d] x [d, #tracks] -> [bs, #tracks]

        return scores


## itemKNN ######



def eval_itemKNN(args, config, device):
    # the pipline
    num_epoch = 20
    _lr = 0.005
    _decay_lr_every = 50
    _lr_decay = 0.1

    test_dataset = SpotifyDataset(args, config, 'test')

    # train:
    config["device"] = device

    # load preference features:
    path = "/mnt/ssd1/xhliu/preference_features_for_DropoutNet/"
    c_feature = {}
    datasets = {'test': test_dataset}
    global base_path
    modes = ['test'+args.session_name]
    datasets = {modes[-1]: test_dataset}
    for mode in modes:
        path = base_path[args.dataset][args.base_model] + mode

        features = torch.load(f"{path}/features.pt", map_location=torch.device('cpu'))
        if os.path.exists(f"{path}/uris.pkl"):
            uris = pickle.load(open(f"{path}/uris.pkl", "rb"))
        else:
            uris = pickle.load(open(f"{path}/all_uri.pkl", "rb"))
        uri2id = datasets[mode].uri2id
        orders = np.zeros(len(uri2id))
        for idx, uri in enumerate(uris):
            id = int(uri2id[uri])
            orders[id] = idx

        features = features[orders]
        c_feature[mode] = features

    model = NearestNeighbour(args, config).to(device)
    topks = config['EVALUATION']["topk"]
    model.set_features(c_feature[modes[-1]])
    metrics = test(args, config, model, test_dataset.test_loader, device)

    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(args.save_path, curr_time)
    log(args.save_path, metrics)
    print(curr_time, metrics)


class NearestNeighbour(nn.Module):
    def __init__(self, args, config):
        super(NearestNeighbour, self).__init__()

    def set_features(self, features):
        self.features = features
        print("dim: ", self.features.shape)
    
    def evaluate(self, batch):
        b_id, seed_tracks, _ = batch # [bs, #tracks], e.g., [[0,1,0,0,1,...,0], ... ]]
        
        # summation of tracks'features
        bundles_feat = seed_tracks @ self.features # [bs, #tracks] x [#tracks, d] -> [bs, d], bs is the number of bundles
        # retrival
        scores = bundles_feat @ self.features.T # [bs, d] x [d, #tracks] -> [bs, #tracks]
        return scores

# TEST UTILS ##########################################
 
def test(args, config, model, dataloader, device):
    
    tmp_metrics = {}
    for m in config['EVALUATION']['BUNDLE']['metrics']: 
        tmp_metrics[m] = {}
        for topk in config['EVALUATION']["topk"]:
            tmp_metrics[m][topk] = [0, 0]
    
    artist_mask = None 
    if type(dataloader.dataset.artist_mask) != None: 
        artist_mask = torch.FloatTensor(dataloader.dataset.artist_mask.T).to(device) 
    
    genre_mask = None 
    if type(dataloader.dataset.genre_mask) != None: 
        genre_mask = torch.FloatTensor(dataloader.dataset.genre_mask.T).to(device) 
        
        
    # device = args.device
    model.eval()

    show_progress = True 
    with tqdm(desc=f'running eval', unit='it', total=len(dataloader), disable=show_progress) as pbar:
        # ipdb.set_trace()
        for batch_i, (index, b_i_input, seq_b_i_input, b_i_gt, b_a_gt, b_g_gt ) in enumerate(dataloader):
            pred_i = model.evaluate((index.to(device), b_i_input.to(device), seq_b_i_input.to(device)))
            pred_i = pred_i - 1e8 * b_i_input.to(device) # mask
            tmp_metrics = get_track_metrics(tmp_metrics, b_i_gt.to(device), pred_i, config['EVALUATION']["topk"])
            # if type(artist_mask) != None: 
            #     pred_a = pred_i @ artist_mask 
            #     a_metrics = get_artist_metrics(b_a_gt.to(device), pred_a, config['EVALUATION']["topk"])
            #     tmp_metrics.update(a_metrics)
            # if type(genre_mask) != None: 
            #     pred_g = pred_i @ genre_mask 
            #     g_metrics = get_genre_metrics(b_g_gt.to(device), pred_g, config['EVALUATION']["topk"])
            #     tmp_metrics.update(g_metrics)
            f_metrics = get_flow_metrics(model.features, pred_i, config['EVALUATION']["topk"])
            tmp_metrics.update(f_metrics)
            pbar.update()
    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]
            
    return metrics

def get_flow_metrics(features, pred, topks): 
    tmp = {"flow": {}} 
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        tmp["flow"][topk] = (torch.mean(torch.stack([pairwise_sim(x_i) for x_i in torch.unbind(features[col_indice, :], dim=0)], dim=0)).item(), col_indice.shape[0])   
    return tmp

def get_genre_metrics(grd, pred, topks): 
    tmp = {"genre_recall": {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)
        tmp["genre_recall"][topk] = get_recall(pred, grd, is_hit, topk)
    return tmp 
        
def get_artist_metrics(grd, pred, topks): 
    tmp = {"artist_recall": {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)

        tmp["artist_recall"][topk] = get_recall(pred, grd, is_hit, topk)
        
    return tmp

def get_track_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}
    # ipdb.set_trace() 
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)
        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk) 

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x
    
    return metrics

def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt/(num_pos+epsilon)).sum().item()

    return [nomina, denorm]

def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit/torch.log2(torch.arange(2, topk+2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float).to(device)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1+topk, dtype=torch.float).to(device)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk+1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg/idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]


def log(save_path, text, p=False):
    with open(save_path, "a") as f:
        if p:
            print(text)
        f.write(str(text))
        f.write("\n")

if __name__ == "__main__":
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MPD', choices=['MPD', 'MUSIC_CAPS', "EMO", 'LFM'], help='dataset for training/eval')
    parser.add_argument('--eval_model', type=str, default='CLCRec', choices=['itemKNN', 'DropoutNet', "CLCRec"])
    parser.add_argument('--base_model', type=str, default='LARP', choices=['LARP', 'jukebox', "Resnet", "MULE", "CLAP_PT", "CLAP_FT", "LARP_wtc", "LARP_fusion", "BLAP_CIC"])
    parser.add_argument('--save_path', type=str, default='/mnt/ssd1/xhliu/blap_tb/evaluate_recommender_results')
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--session_name', type=str, default='')
    args = parser.parse_args()


    if args.dataset == "LFM": 
        args.session_name = "_trunc"

    config = yaml.safe_load(open('/home/rebecca/BLAP_test/configs/config2.yaml', 'r'))
    base_config = config['MODELS']['BASE_MODELS']['BLAP']
    audio_config = config['MODELS']['AUDIO_MODELS']['HTSAT']
    text_config = config['MODELS']['LANGUAGE_MODELS']['BERT']
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.gpu == '-1':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    config['cmdline'] = args.__dict__
    # train: 2048
    # valid: 5
    # test: 20
    curr_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    args.save_path = os.path.join(args.save_path, "_".join([ args.dataset, args.eval_model, args.base_model, curr_time, args.suffix]))+".log"
    log(args.save_path, curr_time)
    log(args.save_path, args)
    config['EVALUATION']['BUNDLE']['batch_size_train'] = 2048
    if args.eval_model == "MF":
        train_MF(args,config,device,'train')
    elif args.eval_model == "itemKNN":
        eval_itemKNN(args,config,device)
    elif args.eval_model == "DropoutNet":
        eval_DropoutNet(args,config,device)
    elif args.eval_model == "CLCRec":
        eval_CLCRec(args,config,device)

    # eval_itemKNN(args,config,device)

    # python3 evaluation/continuation_full.py --dataset MPD --eval_model DropoutNet --base_model jukebox



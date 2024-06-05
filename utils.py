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

#LARP utils 
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






import argparse
import os
import ruamel.yaml as yaml
# import yaml 
import numpy as np
import random
import time
from datetime import datetime
import json
from pathlib import Path
from itertools import product
import ipdb 
import csv
import pprint
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 

from parser_unified import parse_args
from utils import warmup_lr_schedule, step_lr_schedule, move_data_to_device, EarlyStopping, cosine_lr
import utils
# from scheduler_clap import cosine_lr
# from data_cip import Datasets
from data_unified import Datasets

from models.blip_unified import LARP, forward_and_backward # , playlist_feature_update
from models.clap import clap_audio 

import evaluation.continuation2 as continuation
from evaluation.continuation2 import SpotifyDataset, FeatureSetBuilder
from evaluation.crossmodal_retrieval import t2a, a2t

import scipy.sparse as sp

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'



def gather_features(args, model, dataloader, device):
    audio_emb, caption_emb, uris = [], [], []
    # print("MODEL", model)
    
    batch_cnt = len(dataloader )

    with tqdm(desc=f'loading emb', unit='it', total=batch_cnt, disable=not utils.is_main_process()) as pbar:
        for i, (audio, caption, uri) in enumerate(dataloader): 
            a, c = model.return_features(audio.to(device), caption)
            audio_emb.append(a)
            caption_emb.append(c)
            uris += list(uri)
            pbar.update() 
    audio_emb = torch.concatenate(audio_emb)
    caption_emb = torch.concatenate(caption_emb)
    
    return audio_emb, caption_emb, uris

def ret_evaluate(args, config, device, model, blap_dataset, mode, features=None, uris=None): 
    if features == None: 
        feature_dataloader = blap_dataset.valid_dataloader if mode == 'valid' else blap_dataset.test_dataloader
        audio_emb, caption_emb, uris = gather_features(args, model = model, dataloader = feature_dataloader, device=device)
        print(f'loaded: audio emb:{audio_emb.shape}, caption emb: {caption_emb.shape}')
     
    t2a_metrics = t2a(config, audio_emb, caption_emb)
    a2t_metrics = a2t(config, audio_emb, caption_emb)
    metrics = {**t2a_metrics, **a2t_metrics}
    # decision_metric = t2a_metrics[config['EVALUATION']['RETRIEVAL']['decision_metric']]
    decision_metric = t2a_metrics[config['EVALUATION']['RETRIEVAL']['decision_metric']][config['EVALUATION']['RETRIEVAL']['decision_index']]
    return t2a_metrics, decision_metric #, a2t_metrics

def rec_evaluate(args, config, device, model, blap_dataset, rec_dataset, mode, features = None, uris=None, save_feat=False): 
    if features == None: 
        feature_dataloader = blap_dataset.valid_dataloader if mode == 'valid' else blap_dataset.test_dataloader
        audio_emb, caption_emb, uris = gather_features(args, model = model, dataloader = feature_dataloader, device=device)
        # dist.barrier() 
        print(f'loaded: audio emb:{audio_emb.shape}, caption emb: {caption_emb.shape}')
        f = FeatureSetBuilder(args, config, audio_emb, caption_emb)
        features = f.average_modalities(audio_emb, caption_emb)

        # sort features
        if args.verbose: 
            print("reorder features >>>")
        uri2id = rec_dataset.uri2id
        orders = np.zeros(len(uri2id))
        for idx, uri in enumerate(uris):
            id = int(uri2id[uri])
            orders[id] = idx

        features = features[orders]
        if args.verbose:
            print("reorder features <<<")

        if save_feat: 
            torch.save(features, args.output_path)
    nn = continuation.NearestNeighbour(args, config, features=features)
    metrics = continuation.test(args, config, nn, rec_dataset.test_loader, device)
    decision_metric = metrics[config['EVALUATION']['BUNDLE']['decision_metric']][config['EVALUATION']['BUNDLE']['decision_index']]
    return metrics, decision_metric

def train(args, model, data_loader, optimizer, epoch, device, config, tensorboard_logger, scheduler=None): 
    model.train(True)  
    
    data_loader.sampler.set_epoch(epoch) #updates sampler for distributed setting otherwise same ordering each time uri
    batch_cnt = len(data_loader)
   
    epoch_anchor = epoch * batch_cnt
    disable_flag = not (dist.get_rank() == 0) 
    with tqdm(desc=f'training ({config["ablation_loss"]}), epoch {epoch}, thread {dist.get_rank()}', unit='it', total=batch_cnt, disable=not utils.is_main_process()) as pbar:
        for batch_i, (audio, caption, uri) in enumerate(data_loader):
            batch_anchor = epoch_anchor + batch_i
            scheduler(batch_anchor)
            optimizer.zero_grad()
            audio = audio.to(device)
                             
            alpha = float(config['alpha'])*min(1,(epoch*len(data_loader)+batch_i)/(2*len(data_loader))) 
            
            losses = forward_and_backward(model, config, (audio, caption), (None, None), (None, None, None, None), None, alpha = alpha, epoch=epoch)

            optimizer.step() 
        
            if dist.get_rank() == 0: 
                for k, v in losses.items(): 
                    tensorboard_logger.add_scalar(f'loss_{k}', v, batch_anchor)
                tensorboard_logger.add_scalar('lr', optimizer.param_groups[0]["lr"], batch_anchor)
                
            pbar.update()
        
    return {'TRAIN _STATS ': "empty"}

def train_TTC(args, model, data_loader, optimizer, epoch, device, config, tensorboard_logger, scheduler=None): 
    model.train(True)  
    
    data_loader.sampler.set_epoch(epoch) #updates sampler for distributed setting otherwise same ordering each time uri
    
    #housekeeping for logging 
    batch_cnt = len(data_loader)
    epoch_anchor = epoch * batch_cnt
    disable_flag = not (dist.get_rank() == 0) 
    with tqdm(desc=f'training ({config["ablation_loss"]}), epoch {epoch}, thread {dist.get_rank()}', unit='it', total=batch_cnt, disable=not utils.is_main_process()) as pbar:
        for batch_i, batch_data in enumerate(data_loader):
            
            batch_anchor = epoch_anchor + batch_i #logging
            scheduler(batch_anchor) #scheduling learning rate

            # print(batch_data)
            ego_audio_text_pairs, cross_audio_text_pairs, cross_mask = batch_data
            ego_audio, ego_caption, ego_uri = ego_audio_text_pairs
            cross_audio, cross_caption, cross_uri = cross_audio_text_pairs

            optimizer.zero_grad()
            ego_audio = torch.stack(ego_audio, dim=0).to(device, non_blocking=True)
            cross_audio = torch.stack(cross_audio, dim=0).to(device, non_blocking = True)
            cross_mask = torch.from_numpy(cross_mask).to(device)
            
            alpha = float(config['alpha'])*min(1,(epoch*len(data_loader)+batch_i)/(2*len(data_loader))) 
            
            losses = forward_and_backward(model, config, (ego_audio, ego_caption), (cross_audio, cross_caption), (None, None, None, None), cross_mask, alpha = alpha, epoch=epoch)

            optimizer.step() 
        
            if dist.get_rank() == 0: 
                for k, v in losses.items(): 
                    tensorboard_logger.add_scalar(f'loss_{k}', v, batch_anchor)
                tensorboard_logger.add_scalar('lr', optimizer.param_groups[0]["lr"], batch_anchor)
                
            pbar.update()
    return {'TRAIN _STATS ': "empty"} 
    
def train_TPC(args, model, data_loader, optimizer, epoch, device, config, tensorboard_logger, scheduler=None): 
    model.train(True)  
    
    data_loader.sampler.set_epoch(epoch) #updates sampler for distributed setting otherwise same ordering each time uri
    
    #housekeeping for logging 
    batch_cnt = len(data_loader)
    epoch_anchor = epoch * batch_cnt
    disable_flag = not (dist.get_rank() == 0) 
    with tqdm(desc=f'training ({config["ablation_loss"]}), epoch {epoch}, thread {dist.get_rank()}', unit='it', total=batch_cnt, disable=not utils.is_main_process()) as pbar:
        for batch_i, batch_data in enumerate(data_loader):
            
            batch_anchor = epoch_anchor + batch_i #logging
            scheduler(batch_anchor) #scheduling learning rate

            ego_audio_text_pairs, cross_audio_text_pairs, cross_mask, idx, seq, mask, length = batch_data
            ego_audio, ego_caption, ego_uri = ego_audio_text_pairs
            cross_audio, cross_caption, cross_uri = cross_audio_text_pairs

            optimizer.zero_grad()
            ego_audio = torch.stack(ego_audio, dim=0).to(device, non_blocking=True)
            cross_audio = torch.stack(cross_audio, dim=0).to(device, non_blocking = True)
            cross_mask = torch.from_numpy(cross_mask).to(device)
            idx = torch.LongTensor(idx).to(device, non_blocking=True)
            seq = torch.stack(seq, dim=0).to(device, non_blocking=True)
            mask = torch.stack(mask, dim=0).to(device, non_blocking=True)
            length = torch.stack(length, dim=0).to(device, non_blocking=True)

            alpha = float(config['alpha'])*min(1,(epoch*len(data_loader)+batch_i)/(2*len(data_loader))) 
            
            losses = forward_and_backward(model, config, (ego_audio, ego_caption), (cross_audio, cross_caption), (idx, seq, mask, length),  cross_mask, alpha = alpha, epoch=epoch)

            optimizer.step() 
        
            if dist.get_rank() == 0: 
                for k, v in losses.items(): 
                    tensorboard_logger.add_scalar(f'loss_{k}', v, batch_anchor)
                tensorboard_logger.add_scalar('lr', optimizer.param_groups[0]["lr"], batch_anchor)
                
            pbar.update()
    return {'TRAIN _STATS ': "empty"} 
    
def distributed_main(rank, world_size, args):
    # intialize distributed run
    rank, cuda_id = utils.get_gpu_id(rank, args) 
    utils.setup_ddp(rank, world_size, cuda_id, args.master_port)

    # preliminaries 
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = f'cuda:{cuda_id}'
    utils.setup_seed(args.seed + rank)

    #load parameters 
    config = yaml.safe_load(open('/home/rebecca/BLAP_test/configs/config2.yaml', 'r'))
    base_config = config['MODELS']['BASE_MODELS'][args.base_model]
    audio_config = config['MODELS']['AUDIO_MODELS'][args.audio_model]
    text_config = config['MODELS']['LANGUAGE_MODELS'][args.text_model]
    config['cmdline'] = args.__dict__

    audio_config['name'] = args.audio_model
    base_config['ablation_loss'] = config['cmdline']['ablation_loss']
    base_config['embed_dim'] =  config['cmdline']['embed_dim']
    base_config['fusion_method'] =  config['cmdline']['fusion_method']
    base_config['device'] = device

    #setup logging 
    log_path, run_path, checkpoint_model_path, checkpoint_conf_path, best_results_path, summary_path, all_experiments_path, prefix  = utils.setup_logging(args.__dict__)
    decision_metric_name, decision_index = config['EVALUATION'][args.decision_metric.upper()]['decision_metric'], config['EVALUATION'][args.decision_metric.upper()]['decision_index']
    early_stopper = EarlyStopping(config['cmdline']['early_stop'], decision_metric_name, decision_index)
    run = SummaryWriter(run_path)

    if dist.get_rank() == 0: 
        print(f'logging in {run_path}')
        yaml.dump(config, open(os.path.join(checkpoint_conf_path, "configs.yaml"), "w"))  
        print(f'config dumped to {os.path.join(checkpoint_conf_path, "configs.yaml")}')
    #load dataset
    
    dataset = Datasets(args, config)
    total_steps = len(dataset.train_dataloader) * base_config['n_epochs'] 
    valid_dataset = SpotifyDataset(args, config, 'test', device, session=args.session, trunc=args.trunc)

    # test_dataset = SpotifyDataset(args, config, 'test', device)

    #Define model & optimizer 
    base_config["num_tracks"] = dataset.train_dataset.num_tracks
    base_config["num_playlists"] = dataset.train_dataset.num_playlists
    model = LARP(config=base_config, audio_cfg=audio_config, text_cfg=text_config, embed_dim = base_config['embed_dim']).to(device)
    
    ddp_model = DDP(model, device_ids=[int(cuda_id)], find_unused_parameters=True)
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=float(1e-4), betas=(0.9,0.999), eps=1e-08) 

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']    
        model.load_state_dict(state_dict, strict=False)
        optimizer_state_dict = checkpoint['optimizer']
        optimizer.load_state_dict(optimizer_state_dict)

        print('resume checkpoint from %s'%args.checkpoint) 

    scheduler = cosine_lr(optimizer, float(1e-4), 3200, total_steps) #copied from CLAP 
    
    ##Training
    start_train = datetime.now() 

    # save for testing:
    utils.save_checkpoint(ddp_model, optimizer,config, 12345, checkpoint_model_path)
    for epoch in range(base_config['n_epochs']): 
        ##train one epoch 
        if args.mode == 'train': 
            assert base_config["ablation_loss"] == ""
            train_stats = train(args, ddp_model, dataset.train_dataloader, optimizer, epoch, device, base_config, run, scheduler=scheduler) 
        if args.mode == 'TTC': 
            assert "ttc" in base_config["ablation_loss"]
            train_stats = train_TTC(args, ddp_model, dataset.train_dataloader, optimizer, epoch, device, base_config, run, scheduler=scheduler)
        if args.mode == 'TPC': 
            assert "tpc" in base_config["ablation_loss"]
            train_stats = train_TPC(args, ddp_model, dataset.train_dataloader, optimizer, epoch, device, base_config, run, scheduler=scheduler)

        if dist.get_rank() == 0: 
            utils.log_stats(train_stats, 'TRAIN', epoch, log_path, eval_mode='retrieval')
            if epoch % args.save_every and args.save_every != -1:  #only save if master process
                utils.save_checkpoint(ddp_model, optimizer,config, epoch, checkpoint_model_path)

            if epoch % args.validation_interval == 0: #run validation 
                print("validation")
                if 'bundle' in args.eval_mode:
                     
                    rec_validation_stats, rec_decision_metric = rec_evaluate(args, config, device, ddp_model.module, dataset, valid_dataset, mode='test')
                    utils.log_stats(rec_validation_stats, 'test', epoch, log_path, eval_mode='bundle')
                    for metric_name in rec_validation_stats:
                        for topk in rec_validation_stats[metric_name]:
                            name = f"rec_valid/{metric_name}@{topk}"
                            val = rec_validation_stats[metric_name][topk]
                            run.add_scalar(name, val, epoch)
                if 'retrieval' in args.eval_mode: 
                    ret_validation_stats, ret_decision_metric = ret_evaluate(args, config, device, ddp_model.module, dataset, mode='test')
                    utils.log_stats(ret_validation_stats, 'test', epoch, log_path, eval_mode='retrieval')
                    for metric_name in ret_validation_stats:
                        for topk in ret_validation_stats[metric_name]:
                            name = f"ret_valid/{metric_name}@{topk}"
                            val = ret_validation_stats[metric_name][topk]
                            run.add_scalar(name, val, epoch)
                    
                final_decision_metric = rec_decision_metric if args.decision_metric == 'bundle' else ret_decision_metric
                final_validation_stats = rec_validation_stats if args.decision_metric == 'bundle' else ret_validation_stats
                early_stopper.update(config, final_decision_metric, ddp_model.module, optimizer, best_results_path, final_validation_stats, epoch)
            
        dist.barrier() #make sure all distributed models are in sync before next epoch       
        if early_stopper.stop:  
            break 
    end_train = datetime.now()
    summary = {
        'run_info': prefix, 
        'best_epoch': early_stopper.best_epoch, 
        'best_score': early_stopper.best_score,
        'log_path': log_path, 
        'tensorboard_path':  run_path,
        'pretrained_checkpoint': args.checkpoint,
        'checkpoint_path':  checkpoint_model_path,
        'config_path': checkpoint_conf_path, 
        'best_results_path': best_results_path, 
        # 'training_time in hours': abs(early_stopper.end_train - early_stopper.start_train).total_seconds() / 3600.0   
    }    
    if utils.is_main_process(): 
        yaml.dump(summary, open(os.path.join(summary_path, "summary.yaml"), "w"))   
        w = csv.writer(open(all_experiments_path, 'a+'))
        w.writerow([prefix, early_stopper.best_epoch, early_stopper.best_score])
        print("***FINAL RUN INFO****")
        print("\n".join("{} : {}".format(k, v) for k, v in {
            'run_info': prefix,
            'best_epoch': early_stopper.best_epoch, 
            'best_score': early_stopper.best_score, 
            'tensorboard_path':  run_path}))
            # 'training_time in hours': abs(early_stopper.end_train - early_stopper.start_train).total_seconds() / 3600.0}))

    #Clean up multiprocessing threads
    dist.destroy_process_group()

if __name__ == '__main__':
    args = parse_args() 
    
    world_size = min(torch.cuda.device_count(), args.world_size)
    print(f"LAUNCHING DISTRIBUTED RUN -- {world_size} threads")
    mp.spawn(distributed_main, args=(world_size, args), nprocs=world_size, join=True)



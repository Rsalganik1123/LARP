import ruamel.yaml as yaml
import torch 
import ipdb 
import copy
import pickle 
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm  
import torch.distributed as dist 
import torchaudio 
import librosa 
import os 
import json 
from glob import glob 
from torch.utils.data import Dataset, DataLoader

from parser import parse_args
import utils
from data import Datasets 
from models.larp import LARP



def load_embeddings_from_folder(model, device, caption_file):
    # ajson.load(open('caption_example.json', 'r'))
    audio_emb, caption_emb = [], [] 
    all_files = pd.read_json(caption_file)
    with tqdm(desc=f'loading emb', unit='it', total=len(all_files), ) as pbar:
        for i, (audio_file) in enumerate(all_files.audio_files.tolist()): 
            audio, sr = torchaudio.load(audio_file)
            audio = audio.reshape(1, -1).to(device)
            caption = all_files.caption.tolist()[i]
            a, c = model.return_features(audio, caption)
            audio_emb.append(a)
            caption_emb.append(c)
            pbar.update()
    return audio_emb, caption_emb


def load_embedding_from_file(model, device, audio_file, caption): 
    audio, sr = torchaudio.load(audio_file)
    audio = audio.reshape(1, -1).to(device)
    a, c = model.return_features(audio, caption)
    return a, c

if __name__ == '__main__':
    args = parse_args() 
    #SETUP PARAMS 
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    config = yaml.safe_load(open(args.config_path, 'r'))
    config['cmdline'] = args.__dict__
    base_config = config['MODELS']['BASE_MODELS']['LARP']
    base_config['ablation_loss'] = config['cmdline']['ablation_loss']
    base_config['embed_dim'] =  config['cmdline']['embed_dim']
    base_config['fusion_method'] =  config['cmdline']['fusion_method']
    base_config['device'] = device

    audio_config = config['MODELS']['AUDIO_MODELS']['HTSAT']
    audio_config['name'] = args.audio_model
    text_config = config['MODELS']['LANGUAGE_MODELS']['BERT']
    
    base_config["num_tracks"] = 87558  #Change if using LFM checkpoint 
    base_config["num_playlists"] = 1 #Placeholder, value doesn't matter
    model = LARP(config=base_config, audio_cfg=audio_config, text_cfg=text_config, embed_dim = base_config['embed_dim']).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model']    
    model.load_state_dict(state_dict, strict=False)
    
    audio_path = '/KDD2024-LARP/audio_example.mp3'
    caption = 'Hello, thanks for using LARP!'

    audio_emb, caption_emb = load_embedding_from_file(model, device, audio_path, caption)
    
    audio_folder = '/KDD2024-LARP/'
    caption_file = '/KDD2024-LARP/caption_example.json'

    audio_emb, caption_emb = load_embeddings_from_folder(model, device, audio_path, caption)
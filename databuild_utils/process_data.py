
import pandas as pd 
import numpy as np 
from glob import glob 
import ipdb 
import json 
from random import shuffle
import openai 
import shutil
import os 
import torch 
import argparse 
from tqdm import tqdm 
import torchaudio 
import threading
import pickle 
from numpy.random import permutation
import librosa
import argparse  
import math

def upsample(audio_in_path, audio_out_path, sample_rate=48000, no_log=True, segment_start:float=0, segment_end:float=None):
    log_cmd = ' -v quiet' if no_log else ''
    segment_cmd = f'-ss {segment_start} -to {segment_end}' if segment_end else ''
    os.system(
        f'ffmpeg -y -i "{audio_in_path}" -vn {log_cmd} -flags +bitexact '
        f'-ar {sample_rate} -ac 1 {segment_cmd} "{audio_out_path}"')

class AudioTruncator(): 
    def __init__(self, audio_folder, output_folder, meta_folder, uri_selection=None, num_samples=1, verbose=False): 
        self.output_folder = output_folder
        self.meta_folder = meta_folder
        if not os.path.exists(output_folder): 
            print(f'creating output folder :{output_folder} ')
            os.makedirs(output_folder, exist_ok=True)
        if not os.path.exists(meta_folder): 
            print(f'creating meta folder :{meta_folder} ')
            os.makedirs(meta_folder, exist_ok=True) 
        self.audio_folder = audio_folder 
        self.uri_selection = pickle.load(open(uri_selection, 'rb')) if uri_selection != None else None 
        self.audio_files = self.__check_loaded(self.uri_selection)
        self.rng = self.__set_seed(198)
        self.num_samples = num_samples
        self.fail_list = []
        self.verbose = verbose 

    def __check_loaded(self, given_uris=None): 
        loaded = glob(f'{self.output_folder}*')
        loaded_uris = [l.split('/')[-1].split('.mp3')[0] for l in loaded] 
        all_uris = [l.split('/')[-1].split('.mp3')[0] for l in glob(f'{self.audio_folder}*')]
        uris_to_load = list(set(all_uris)- set(loaded_uris))
        if given_uris != None: 
            uris_to_load = list(np.intersect1d(all_uris, given_uris))
        files = [f'{self.audio_folder}{l}.mp3' for l in uris_to_load]
        print(f"total number of uris: {len(all_uris)}, previously loaded uris: {len(loaded_uris)}, uris left to load: {len(files)}")
        return files 

    def __set_seed(self, seed): 
        return np.random.default_rng(seed)    
    def __gen_random_samples(self, file_path):
        uri = file_path.split('/')[-1].split(".")[0] 
        waveform, sr = librosa.load(file_path)
        length = librosa.get_duration(y=waveform) #in seconds
        # print(length, math.floor((length-11)%10))
        audio_out_path = os.path.join(self.output_folder, f'{uri}')
        if not os.path.exists(audio_out_path): 
            os.makedirs(audio_out_path, exist_ok=True)
        if length < 10: 
            print('file too short')
            return False 
        elif length == 10: 
            upsample(file_path, audio_out_path+f'/{uri}_0.mp3', sample_rate=48000, no_log=True)
            json.dump({
                'uri': uri, 
                'full_file': file_path, 
                'segment_start': 0, 
                'segment_end': 10, 
                'sample_idx': 0
            }, open(os.path.join(self.meta_folder, f'{uri}.json'), 'w')) 
        else: 
            segment_starts = self.rng.choice(np.arange(0, length-11), math.floor((length-11)%10), replace = False)
            segment_ends = [start + 10 for start in segment_starts] 
            meta = [] 
            for i in range(len(segment_starts)): 
                segment_start, segment_end = segment_starts[i], segment_ends[i]
                upsample(file_path, audio_out_path+f'/{uri}_{i}.mp3', sample_rate=48000, no_log=True, segment_start=segment_start, segment_end=segment_end)
                meta.append({
                    'uri': uri, 
                    'full_file': file_path, 
                    'segment_start': segment_start, 
                    'segment_end': segment_end, 
                    'sample_idx': i 
                })
            json.dump(meta, open(os.path.join(self.meta_folder, f'{uri}.json'), 'w')) 
            
            
        return True
    def __check_corruption(self, file): 
        waveform, sr = torchaudio.load(file)
        if waveform.shape != torch.Size([1, 480000]): 
            print(waveform.shape, sr)
            # return False 
            raise Exception 
        return True
    def __multi_process_helper(self, data_chunk, idx, mode='truncate'): 
        batch_cnt = len(data_chunk)
        with tqdm(desc=f'running thread: {idx}', unit='it', total=batch_cnt, disable=not args.verbose) as pbar:
            for i in data_chunk:
                try:
                    if mode == 'truncate':
                        self.__gen_random_samples(i)
                    if mode == 'check': 
                        self.__check_corruption(i)
                    if self.verbose: 
                        print(f"[SUCCESS]: file: {i}")
                except Exception as e:
                    uri = i.split('/')[-2]
                    self.fail_list.append(uri)
                    print(f"[ERROR]: {e}, URI: {uri}, file: {i}")
                pbar.update() 
    def run_truncate(self, multi_threading):
        if multi_threading: 
            data_chunk = np.array_split(self.audio_files, 10)
            print(f"NUMBER OF URIS to LOAD:{len(self.audio_files)}, SPLIT INTO: {len(data_chunk)} CHUNKS --> {len(data_chunk[0])} PER CHUNK")
            print("ORGANIZING MULTITHREAD... ")
            thread_list = [] 
            for i,t in enumerate(data_chunk):
                m = threading.Thread(target=self.__multi_process_helper, args=(t,i))  
                thread_list.append(m)
            print("STARTING MULTITHREAD...")
            for m in thread_list:
                m.start()  
            for m in thread_list:
                m.join()
        else: 
            for f in tqdm(self.audio_files):
                if '.mp3' not in f: continue 
                if not self.__gen_random_samples(f): print("ERROR: ", f)
        pickle.dump(self.fail_list, open(os.path.join(self.output_folder, 'truncation_failed_uris.pkl'), 'wb') ) 
    def run_corruption_check(self, multi_threading):
        # ipdb.set_trace() 
        self.audio_files = glob(f'{self.output_folder}*/*')
        if multi_threading: 
            data_chunk = np.array_split(self.audio_files, 30)
            print(f"NUMBER OF URIS to LOAD:{len(self.audio_files)}, SPLIT INTO: {len(data_chunk)} CHUNKS --> {len(data_chunk[0])} PER CHUNK")
            print("ORGANIZING MULTITHREAD... ")
            thread_list = [] 
            for i, t in enumerate(data_chunk):
                m = threading.Thread(target=self.__multi_process_helper, args=(t,i), kwargs={'mode': 'check'})  
                thread_list.append(m)
            print("STARTING MULTITHREAD...")
            for m in thread_list:
                m.start()  
            for m in thread_list:
                m.join()
        else: 
            for f in tqdm(self.audio_files):
                try: 
                    self.__check_corruption(f)
                except: 
                    print('ERROR: ', f)
                    self.fail_list.append(f)
                
        pickle.dump(self.fail_list, open(os.path.join('truncation_failed_uris.pkl'), 'wb') ) 

def parse_truncation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio_folder", type=str, default='/mnt/ssd1/rebecca/lfm/mp3/', help="data path to full mp3 files")
    parser.add_argument("--audio_output_folder",type=str, default = '/mnt/ssd1/rebecca/lfm/trunc_mp3/', help="output path")
    parser.add_argument("--meta_output_folder",type=str, default = '/mnt/ssd1/rebecca/lfm/trunc_meta/', help="output path")
    parser.add_argument("--mt", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    return parser.parse_args()

if __name__ == '__main__': 
    args = parse_truncation_args()
    truncator = AudioTruncator(audio_folder=args.input_audio_folder, output_folder =args.audio_output_folder, meta_folder = args.meta_output_folder, num_samples=3, verbose=args.verbose)
    truncator.run_truncate(multi_threading=args.mt)
    # truncator.run_corruption_check(multi_threading=args.mt)
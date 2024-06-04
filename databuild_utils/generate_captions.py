import pandas as pd 
import numpy as np 
from glob import glob 
import ipdb 
import json 
from random import shuffle
import openai 
import os 
import pickle 
import argparse 
from tqdm import tqdm 
pd.options.mode.chained_assignment = None


#UTILS 

standard_music_keys = ['danceability', 'energy', 'key', 'loudness', 'mode', 
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo', 'time_signature']
notes = ['Ambigious', 'C', 'C sharp', 'D', 'D sharp', 'E', 'F', 'F sharp', 'G', 'A flat', 'A', 'B flat', 'B',]
modes = dict(zip(list(range(-1, 12)), notes))

def metadata_caption(track, artist, album ): 
    return f"{track} is a song by the artist {artist} on the album {album}"

def genre_caption(genre_list): 
    return f"the genres in this song are {' and '.join(genre_list)}"

def mus_feat_caption(mus_feat): 
    mus_str = {" and ".join("{}: {}".format(k, v) for k, v in mus_feat.items())}
    return f"the musical attributes in this song are {mus_str}"

def min_max_scaling(data): 
    return (data-data.min())/(data.max()-data.min())

def convert_key(mode, key): 
    
        description = modes[key]
        if description == 'Ambigious': 
            return description 
        if mode == 0: 
            description += ' minor'
        else: description += " major"
        return description 




class CaptionBuilder(): 
    def __init__(self, data_modes_dict, output_path, normalization, verbose, split_path): 
        #PARAMS 
        self.verbose = verbose
        self.normalization = normalization
        self.data_modes = data_modes_dict
        self.modalities = data_modes_dict.keys() 
        self.output_path = output_path
        self.split_uri = pickle.load(open(split_path, 'rb')) if split_path != None else []
        if split_path != None: 
            self.split_name = split_path.split('/')[-1].split('_')[0]
            assert self.split_name in ['train', 'val', 'valid', 'test']
        #LOAD DATA 
        self.data_collections = self.__load_data()
        print({f'{k} : {len(v)}'for k, v in self.data_collections.items()})
        self.full_data = self.__build_dataframe()
        print(f"full data loaded: {len(self.full_data)}")
    
    def __collect_files(self, mode, folder): 
        # ipdb.set_trace() 
        if self.verbose: print(f"collecting files for {mode}")
        files = glob(folder+'/*')
        if len(files) ==0: print(f'no files found for mode : {mode}, check paths in data_path_dict')
        data = [] 
        with tqdm(desc=f'loading {mode} files', unit='it', total=len(files)) as pbar:
            for f in files: 
                if mode == 'mp3': 
                    if os.path.isdir(f):
                        uri = f.split('/')[-1] 
                        if len(self.split_uri) > 0:  
                            if uri not in self.split_uri: continue 
                        slices = glob(f+"/*")
                        if len(slices) == 0: 
                            continue 
                        short_path = "/".join(np.random.choice(slices).split('/')[-2:])
                        temp = {'file_path': f, 'track_uri': uri, 'slices': slices, 'short_path': short_path}
                else: 
                    temp = json.load(open(f, 'r'))
                    uri = f.split('/')[-1].split('.')[0]
                    if 'genre' in mode: 
                        temp['artist_uri'] = uri
                    else: 
                        temp['track_uri'] = uri
                data.append(temp) 
                pbar.update() 
        return pd.DataFrame(data)

    def __load_data(self): 
        # ipdb.set_trace() 
        if self.verbose: print("loading datasets")
        dfs = {}
        for info_type, data in self.data_modes.items():
            if self.verbose: print(f"adding {info_type}")
            path, keys = data 
            dfs[info_type] = self.__collect_files(info_type, path)
            if len(keys) > 0: 
                dfs[info_type] = dfs[info_type][keys]
            if info_type == 'music': 
                self.mus_keys = keys
                self.mus_keys.remove('track_uri')
        return dfs

    def __build_dataframe(self):
        if self.verbose: print("building dataframe")
        merged = self.data_collections['meta'].rename(columns={'primary_artist_uri': 'artist_uri', 'primary_artist_name': 'artist_name'})
        merged['artist_uri'] = merged['artist_uri'].apply(lambda x: x.split(':')[-1])
        if 'genre' in self.modalities: 
            artists = self.data_collections['genre']
            merged = pd.merge(merged, artists, on='artist_uri')
        if 'music' in self.modalities: 
            audio_feat = self.data_collections['music']
            merged = pd.merge(merged, audio_feat, on='track_uri')
        mp3 = self.data_collections['mp3']
        merged = pd.merge(merged, mp3, on='track_uri')
        merged = merged.reset_index(drop=True)
        return merged 

    def __categorize_mus(self, data):
        for k in data.columns: 
            if k in ['danceability', 'loudness', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'valence', 'duration_ms']: 
                labels = [f'low {k}', f'average {k}', f'high {k}']
                data[k] = pd.cut(data[k], 3, right=True, labels=labels, retbins=False, precision=3, 
                        include_lowest=False, duplicates='raise', ordered=True)
            if k in ['tempo']: 
                labels = [f'slow {k}', f'average {k}', f'fast {k}']
                data[k] = pd.cut(data[k], 3, right=True, labels=labels, retbins=False, precision=3, 
                        include_lowest=False, duplicates='raise', ordered=True)
            
            
        data['key+mode'] = data[['mode', 'key']].apply(lambda x: convert_key(x.mode, x.key), axis=1)
        self.mus_keys = ['danceability', 'loudness', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'tempo', 'speechiness', 'valence'] + ['key+mode']
        self.full_data[f'{k}'] = data[k]
        return data[self.mus_keys]

    def build_captions(self): 
        if self.verbose: print("building captions")
        fd = self.full_data
        caption_set = {} 
        for caption_type in self.modalities: 
            if self.verbose: print(f'loading {caption_type} captions')
            if caption_type == 'meta': 
                thruples = list(zip(fd['track_name'].tolist(), fd['artist_name'].tolist(), fd['album_name'].tolist()))
                meta_captions = [metadata_caption(track, artist, album) for (track, artist, album) in thruples]
                caption_set['meta'] = meta_captions
                fd['meta_caption'] = meta_captions
            if caption_type == 'genre': 
                genre_captions = [genre_caption(genres) for genres in fd['genres'].tolist()]
                caption_set['genre'] = genre_captions
                fd['genre_caption'] = genre_captions
            if caption_type == 'music': 
                mus_df =fd[standard_music_keys]
                for k in mus_df.columns: 
                        if k == 'key' or k == 'mode' or k == 'time_signature': continue  
                        mus_df[k] = min_max_scaling(mus_df[k])
                mus_df = self.__categorize_mus(mus_df) 
                mus_feat = mus_df.to_dict(orient='records') 
                music_captions = [mus_feat_caption(m) for m in mus_feat]
                caption_set['music'] = music_captions
                fd['music_caption'] = music_captions
        caption_set = [f'{m}_caption' for m in self.modalities if m != 'mp3'] 
        fd['combo_caption'] = fd[caption_set].apply(lambda x: ' '.join(x), axis=1) 
        self.full_data = fd 

    def save_data(self): 
        pickle.dump(self.full_data, open(f'{self.output_path}{self.split_name}_captions.pkl', 'wb'))
        if self.verbose: print(f'dumped dataset of size:{len(self.full_data)} to {self.output_path}{self.split_name}_captions.pkl')

def parse_caption_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--modes', nargs='+', choices=['meta', 'genre', 'music', 'mp3'], default=['meta', 'genre', 'music', 'mp3'])
    parser.add_argument('--norm_mode', default=None, type=str)
    parser.add_argument('--min_max', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output_path')
    parser.add_argument('--dataset_folder', type = str, default = '/mnt/ssd1/rebecca/lfm/')
    parser.add_argument('--split', type=str, default=None)

    return parser.parse_args()

def build_dict(args):
    
    data_path_dict= {
        'meta': [os.path.join(args.dataset_folder, 'temp_metadata'), ['track_uri', 'primary_artist_uri', 'track_name', 'primary_artist_name', 'album_name'] ], 
        'genre': [os.path.join(args.dataset_folder, 'temp_genre'), ['artist_uri', 'genres']],
        'music': [os.path.join(args.dataset_folder, 'temp_music'), ['track_uri'] + standard_music_keys],
        'mp3': [os.path.join(args.dataset_folder, 'trunc_mp3'), []], 
        }

    return {k: data_path_dict[k] for k in args.modes}


if __name__ == "__main__":
    args = parse_caption_args()
    data_modes_dict = build_dict(args)
    captioner = CaptionBuilder(data_modes_dict, args.output_path,  args.norm_mode, args.verbose, args.split)
    captioner.build_captions()
    captioner.save_data()
import pickle 
import json
import os
import sys
import numpy as np
from tqdm import tqdm

import scipy.sparse as sp 
import argparse 
import ipdb 
import pandas as pd 

def pad(x): 
    return [0] if len(x) < 1 else x


class InteractionBuilder(): 
    def __init__(self, project_folder, test_interactions_path=None, seed = 45, verbose=True): 
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose 

        #PATHS FOR I/O
        self.project_folder = project_folder
        self.caption_folder = f'{self.project_folder}caption_sets/'
        self.data_folder = os.path.join(self.project_folder, 'og_csv')
        # self.tracks_path = os.path.join(self.data_folder, 'tracks.csv')
        # self.interaction_path = os.path.join(self.data_folder, 'interactions.csv')
        self.output_folder = f'{self.project_folder}evaluation_sets/bundle/'
        if not os.path.exists(self.output_folder): 
            os.makedirs(self.output_folder, exist_ok=True)

        #load all data 
        # self.all_tracks = pd.read_csv(self.tracks_path, sep='\t')
        # self.all_interactions = pd.read_csv(self.interaction_path, sep='\t')
        # self.test_interactions = None 
        # if test_interactions_path != None:
        #     if args.verbose: print('loading test interactions from provided path') 
        #     self.test_interactions = pd.read_csv(test_interactions_path, sep='\t') 
        #     self.all_interactions = pd.concat([self.all_interactions, self.test_interactions])
        #     self.all_interactions['track_uri'] = self.all_interactions.track_uri.apply(lambda x: x.split(":")[-1]).reset_index(drop=True)
            

        # self.pid2pidx = {pid:idx for idx, pid in enumerate(self.all_interactions['pid'].unique())}
        # self.uri2tidx = {uri: idx for idx, uri in enumerate(self.all_interactions['track_uri'].unique())}
        

        self.train_int = pickle.load(open(f"{self.project_folder}clean_org/train_interactions.pkl", "rb")).reset_index(drop=True) #self.__check_files('train', 'int')
        self.valid_int = pickle.load(open(f"{self.project_folder}clean_org/valid_interactions.pkl", "rb")).reset_index(drop=True)#self.__check_files('valid', 'int')
        self.test_int = pickle.load(open(f"{self.project_folder}clean_org/test_interactions_trunc.pkl", "rb")).reset_index(drop=True) #self.__check_files('test', 'int')

        self.train_uri = self.__check_files('train', 'uri')
        self.valid_uri = self.__check_files('valid', 'uri') 
        self.test_uri = self.__check_files('test', 'uri') 
        if self.verbose: print(f'{len(self.train_uri)} train-uris, {len(self.valid_uri)} valid uris, {len(self.test_uri)} test uris')
    
        self.uri2id = {}

    def __check_files(self, split, mode):
        if mode == 'uri': 
            if split == 'test': 
                uri_set = pickle.load(open(f"{self.project_folder}clean_org/{split}_uri_trunc.pkl", "rb"))
                captions = pickle.load(open(os.path.join(self.caption_folder, f'{split}_captions_trunc.pkl'), 'rb'))
                caption_set = np.intersect1d(captions.track_uri.unique(), uri_set).tolist()
                int_set = pickle.load(open(f"{self.project_folder}clean_org/{split}_interactions_trunc.pkl", "rb")).track_uri.unique().tolist() 
                filtered_uris = np.intersect1d(caption_set, int_set).tolist()  
                # pickle.dump(captions[captions.track_uri.isin(filtered_uris)].reset_index(drop=True), open(f"{self.project_folder}caption_sets/{split}_captions_cleaned.pkl", 'wb'))
                # pickle.dump(filtered_uris, open(f"{self.project_folder}clean_org/{split}_uri_cleaned_trunc.pkl", "wb")) 
                # print(len(filtered_uris))
                return filtered_uris
            
            uri_set = pickle.load(open(f"{self.project_folder}clean_org/{split}_uri.pkl", "rb"))
            captions = pickle.load(open(os.path.join(self.caption_folder, f'{split}_captions.pkl'), 'rb'))
            caption_set = np.intersect1d(captions.track_uri.unique(), uri_set).tolist()
            int_set = pickle.load(open(f"{self.project_folder}clean_org/{split}_interactions.pkl", "rb")).track_uri.unique().tolist() 
            filtered_uris = np.intersect1d(caption_set, int_set).tolist()  
            pickle.dump(captions[captions.track_uri.isin(filtered_uris)].reset_index(drop=True), open(f"{self.project_folder}caption_sets/{split}_captions_cleaned.pkl", 'wb'))
            pickle.dump(filtered_uris, open(f"{self.project_folder}clean_org/{split}_uri_cleaned.pkl", "wb")) 
            return filtered_uris

    def __convert_turi2aid(self, x):
        uri = self.tid2turi[x]
        return self.auri2aid[self.turi2auri[uri]]

    def __array_map(self, x):
        x[:, 1] = np.array(list(map(self.__convert_turi2aid, x[:, 1])))
        return x

    def __make_playlist(self, dataset, mode, uris):
        playlists = {}
        with tqdm(desc=f'finding {mode} playlists', unit='it', total=len(dataset)) as pbar:
            for i in dataset.iterrows():
                uri = i[1]['track_uri'].split("track:")[-1]
                pid = i[1]['pid']
                if uri not in uris:
                    continue
                if pid in playlists:
                    playlists[pid].append(uri)
                else:
                    playlists[pid] = [uri]
                pbar.update()

        playlists2 = []
        for i in playlists:
            playlists2.append(playlists[i])
        return playlists2

    def __make_bi_pairs(self, mode, bid, playlist): 
        bi = []
        split_uri2id = {}
        with tqdm(desc=f'creating {mode} bundles', unit='it', total=len(playlist)) as pbar:
            for uris in playlist:
                for uri in uris:
                    if uri not in self.uri2id:
                        self.uri2id[uri] = len(self.uri2id)
                    if uri not in split_uri2id: 
                        split_uri2id[uri] = len(split_uri2id)
                    tid = split_uri2id[uri]
                    bi.append([bid, tid])
                bid+=1
                pbar.update() 
        return bid, bi, split_uri2id

    def __run_conversion(self, uri_list, idx_dict): 
        converted_list = [] 
        with tqdm(desc=f'converting uri to idx', unit='it', total=len(uri_list)) as pbar:
            for u in uri_list: 
                converted_list.append(idx_dict[u])
                pbar.update() 
        return np.array(converted_list) 

    def __get_bundle_items(self, task): 
        
        data = np.load(f"{self.output_folder}bi_{task}_trunc.npy")
        bundle_items, bundle_sum = {}, {}
        for b,i in data:
            if b in bundle_items:
                bundle_items[b].append(i)
                bundle_sum[b] += 1
            else:
                bundle_items[b] = [i]
                bundle_sum[b] = 1   
        return bundle_items, bundle_sum
    
    def __dict_to_np(self, d):  
        o = []
        for k in d:
            for v in d[k]:
                o.append([k,v])
        return np.array(o, dtype=np.int32)
        
    def __save_bundle_split(self, task, bundle_items, bundle_sum): 
        bundle_input, bundle_masked = {}, {}
        for b in bundle_items:
            b = int(b)
            items = np.array(bundle_items[b],dtype=int)
            # shuffle
            self.rng.shuffle(items)
            line = round(bundle_sum[b]/2)
            bundle_input[b] = items[:line]
            bundle_masked[b] = items[line:]
       
        np.save(os.path.join(self.output_folder, f"bi_{task}_input_trunc.npy"), self.__dict_to_np(bundle_input))
        np.save(os.path.join(self.output_folder, f"bi_{task}_gt_trunc.npy"), self.__dict_to_np(bundle_masked))

    def generate_full_matrix(self): 
        if self.verbose: print('generating full matrix')
        columns = self.__run_conversion(self.all_interactions['pid'].tolist(), self.pid2pidx)
        rows = self.__run_conversion(self.all_interactions['track_uri'].tolist(), self.uri2tidx)
        mat = np.vstack([columns, rows])
        with open(f'{self.output_folder}bi_full.npy', 'wb') as fp: 
            np.save(fp, mat)
            if self.verbose: print(f'saved mat of size {mat.shape} to {self.output_folder}bi_full.npy')
        return 0 
    
    def create_bundle_sets(self): 
        
        train = self.__make_playlist(self.train_int, 'train', self.train_uri)
        b_id, bi_train, train_uri2id = self.__make_bi_pairs('train', 0, train)
        np.save(f"{self.output_folder}bi_train.npy", np.array(bi_train))
        
        valid = self.__make_playlist(self.valid_int, 'valid', self.valid_uri)
        b_id, bi_valid, valid_uri2id = self.__make_bi_pairs('valid', 0, valid)
        np.save(f"{self.output_folder}bi_valid.npy", np.array(bi_valid))

        test = self.__make_playlist(self.test_int, 'test', self.test_uri) 
        b_id, bi_test, test_uri2id = self.__make_bi_pairs('test', 0, test)
        np.save(f"{self.output_folder}/bi_test_trunc.npy", np.array(bi_test))
        
        with open(f"{self.output_folder}train_datasize", "w") as f:
            f.write(", ".join([str(len(train)), str(len(train_uri2id))]))

        with open(f"{self.output_folder}valid_datasize", "w") as f:
            f.write(", ".join([str(len(valid)), str(len(valid_uri2id))]))

        with open(f"{self.output_folder}test_datasize_trunc", "w") as f:
            f.write(", ".join([str(len(test)), str(len(test_uri2id))]))
       
        with open(f"{self.output_folder}full_datasize", "w") as f:
            f.write(", ".join([str(len(train) + len(valid) + len(test)), str(len(self.uri2id))]))
         
        with open(f"{self.output_folder}train_uri2id", "w") as f:
            f.write(json.dumps(train_uri2id))
        
        with open(f"{self.output_folder}valid_uri2id", "w") as f:
            f.write(json.dumps(valid_uri2id))

        with open(f"{self.output_folder}test_uri2id_trunc", "w") as f:
            f.write(json.dumps(test_uri2id))

        with open(f"{self.output_folder}all_uri2id", "w") as f:
            f.write(json.dumps(self.uri2id))
        return 0 

    def split_gt(self):
        valid_bundle_items, valid_bundle_sum = self.__get_bundle_items('valid')
        test_bundle_items, test_bundle_sum = self.__get_bundle_items('test')

        self.__save_bundle_split('valid', valid_bundle_items, valid_bundle_sum)
        self.__save_bundle_split('test', test_bundle_items, test_bundle_sum)
        return 0 

    def mine_pairs_and_parents(self, pair_samples): 
        keys = ['track_uri', 'meta_caption', 'short_path']

        train_interactions = self.all_interactions[self.all_interactions.track_uri.isin(self.train_uri)].reset_index(drop=True)
        grouped_pids_by_track = train_interactions.groupby('track_uri')['pid'].apply(list).reset_index(name='pids')
        grouped_tracks_by_pid = train_interactions.groupby('pid')['track_uri'].apply(list).reset_index(name='tids')
        

        pid2pidx = {pid:idx for idx, pid in enumerate(grouped_tracks_by_pid['pid'].unique())}
        uri2tidx = json.load(open(self.output_folder+'train_uri2id', 'r')) #{uri: idx for idx, uri in enumerate(self.train_uri)}
        
        with open(f'{self.project_folder}pair_sets/pt.size', "w") as f:
            f.write(", ".join([str(len(pid2pidx)), str(len(uri2tidx))]))

        train_interactions['pidx'] = train_interactions['pid'].apply(lambda x: pid2pidx[x])
        train_interactions['tidx'] = train_interactions['track_uri'].apply(lambda x: uri2tidx[x])
        pt_pairs = train_interactions[['pidx', 'tidx']].to_numpy()
        pickle.dump(pt_pairs, open(f'{self.project_folder}pair_sets/pt_pairs.pkl', 'wb'))

            
        uris, final_neighbours = [], []
        with tqdm(desc=f'sampling for neighbours', unit='it', total=len(grouped_pids_by_track.track_uri.unique())) as pbar:
            for track_uri in grouped_pids_by_track.track_uri.unique(): 
                pids = grouped_pids_by_track[grouped_pids_by_track.track_uri == track_uri].pids.tolist()[0]
                neighbour_groups = grouped_tracks_by_pid[grouped_tracks_by_pid.pid.isin(pids)].tids.tolist() 
                neighbour_list = list(set([t for l in neighbour_groups for t in l])) 
                if pair_samples > 0: 
                    neighbour_list = self.rng.choice(neighbour_list, pair_samples).tolist() 
                final_neighbours.append(neighbour_list)
                uris.append(track_uri)
                pbar.update()

        df = pd.DataFrame({'track_uri':uris, 'pair': final_neighbours})
        df['num_pair'] = df.pair.apply(len)
        if self.verbose: 
            print("AVERAGE NUMBER OF SAMPLES PER TRACK:", df.num_pair.mean())
        df = df.drop(columns='num_pair')
        pair_df = df.explode(['pair']).reset_index(drop=True)
        pair_df = pair_df.rename(columns={'track_uri': 'track_uri_A', 'pair': 'track_uri_B'})
        
        #CIC PAIRS 
        caption_org = pickle.load(open(self.caption_folder+'train_captions.pkl', 'rb'))
        caption_uri2idx = dict(zip(caption_org.track_uri.unique(), range(len(caption_org.track_uri.unique())))) 
        pair_df['idx_A'] = pair_df['track_uri_A'].apply(lambda x: caption_uri2idx[x])
        pair_df['idx_B'] = pair_df['track_uri_B'].apply(lambda x: caption_uri2idx[x])
        
       
        set_A = caption_org.loc[pair_df.idx_A.tolist()][keys].add_suffix('_A').reset_index(drop=True)
        set_B = caption_org.loc[pair_df.idx_B.tolist()][keys].add_suffix('_B').reset_index(drop=True)
        sets_df = pd.concat([set_A, set_B], axis=1) 
        # pair_df_merged = pd.merge(pair_df, sets_df, on='track_uri_A')        

        sets_df['idx'] = sets_df['track_uri_A'].apply(lambda x: uri2tidx[x])
        uir2pids = {i[1].track_uri: i[1].pids for i in grouped_pids_by_track.iterrows()}
        sets_df['pidx'] = sets_df['track_uri_A'].apply(lambda x: pid2pidx[self.rng.choice(uir2pids[x])])

        pickle.dump(sets_df, open(f'{self.project_folder}pair_sets/{pair_samples}_pairs.pkl', 'wb'))

    def build_masks(self): 
        self.turi2tid = json.load(open(f'{self.output_folder}test_uri2id', 'r')) 
        test_captions = pickle.load(open(f'{self.caption_folder}test_captions.pkl', 'rb'))
        self.tid2turi = {v: k for k, v in self.turi2tid.items()}    
        self.auri2aid = dict(zip(test_captions.artist_uri.unique(), range(len(test_captions.artist_uri.unique()))))
        self.turi2auri = dict(zip(test_captions.track_uri.tolist(), test_captions.artist_uri.tolist()))
        bi_test = np.load(f'{self.output_folder}bi_test.npy')
        bi_test_gt = np.load(f'{self.output_folder}bi_test_gt.npy')

        #BUILD ARTIST GT 
        ba_test = self.__array_map(bi_test)
        ba_test_gt = self.__array_map(bi_test_gt)
        num_artists = np.max(ba_test)

        np.save(f'{self.output_folder}ba_test.npy', ba_test) 
        np.save(f'{self.output_folder}ba_test_gt.npy', ba_test_gt)

        #BUILD ARTIST MASK 
        with open(f'{self.output_folder}test_datasize') as f:
            num_bundles, num_items = [eval(i.strip()) for i in f.read().split(",")]

        aids = [self.auri2aid[self.turi2auri[self.tid2turi[idx]]] for idx in range(num_items)]
        tids = list(range(num_items)) 
        row_idx = aids #--> #artists 
        col_idx = tids #--> #tracks

        artist_mask = np.zeros((len(set(row_idx)), len(col_idx)), dtype=int)
        artist_mask[row_idx, col_idx] += 1
        np.save(f'{self.output_folder}artist_mask_test.npy', artist_mask)    

        #BUILD GENRE GT 
        bi_test = np.load(f'{self.output_folder}bi_test.npy')
        bi_test_gt = np.load(f'{self.output_folder}bi_test_gt.npy')

        all_genres = list(set([g for genre_vector in test_captions.genres.tolist() for g in genre_vector]))
        num_genres = len(all_genres) 
        gname2gid = dict(zip(all_genres, range(1, len(all_genres)+1)))
        test_captions['genre_idx'] = test_captions['genres'].apply(lambda x: [gname2gid[g] for g in x])
        test_captions['genre_idx'] = test_captions['genre_idx'].apply(lambda x: pad(x)) 
        test_captions['tid'] = test_captions['track_uri'].apply(lambda x: self.turi2tid[x]) 

        interaction_df = pd.DataFrame({"pid": bi_test[:, 0], 'tid': bi_test[:, 1]})

        genre_df = pd.merge(interaction_df, test_captions[['tid', 'genre_idx']]) 
        genre_df = genre_df.explode('genre_idx').astype(int)  
        genre_gt = genre_df[['pid', 'genre_idx']].to_numpy()
        np.save(f'{self.output_folder}bg_test_gt.npy', genre_gt)

        #BUILD GENRE MASK 

        all_genres = list(set([g for genre_vector in test_captions.genres.tolist() for g in genre_vector]))
        genre_df = test_captions[['track_uri','genre_idx']].explode('genre_idx').astype({'genre_idx': int}) 

        gids = genre_df.genre_idx.tolist() 
        tids = [self.turi2tid[uri] for uri in genre_df.track_uri]

        row_idx = np.array(gids) #--> #genres
        col_idx = np.array(tids) #--> #tracks

        genre_mask = np.zeros((len(set(row_idx)), len(set(col_idx))), dtype=int)
        genre_mask[row_idx, col_idx] += 1
        np.save(f'{self.output_folder}genre_mask_test.npy', genre_mask)



def parse_mat_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_folder', default='/mnt/ssd1/rebecca/lfm/', help='folder where original dataset files are stored')
    parser.add_argument('--test_interactions_path', default=None, help='file where test_interactions detailed (only necessary if separate from train_interactions)')
    parser.add_argument("--verbose", action='store_true', help='print statements to update progress')
    return parser.parse_args()

if __name__ == '__main__': 
     
    args = parse_mat_args() 
    b = InteractionBuilder(args.project_folder, args.test_interactions_path)
    b.generate_full_matrix()
    b.create_bundle_sets()
    b.split_gt()
    b.mine_pairs_and_parents(pair_samples=0)  
    b.build_masks() 

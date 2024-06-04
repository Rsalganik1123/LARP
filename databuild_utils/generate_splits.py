import random
import pandas as pd 
pd.options.mode.chained_assignment = None
import argparse 
import os 
import ipdb 
import pickle
import numpy as np 
 

class PlaylistSplitter(): 
    def __init__(self, train_size, valid_size, test_size, min_playlist_length, max_playlist_length, project_folder, pair_samples, test_interactions_path=None, seed = 45, verbose=True): 
        ##SETTINGS
        self.train_size = train_size
        self.valid_size = valid_size 
        self.test_size = test_size 
        self.min_playlist_length = min_playlist_length
        self.max_playlist_length = max_playlist_length
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.verbose = True
        self.pair_samples = pair_samples
        #PATHS FOR I/O
        self.project_folder = project_folder
        self.data_folder = os.path.join(self.project_folder, 'og_csv')
        self.tracks_path = os.path.join(self.data_folder, 'tracks.csv')
        self.interaction_path = os.path.join(self.data_folder, 'interactions.csv')
        self.pair_folder = None 
        if self.pair_samples > -1: 
            self.pair_folder = os.path.join(self.project_folder, 'pair_sets')
            if not os.path.exists(self.pair_folder): 
                if args.verbose: print(f'creating folder {self.pair_folder} for storing pairs')
                os.makedirs(self.pair_folder, exist_ok=True)
        self.output_folder = os.path.join(self.project_folder, 'clean_org')  
        if not os.path.exists(self.output_folder): 
            if args.verbose: print(f'creating folder {self.output_folder} for storing splits')
            os.makedirs(self.output_folder, exist_ok=True)

        #load all data 
        self.all_tracks = pd.read_csv(self.tracks_path, sep='\t')
        self.all_interactions = pd.read_csv(self.interaction_path, sep='\t')
        self.test_interactions = None 
        if test_interactions_path != None:
            if args.verbose: print('loading test interactions from provided path') 
            self.test_interactions = pd.read_csv(test_interactions_path, sep='\t') 

    def __check_for_length(self, interactions):
        if self.verbose: print(f'filtering for playlist length: {self.min_playlist_length}')
        if self.min_playlist_length == 0: 
            return  interactions.pid.unique().tolist() 
        grouped_tracks_by_pid = interactions.groupby('pid')['track_uri'].apply(list).reset_index(name='tids')
        grouped_tracks_by_pid['num_tids'] = grouped_tracks_by_pid['tids'].apply(len)
        filtered =  grouped_tracks_by_pid[(grouped_tracks_by_pid.num_tids > self.min_playlist_length)]
        if self.verbose: print(f'-> previously {len(grouped_tracks_by_pid)} pids, now {len(filtered)} pids')
        filtered['trunc_tids'] = filtered['tids'].apply(lambda x: x[:101])
        truncated = filtered.rename(columns={'trunc_tids':'track_uri'})
        return truncated.explode('track_uri')[['pid', 'track_uri']].reset_index(drop=True) 
    
    def __make_valid_test(self, exclude):
        if self.verbose: print('making valid/test set')
        interactions = self.all_interactions
        if self.test_interactions != None: 
            interactions = self.test_interactions 
        filtered_interactions = interactions[~interactions.track_uri.isin(exclude)]
        subset_pids = self.rng.choice(filtered_interactions.pid.unique(), size=self.valid_size, replace=False).tolist() 
        subset_interactions = filtered_interactions[filtered_interactions.pid.isin(subset_pids)]
        final_interactions = self.__check_for_length(subset_interactions)
        if self.verbose: print(f'--> created valid/test dataset with {len(final_interactions.pid.unique())} playlists and {len(final_interactions.track_uri.unique())} tracks')
        return final_interactions 

    def __make_train(self):
        if self.verbose: print('making train set')
        train_uris = self.rng.choice(self.all_tracks.track_uri.unique(), size=self.train_size, replace=False).tolist()   
        train_tracks = self.all_tracks[self.all_tracks.track_uri.isin(train_uris)].reset_index(drop=True) 
        train_interactions = self.all_interactions[self.all_interactions.track_uri.isin(train_tracks.track_uri.tolist())]
        if self.verbose: 
            print(f"--> created train dataset with {len(train_tracks.track_uri.unique())} tracks")
        return train_tracks , train_interactions
    
    def run_split(self): 
        train_tracks, train_interactions = self.__make_train()
        train_uris = train_tracks.track_uri.unique().tolist()
        valid_interactions = self.__make_valid_test(exclude = train_uris)
        valid_uris = valid_interactions.track_uri.unique().tolist()
        test_interactions = self.__make_valid_test(exclude = list(train_uris + valid_uris))
        test_uris = test_interactions.track_uri.unique().tolist()

        pickle.dump(train_uris, open(f'{self.output_folder}/train_uri.pkl', 'wb'))
        pickle.dump(valid_uris, open(f'{self.output_folder}/valid_uri.pkl', 'wb'))
        pickle.dump(test_uris, open(f'{self.output_folder}/test_uri.pkl', 'wb'))

        pickle.dump(train_interactions, open(f'{self.output_folder}/train_interactions.pkl', 'wb'))
        pickle.dump(valid_interactions, open(f'{self.output_folder}/valid_interactions.pkl', 'wb'))
        pickle.dump(test_interactions, open(f'{self.output_folder}/test_interactions.pkl', 'wb'))
        
        if args.verbose: 
            print(f"wrote {len(train_uris)} uris to {self.output_folder}/train_uri.pkl")
            print(f"wrote {len(valid_uris)} uris to {self.output_folder}/valid_uri.pkl")
            print(f"wrote {len(test_uris)} uris to {self.output_folder}/test_uri.pkl")

        if args.pair_samples > -1:
            train_pairs = self.__mine_pairs_and_playlists(train_uris) 
            pickle.dump(train_pairs, open(f'{self.pair_folder}/{self.num_samples}_pairs.pkl', 'wb'))
            if args.verbose: print(f"wrote {pairs} uris to {self.pair_folder}/{self.num_samples}_pairs.pkl")

class SessionSplitter():
    def __init__(self, train_size, test_size, project_folder, seed=72): 
        self.train_size = train_size
        self.test_size = test_size 
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.verbose = True
        
        #PATHS FOR I/O
        self.project_folder = project_folder
        self.data_folder = os.path.join(self.project_folder, 'og_csv')
        self.tracks_path = os.path.join(self.data_folder, 'tracks.csv')
        self.interaction_path = os.path.join(self.data_folder, 'listening_events.tsv')
        self.user_path = os.path.join(self.data_folder, 'users.tsv')
        self.pair_folder = None 
        if self.pair_samples > -1: 
            self.pair_folder = os.path.join(self.project_folder, 'pair_sets')
            if not os.path.exists(self.pair_folder): 
                if args.verbose: print(f'creating folder {self.pair_folder} for storing pairs')
                os.makedirs(self.pair_folder, exist_ok=True)
        self.output_folder = os.path.join(self.project_folder, 'clean_org')  
        if not os.path.exists(self.output_folder): 
            if args.verbose: print(f'creating folder {self.output_folder} for storing splits')
            os.makedirs(self.output_folder, exist_ok=True)

        #load all data 
        self.all_tracks = pd.read_csv(self.tracks_path, sep='\t')
        self.all_users = pd.read_csv(self.user_path, sep='\t')
    def __create_sessions(self, subset, split_name, exclude=[]): 
        #FILTER BY YEAR
        d = datetime.datetime(2020, 1, 1, 0, 0)
        subset['timestamp'] = pd.to_datetime(subset['timestamp'])
        subset_2020 = subset[subset.timestamp > d]
        #CREATE DAILY BINS
        idx = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        bins = idx.tolist()
        subset_2020['session_id'] = pd.cut(
            subset_2020['timestamp'], 
            bins,
            labels=list(range(1, len(bins))),
            right=False
        )
        #SELECT MOST POPULOUS SESSION PER USER 
        subset_2020_by_user = subset_2020.groupby('user_id').agg({'track_id':list, 'session_id':list}).reset_index()
        uids = [] 
        tids = [] 
        subset_2020 = subset_2020.astype({'session_id':int})
        for user_id in tqdm(subset_2020_by_user.user_id.unique()): 
            u_subset = subset_2020[subset_2020.user_id == user_id]
            top_session_id, num_tracks = C(u_subset.session_id.tolist()).most_common(1)[0]
            session_tracks = u_subset[u_subset.session_id == top_session_id].track_id.tolist() 
            uids.append(user_id)
            tids.append(session_tracks) 
        session_set = pd.DataFrame({'pid':uids, 'track_id':tids}) 
        session_set['num_tids'] = session_set.track_id.apply(len)
        session_set_30 = session_set[session_set.num_tids > 30] 
        session_final_set = session_set_30.explode('track_id')
        session_final_set_track_uris = pd.merge(session_final_set, track_set)[['pid', 'track_uri']]
        if len(exclude) > 0: 
            session_final_set_track_uris = session_final_set_track_uris[~session_final_set_track_uris.track_uri.isin(exclude)].reset_index(drop=True)
        pickle.dump(session_final_set_track_uris.track_uris.unique(), open(f'{self.output_folder}{split_name}_uri.pkl', 'wb'))
        pickle.dump(session_final_set_track_uris, open(f'{self.output_folder}{split_name}_interaction.pkl', 'wb'))
        return session_final_set_track_uris.track_uris.unique(), session_final_set_track_uris.pid.unique()

        
    def __load_int(self, train_uids, test_uids): 
        train_dfs, test_dfs = [], []
        with pd.read_csv(self.interaction_path, sep='\t', iterator=True, chunksize=10000) as reader:
            for chunk in tqdm(reader):
                inter = chunk
                train_subset = inter[inter.user_id.isin(train_uids)]
                train_dfs.append(train_subset)
                test_subset = inter[inter.user_id.isin(test_uids)]
                test_dfs.append(test_subset)
        return train_dfs, test_dfs 
            
    def __make_pids(self):
        if self.verbose: print('making train set')
        train_pids = self.rng.choice(self.all_users.uid.unique(), size=self.train_size, replace=False).tolist()   
        option_pool = list(set(self.all_users.uid.unique()) - set(train_pids))
        test_pids = self.rng.choice(option_pool, size=self.test_size, replace=False).tolist()   
        return train_pids, test_pids 
    
    def run_split(self): 
        train_pids, test_pids = self.__make_pids()
        train_dfs, test_dfs  = self.__load_int()
        train_subset = pd.concat(train_dfs, ignore_index=True) 
        train_uri, train_pid = self.__create_sessions(train_subset, 'train')
        test_subset = pd.concat(test_dfs, ignore_index=True) 
        test_uri, test_pid = self.__create_sessions(test_subset, 'test', exclude = train_uri)


def parse_split_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['LFM', 'MPD'])
    parser.add_argument('--train_size', type=int, default=100_000, help='number of tracks to be included in the final dataset')
    parser.add_argument('--valid_size', type=int, default=1_000, help='number of playlists to be included in the final dataset. Note: all associated tracks will be selected')
    parser.add_argument('--test_size', type=int, default=1_000, help='number of playlists to be included in the final dataset. Note: all associated tracks will be selected')
    parser.add_argument('--pair_samples', type=int, default=-1, help='whether to store neighbor samples for CIC/CIP loss')
    parser.add_argument('--min_playlist_length', type=int, default=40, help='minimum number of tracks a playlist must have in order to be included in dataset, anything below will be filtered out')
    parser.add_argument('--project_folder', default='/ssd3/rebecca/lfm/', help='folder where original dataset files are stored')
    parser.add_argument('--test_interactions_path', default=None, help='file where test_interactions detailed (only necessary if separate from train_interactions)')
    parser.add_argument("--verbose", action='store_true', help='print statements to update progress')
    return parser.parse_args()

if __name__ == '__main__':  
    args = parse_split_args()
    if args.dataset == 'MPD': 
        splitter = PlaylistSplitter(args.train_size, args.valid_size, args.test_size, args.min_playlist_length,args.max_playlist_length, args.project_folder, args.pair_samples,verbose=args.verbose)
        splitter.run_split()
    if args.dataset == 'LFM': 
        splitter = SessionSplitter(args.train_size, args.valid_size, args.test_size, args.project_folder, verbose=args.verbose)
        splitter.run_split()




import pandas as pd 
import pickle 
import argparse 
import ipdb 
import os 

#UTILS FOR LFM 

def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--track_path', '-t', default='/ssd3/rebecca/lfm/spotify-uris.tsv', help='folder where the spotify_uris.tsv is stored')
    parser.add_argument('--interaction_path', '-i', default='/ssd3/rebecca/lfm/listening-counts.tsv', help='folder where the spotify_uris.tsv is stored')
    parser.add_argument('--output_folder', '-o', default='/ssd3/rebecca/lfm/og_csv/', help='folder where dataset files will be stored')
    return parser.parse_args()

def load_data(): 
    ipdb.set_trace()
    args = parse_args()
    if not os.path.exists(args.output_folder): 
        print(f'making folder {args.output_folder}')
        os.makedirs(args.output_folder, exist_ok=True)
    track_data = pd.read_csv(args.track_path, sep='\t')
    track_data = track_data.rename(columns={'uri': 'track_uri'})
    track_data.to_csv(f'{args.output_folder}tracks.csv', sep='\t')
    int_data = pd.read_csv(args.interaction_path, sep='\t')
    int_data_with_uri = pd.merge(track_data, int_data, on='track_id')
    int_data_with_uri = int_data_with_uri.rename(columns={'user_id': 'pid'})
    int_data_with_uri = int_data_with_uri[['pid', 'track_uri']].reset_index(drop=True)
    int_data_with_uri.to_csv('/ssd3/rebecca/lfm/og_csv/interactions.csv', sep='\t')

load_data() 
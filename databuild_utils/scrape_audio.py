import pandas as pd 
import time 
import ipdb 
import argparse
import requests
import urllib3
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException
import os
import pickle
import requests
from tqdm import tqdm
import numpy as np 
import threading
import shutil
import re
import sys
import json 
import shutil
from glob import glob 


from secret import * 


os.environ['SPOTIPY_CLIENT_ID'] = spotify_client_id
os.environ['SPOTIPY_CLIENT_SECRET'] = spotify_client_secret

#FUNCTIONS FOR TESTING CONNECTION 
def test_spotify(): 
    session = requests.Session()
    retry = urllib3.Retry(
        respect_retry_after_header=False
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(),
                        requests_session=session)
    artist_uri = ['spotify:artist:2wIVse2owClT7go1WT98tk']
    album_uri = ['spotify:album:7AUTLy9C6cIsijk9GzOW8T']
    track_uri = ['spotify:track:0UaMYEvWZi0ZqiDOoHU3YI']
    try: 
        album_results = sp.albums(album_uri)
        track_results = sp.tracks(track_uri)
        audio_results = sp.audio_features(track_uri)
        artist_results = sp.artists(artist_uri)
        ipdb.set_trace()
        print("SUCCESSFULLY SCRAPED!")
    except SpotifyException as e: 
            if e.http_status == 429:
                print("'retry-after' value:", e.headers['retry-after'])
                retry_value = e.headers['retry-after']
                if int(e.headers['retry-after']) > 60: 
                    print("STOP FOR TODAY, retry value too high {}".format(retry_value))

def test_spotdl(): 
    uri = '0lEtFXBncZfDKm8m70zEwG'
    audio_format = 'mp3'
    cmd = f"spotdl https://open.spotify.com/track/{uri} --format {audio_format}"
    res = os.popen(cmd).read()
    print(res)

#SCRAPE UTILS
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def __find_mp3(name, audio_format):
    name = name.strip('?!')
    for i in os.listdir("./"):
        s = name.split(" ")
        if i.endswith(s[-1]+f".{audio_format}") and i.startswith(s[0]):
            return i

def __check_downloaded(output_path, mode): 
    files = glob(f'{output_path}*{mode}')
    uris = [f.split('/')[-1].split(mode)[0] for f in files]
    return set(uris)

def __collect_artists(args):
    meta_files = tqdm(glob(f'{args.input}*.json')) 
    artist_uris = [] 
    for mfp in meta_files: 
        data = json.load(open(mfp, 'r'))
        artist_uris.append(data['primary_artist_uri'])
    artist_uris =list(set(artist_uris))
    pickle.dump(artist_uris, open(f'{args.output}all_artist_uris.pkl', 'wb'))
    print(f'saved {len(artist_uris)} to {args.output}all_artist_uris.pkl')

class MultiThreadLoader(object): 
    def __init__(self, output_path, run_function, platform=''): 
        self.output_path = output_path
        self.run_function = run_function
        self.fail_list = [] 
        self.success_count = 0 
        self.platform = platform
    def process(self, x, idx, **kwargs):
        batch_cnt = len(x)
        
        d = []
        with tqdm(desc=f'running thread: {idx}', unit='it', total=batch_cnt) as pbar:
            for i in x:
                try:
                    if self.run_function(i, self.output_path, **kwargs): 
                        d.append(i[0])
                        self.success_count += 1 
                except Exception as e:
                    self.fail_list.append(i)
                    print(f"[ERROR]: {e}", i)
                pbar.update() 
        return d

#HELPER FUNCTIONS 
def download_youtube_audio(uri: str, audio_folder_path, audio_format, sample_rate=48000, verbose=False):
    os.chdir(audio_folder_path)
    if 'spotify:' in uri: 
        uri = uri.split(":")[-1]
    # ipdb.set_trace() 
    cmd = f"spotdl https://open.spotify.com/track/{uri} --format {audio_format} --ffmpeg-args '-ar {sample_rate}'" 
    res = os.popen(cmd).read()
    if "Downloaded" in res:
        # 找到音乐的名称 -- Find the name of the music
        music_name = re.findall(r'"(.*?)"', res)[0]
        file_name = __find_mp3(music_name, audio_format) 
        # os.rename(file_name, os.path.join(audio_folder_path, f"{uri}.{audio_format}")) 
        src_path, dst_path = file_name, os.path.join(audio_folder_path, f"{uri}.{audio_format}")
        dest = shutil.move(src_path, dst_path) 
        if verbose: print(f"moved from {file_name} to {dest}")
        # move this file to save dir
        if not os.path.exists(dst_path): 
            return False 
        return True
    else: 
        print(res)
    return False

def download_spotify_artist_info(batch, spotify_output_path, loader): 
    audiodata_path = spotify_output_path 
    all_audio_data = [] 
    results = loader.artists(batch)
    if results:  
        for uri, artist in zip(batch, results['artists']):
            if artist: 
                data = {
                    'followers': artist['followers']['total'],
                    'popularity': artist['popularity'],  
                    'genres': artist['genres'], 
                }
                if 'spotify:' in uri: 
                    uri = uri.split(":")[-1]
                json.dump(data, open(f'{audiodata_path}{uri}.json', 'w')) 
        return True
    return False

def download_spotify_track_info(batch, spotify_output_path,  loader):
    #load audio features
    audiodata_path = spotify_output_path 
    
    results = loader.audio_features(batch)
    if results:  
        for uri, track in zip(batch, results):
            if track:
                data = {
                    'track_uri': uri,
                    'danceability': track['danceability'], 
                    'energy': track['energy'],
                    'key': track['key'],
                    'loudness': track['loudness'],
                    'mode': track['mode'], 
                    'speechiness': track['speechiness'], 
                    'acousticness': track['acousticness'], 
                    'instrumentalness': track['instrumentalness'], 
                    'liveness': track['liveness'], 
                    'valence': track['valence'], 
                    'tempo': track['tempo'], 
                    'duration_ms': track['duration_ms'],  
                    'time_signature': track['time_signature']
                }
                # artist_uri_list.append(track['artists'][0]['uri'])
                
                if 'spotify:' in uri: 
                    uri = uri.split(":")[-1]
                json.dump(data, open(f'{audiodata_path}{uri}.json', 'w'))
        # with open(artist_list, 'a+') as f: 
        #     for a in artist_uri_list: 
        #         f.write(a+',')
        return True
    return False  

def download_spotify_metadata(batch, spotify_output_path, loader): 
    audiodata_path = spotify_output_path 
    all_audio_data = [] 
    results = loader.tracks(batch)
    if results:  
        for uri, track in zip(batch, results['tracks']):
            if track: 
                data = {
                    'track_uri': uri,
                    'track_name': track['name'], 
                    'primary_artist_name': track['artists'][0]['name'],
                    'primary_artist_uri': track['artists'][0]['uri'],
                    'album_name': track['album']['name'],
                    'album_uri': track['album']['uri'],
                    'preview_url': track['preview_url']
                }
                if len(track['album']['images']) > 0: 
                    data['album_img'] =  track['album']['images'][0]['url']
                else: 
                    data['album_img'] = None 
                if 'spotify:' in uri: 
                    uri = uri.split(":")[-1]
                json.dump(data, open(f'{audiodata_path}{uri}.json', 'w')) 
        return True
    return False
      
#LAUNCH FUNCTIONS 
def scrape_audio(args): 
    #PRELIMINARIES 
    data_path, output_path = args.input, args.output
    if not os.path.exists(output_path): 
        os.makedirs(output_path, exist_ok=True)
    audio_format = 'mp3'
    track_uris = pickle.load(open(data_path, 'rb'))
    track_uris = track_uris[:int(len(track_uris)/2)]
    downloaded = __check_downloaded(output_path=output_path, mode=audio_format)
    print(f"downloaded already: {len(downloaded)}")
    
    #SCRAPING 
    if args.mt:
        print('multi-threading selected') 
        num_thread=20
        mt = MultiThreadLoader(output_path, download_youtube_audio)
        thread_list = [] 
        ids = set([t.split(":")[-1] for t in track_uris])
        new_uris = list(ids - downloaded) 
        data_chunk = np.array_split(new_uris, num_thread-1)
        print(f"NUMBER OF URIS to LOAD:{len(new_uris)}, SPLIT INTO: {len(data_chunk)} CHUNKS --> {len(data_chunk[0])} PER CHUNK")
        print("ORGANIZING MULTITHREAD... ")
        for i,t in enumerate(data_chunk):
            m = threading.Thread(target=mt.process, args=(t,i), kwargs={'audio_format': audio_format})  
            thread_list.append(m)
        print("STARTING MULTITHREAD...")
        for m in thread_list:
            print(f"STARTING THREAD :{m}")
            m.start()  

        for m in thread_list:
            m.join()
        pickle.dump(mt.fail_list, open('failed_to_load_audio_uris.pkl', 'wb'))
        return []   
    else:
        org_list = [] 
        success_list = [] 
        for uri in tqdm(track_uris):
            if ':' in uri:
                uri = uri.split(":")[-1]
            if uri in downloaded:
                success_list.append(uri)
            elif download_youtube_audio(uri, output_path, audio_format):
                success_list.append(uri)
                org_list = {'uri': uri, 'fp': f'{output_path}/{uri}.{audio_format}'}
                downloaded.add(uri)
        return org_list

def scrape_track_metadata(args): 
    #PRELIMINARIES 
    data_path, output_path = args.input, args.output
    if not os.path.exists(output_path): 
        os.makedirs(output_path, exist_ok=True)
        print(f'created {output_path} for store metadata jsons')
    artist_uri_path = os.path.join(output_path, 'artist_uris.txt')
    #SETTING UP SPOTIFY READER CONNECTION 
    track_uris = pickle.load(open(data_path, 'rb'))

    session = requests.Session()
    retry = urllib3.Retry(
        respect_retry_after_header=False
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(),
                        requests_session=session)
    #SPLITTING DATA INTO BATCHES
    downloaded = __check_downloaded(output_path=output_path, mode='json')
    print(f"previously loaded {len(downloaded)} tracks")
    batch_size = 30 # Note; setting this too high will cause rate limit errors 

    ids = set([t.split(":")[-1] for t in track_uris])
    new_track_uris = list(ids - downloaded) 
    batches = list(chunks(new_track_uris, batch_size))
    fail = [] 
    #SCRAPING
    with tqdm(desc=f'loading audio metadata', unit='it', total=len(batches)) as pbar:
        for batch in batches:
            try: 
                time.sleep(5) 
                if not download_spotify_metadata(batch, output_path, sp): 
                    fail.append(batch)
            except SpotifyException as e:
                if e.http_status == 429:
                    print("'retry-after' value:", e.headers['retry-after'])
                    retry_value = e.headers['retry-after']
                    if int(e.headers['retry-after']) > 200:
                        print("STOP FOR TODAY, retry value too high {}".format(retry_value))
                        exit() 
                    else:
                        time.sleep(retry_values)
                        continue
                else:
                    continue
            pbar.update()
        pickle.dump(fail, open("failed_track_loads.pkl", 'wb'))
     
def scrape_track_audiodata(args): 
    #PRELIMINARIES 
    data_path, output_path = args.input, args.output
    if not os.path.exists(output_path): 
        os.makedirs(output_path, exist_ok=True)
        print(f'created {output_path} for store audio data jsons')
    #SETTING UP SPOTIFY READER CONNECTION 
    track_uris = pickle.load(open(data_path, 'rb'))

    session = requests.Session()
    retry = urllib3.Retry(
        respect_retry_after_header=False
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(),
                        requests_session=session)
    #SPLITTING DATA INTO BATCHES
    downloaded = __check_downloaded(output_path=output_path, mode='json')
    print(f"previously loaded {len(downloaded)} tracks")
    batch_size = 100 # Note; setting this too high will cause rate limit errors 

    ids = set([t.split(":")[-1] for t in track_uris])
    new_track_uris = list(ids - downloaded) 
    batches = list(chunks(new_track_uris, batch_size))
    #SCRAPING
    with tqdm(desc=f'loading audio metadata', unit='it', total=len(batches)) as pbar:
        for batch in batches:
            try: 
                time.sleep(5) 
                if not download_spotify_track_info(batch, output_path, sp): 
                    fail.append(batch)
            except SpotifyException as e:
                if e.http_status == 429:
                    print("'retry-after' value:", e.headers['retry-after'])
                    retry_value = e.headers['retry-after']
                    if int(e.headers['retry-after']) > 200:
                        print("STOP FOR TODAY, retry value too high {}".format(retry_value))
                        exit() 
                    else:
                        time.sleep(retry_values)
                        continue
                else:
                    continue
            pbar.update()
        pickle.dump(fail, open("failed_track_loads.pkl", 'wb'))
     
def scrape_genre_data(args): 
    #PRELIMINARIES 
    data_path, output_path = args.input, args.output
    uris = pickle.load(open(data_path, 'rb'))
    if not os.path.exists(output_path): 
        os.makedirs(output_path, exist_ok=True)
        print(f'created {output_path} for store artist jsons')
    #SETTING UP SPOTIFY READER CONNECTION 
    session = requests.Session()
    retry = urllib3.Retry(
        respect_retry_after_header=False
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(),
                        requests_session=session)
    downloaded = __check_downloaded(output_path=output_path, mode='json')
    print(f"previously loaded {len(downloaded)} tracks")
    fail = [] 
    batch_size = 40 
    ids = set([t.split(":")[-1] for t in uris])
    new_track_uris = list(ids - downloaded) 
    batches = list(chunks(new_track_uris, batch_size))
    start, end = batch_size, (batch_size) + (batch_size-1)
    for batch in tqdm(batches):
        try: 
            time.sleep(5) 
            if not download_spotify_artist_info(batch, output_path, sp): 
                fail.append(batch)
        except SpotifyException as e:
            if e.http_status == 429:
                print("'retry-after' value:", e.headers['retry-after'])
                retry_value = e.headers['retry-after']
                if int(e.headers['retry-after']) > 200:
                    print("STOP FOR TODAY, retry value too high {}".format(retry_value))
                    exit() 
                else:
                    time.sleep(retry_values)
                    continue
            else:
                continue
        pickle.dump(fail, open("failed_artist_loads.pkl", 'wb'))

def parse_scraping_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="which function to run")
    parser.add_argument("--input", type=str, help="data path")
    parser.add_argument("--output",type=str,help="output path")
    parser.add_argument('--mt', action='store_true', help='whether to run the multi-threading version of scrape function')
    return parser.parse_args()

if __name__ == '__main__': 
    args = parse_scraping_args()
    if args.mode == 'test_spotify': test_spotify()
    if args.mode == 'test_spotdl': test_spotdl()
    if args.mode == 'scrape_audio': scrape_audio(args) 
    if args.mode == 'scrape_metadata': scrape_track_metadata(args)
    if args.mode == 'find_artists': __collect_artists(args)
    if args.mode == 'scrape_music_feat': scrape_track_audiodata(args)
    if args.mode == 'scrape_genres': scrape_genre_data(args)
    

import json
import os
from pprint import *
from tqdm import tqdm
import sys
import pandas as pd 

#UTILS FOR MPD 

''' 
NOTE: CODE TAKEN FROM : https://github.com/tmscarla/spotify-recsys-challenge/blob/master/run/mpd_to_csv.py
''' 

path_load = "MPD DATA FOLDER" 

path_save = "WHERE TO SAVE CSVS" 

test_path = 'WHERE TO PUT TEST SET'

playlist_fields = ['pid','name', 'collaborative', 'modified_at', 'num_albums', 'num_tracks', 'num_followers',
'num_tracks', 'num_edits', 'duration_ms', 'num_artists','description']

track_fields = ['tid', 'arid' , 'alid', 'track_uri', 'track_name', 'duration_ms']

album_fields = ['alid','album_uri','album_name']

artist_fields = ['arid','artist_uri','artist_name']

interaction_fields = ['pid','tid','pos']

interactions = []
playlists = []
tracks = []
artists = []
albums = []

count_files = 0
count_playlists = 0
count_interactions = 0
count_tracks = 0
count_artists = 0
count_albums = 0
dict_tracks = {}
dict_artists = {}
dict_albums = {}


def process_mpd(path):
    global count_playlists
    global count_files
    filenames = os.listdir(path)
    for filename in tqdm(sorted(filenames)):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            process_info(mpd_slice['info'])
            for playlist in mpd_slice['playlists']:
                process_playlist(playlist)
                pid = playlist['pid']
                for track in playlist['tracks']:
                    track['pid']=pid
                    new = add_id_artist(track)
                    if new: process_artist(track)
                    new = add_id_album(track)
                    if new: process_album(track)
                    new = add_id_track(track)
                    if new: process_track(track)
                    process_interaction(track)
                count_playlists += 1
            count_files +=1

    show_summary()
    
def process_info(value):
    #print (json.dumps(value, indent=3, sort_keys=False))
    pass

def add_id_track(track):
    global count_tracks
    if track['track_uri'] not in dict_tracks:
        dict_tracks[track['track_uri']] = count_tracks
        track['tid'] = count_tracks
        count_tracks += 1
        return True
    else:
        track['tid'] = dict_tracks[track['track_uri']]
        return False

def add_id_artist(track):
    global count_artists
    if track['artist_uri'] not in dict_artists:
        dict_artists[track['artist_uri']] = count_artists
        track['arid'] = count_artists
        count_artists += 1
        return True
    else:
        track['arid'] = dict_artists[track['artist_uri']]
        return False

def add_id_album(track):
    global count_albums
    if track['album_uri'] not in dict_albums:
        dict_albums[track['album_uri']] = count_albums
        track['alid'] = count_albums
        count_albums += 1
        return True
    else:
        track['alid'] = dict_albums[track['album_uri']]
        return False

def process_track(track):
    global track_fields
    info = []
    for field in track_fields:
        info.append(track[field])
    tracks.append(info)

def process_album(track):
    global album_fields
    info = []
    for field in album_fields:
        info.append(track[field])
    albums.append(info)

def process_artist(track):
    global artist_fields
    info = []
    for field in artist_fields:
        info.append(track[field])
    artists.append(info)

def process_interaction(track):
    global interaction_fields
    global count_interactions
    info = []
    for field in interaction_fields:
        info.append(track[field])
    interactions.append(info)
    count_interactions +=1

def process_playlist(playlist):
    global playlist_fields
    if not 'description' in playlist:
        playlist['description'] = None
    info = []
    for field in playlist_fields:
        info.append(playlist[field])
    playlists.append(info)
               
def show_summary():
    print (f"# files:{count_files}")
    print (f"# playlists: {count_playlists}")
    print (f"# tracks: {count_tracks}")
    print (f"# artists: {count_artists}")
    print (f"# albums: {count_albums}")
    print (f"# interactions: {count_interactions}")

process_mpd(path_load)

import csv

with open(path_save+"artists.csv", "w") as f:
    writer = csv.writer(f,delimiter = "\t",)
    writer.writerow(artist_fields)
    writer.writerows(artists)
print ("artists.csv done")

with open(path_save+"albums.csv", "w") as f:
    writer = csv.writer(f,delimiter = "\t",)
    writer.writerow(album_fields)
    writer.writerows(albums)
print ("albums.csv done")
    
with open(path_save+"interactions.csv", "w") as f:
    writer = csv.writer(f,delimiter = "\t",)
    writer.writerow(interaction_fields)
    writer.writerows(interactions)
print ("interactions.csv done")

with open(path_save+"tracks.csv", "w") as f:
    writer = csv.writer(f,delimiter = "\t",)
    writer.writerow(track_fields)
    writer.writerows(tracks)
print ("tracks.csv done")

with open(path_save+"train_playlists.csv", "w") as f:
    writer = csv.writer(f,delimiter = "\t",)
    writer.writerow(playlist_fields)
    writer.writerows(playlists)
print ("train_playlists.csv done")


def process_test(): 
    data = json.load(open(test_path))
    # exclude = 
    test_set, pids = [], [] 
    for p in tqdm(data['playlists']) : 
        tracks = p['tracks']
        pid = p['pid']
        if len(tracks) == 100: 
            track_df = pd.DataFrame.from_records(tracks)
            test_set.append(track_df)
            pids.extend([pid]*len(track_df))

    
    all_test_df = pd.concat(test_set)
    ipdb.set_trace() 
    all_test_df['pid'] = pids
    all_test_df.to_csv(path_save+'test_interactions.csv', sep='\t')
    


train_interactions = pd.read_csv(path_save+"train_playlists.csv", sep='\t')
tracks = pd.read_csv(path_save+'tracks.csv', sep='\t')
int_with_train = pd.merge(train_interactions, tracks, on='tid')
int_with_train['track_uri'] = int_with_train['track_uri'].apply(lambda x: x.split(':')[-1])


#!/bin/bash

track_path=''
interaction_path=''
data_folder=''

#FORMATTING 
python -u extractions/tsv_to_csv.py \
-t $track_path \
-i $interaction_path \
-o $data_folder

#CREATING SPLITS 
python -u generate_splits.py \
--train_size 100000 \
—-valid_size 1000 \
—-test_size 1000 \
—-min_playlist_length 10 \
—-project_folder $data_folder

#SCRAPING AUDIO 
python -u scrape_audio.py \
--mode scrape_audio \
--input $data_folder+'/lfm/clean_org/train_uri.pkl' \
--output $data_folder+'/lfm/mp3/'

python -u scrape_audio.py \
--mode scrape_audio \
--input $data_folder+'/lfm/clean_org/valid_uri.pkl' \
--output $data_folder+'/lfm/mp3/'

python -u scrape_audio.py \
--mode scrape_audio \
--input $data_folder+'/lfm/clean_org/test_uri.pkl' \
--output $data_folder+'/lfm/mp3/'

#SCRAPING METADATA 
python -u scrape_audio.py \
--mode scrape_metadata \
--input $data_folder+'/lfm/clean_org/train_uri.pkl' \
--output $data_folder+'/lfm/temp_metadata/'

python -u scrape_audio.py \
--mode scrape_metadata \
--input $data_folder+'/lfm/clean_org/valid_uri.pkl' \
--output $data_folder+'/lfm/temp_metadata/'

python -u scrape_audio.py \
--mode scrape_metadata \
--input $data_folder+'/lfm/clean_org/test_uri.pkl' \
--output $data_folder+'/lfm/temp_metadata/'

#PROCESS AUDIO 
python process_data.py \
—-input_audio_folder $data_folder+'/lfm/mp3/' \
—-audio_output_folder $data_folder+'/lfm/trunc_mp3/' \
-—meta_output_folder $data_folder+'/lfm/trunc_meta/' \
--mt 

#SCRAPE GENRE FEATURES -- OPTIONAL
python -u scrape_audio.py \
--mode find_artists \
--input $data_folder+'/lfm/temp_metadata/' \
--output $data_folder+'/lfm/clean_org/' \`

python -u scrape_audio.py \
--mode scrape_genres \
--input $data_folder+'/lfm/clean_org/all_artist_uris.pkl' \
--output $data_folder+'/lfm/temp_genre/'

#SCRAPE MUSIC FEATURES -- OPTIONAL 
python -u scrape_audio.py \
--mode scrape_music_feat
--input $data_folder+'/lfm/clean_org/train_uri.pkl' \
--output $data_folder+'/lfm/temp_music/' \

python -u scrape_audio.py \
--mode scrape_music_feat
--input $data_folder+'/lfm/clean_org/valid_uri.pkl' \
--output $data_folder+'/lfm/temp_music/' \

python -u scrape_audio.py \
--mode scrape_music_feat
--input $data_folder+'/lfm/clean_org/test_uri.pkl' \
--output $data_folder+'/lfm/temp_music/' \

#GENERATE CAPTIONS 
python -u generate_captions.py \
--dataset_folder $data_folder+'/lfm/' \ 
--split $data_folder+'/lfm/clean_org/train_uri.pkl' \
--output_path $data_folder+'/lfm/caption_sets/'

python -u generate_captions.py \
--dataset_folder $data_folder+'/lfm/' \ 
--split $data_folder+'/lfm/clean_org/valid_uri.pkl' \
--output_path $data_folder+'/lfm/caption_sets/'

python -u generate_captions.py \
--dataset_folder $data_folder+'/lfm/' \ 
--split $data_folder+'/lfm/clean_org/test_uri.pkl' \
--output_path $data_folder+'/lfm/caption_sets/'

#RUN STATS CHECK -- OPTIONAL 
python -u stats.py \
--dataset_folder $data_folder+'/lfm/' 

#BUILD INTERACTION MATRICES 
python -u build_interactions.py \
--project_folder $data_folder+'/lfm/' 




# Building Environment


# Running our Code 
TBD 

# Accessing Checkpoints 
TBD 

# Dataset Building  
Since this dataloading procedure is very time consuming, we have provided piecewise code and a full service script.
 
## Full service script 
- you can run ```KDD2024-LARP/clean_databuild/runner.sh```
    - NOTE: you will need to manually change some paths in the runner.sh file. 
        - ```track_path```: should be the exact location of the tracks.csv file 
        - ```interaction_path```: should be the exact location of interaction.csv file 
        - ```data_folder```: should be the ```<project_folder>``` where you plan store all the data. This doesn't have be inside KDD2024-LARP. 
## Piecewise dataloading procedure 
1. Downloading original data
    - LFM 
        - Download data from: http://www.cp.jku.at/datasets/LFM-2b/
            -  NOTE: only files you need: spotify-uris.tsv.bz2  and listening-counts.tsv.bz2
        - Run ```bzip2 -d filename.bz2``` to unzip files
            - NOTE: the files are large so this may take a few minutes (especially for the interactions).
        - Rename files to follow proper structure: 
            - run ```KDD2024-LARP/clean_databuild/extractions/tsv_to_csv.py -t <track path> -i <interaction path> -o <data_folder>```
    - MPD 
        - Download data from: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files 
            - NOTE: you will need to make an account. 
            - NOTE: download both train and test sets. 
        - Rename files to follow propeor structure: 
            - run ```KDD2024-LARP/clean_databuild/extractions/json_to_csv.py```
2. Create dataset splits:  
    - run ```KDD2024-LARP/dataset_build/generate_splits.py --train_size 100_000 —valid_size 1_000 —test_size 1_000 —min_playlist_length 10 —project_folder <output_folder from previous step>```
3. Setup spotify credentials: 
    - Make an account here: https://developer.spotify.com/dashboard
    - Create a file ```/KDD2024-LARP/clean_build/secret.py``` where you store your ```client secret``` and ```client id```: 
        ```
        spotify_client_secret = <secret here> 
        spotify_client_id = <id here> 
        ```
4. Scrape for audio samples:
    - run: ```python scrape_audio.py --mode scrape_audio --input <project_folder>/<dataset>/clean_org/train_uri.pkl --output <project_folder>/<dataset>/mp3/```
        - NOTE: must be repeated for each split — train, valid, test
5. Scrape metadata associated with audio (if not already available): 
    - run: ```python scrape_audio.py --mode scrape_metadata --input <project_folder>/<dataset>/clean_org/train_uri.pkl --output <project_folder>/<dataset>/temp_metadata/```
        - NOTE: repeat for each split — train, valid, test 
6. Sample from full audio to create 10 second clips:
    - run: ```python process_data.py —input_audio_folder <project_folder>/<dataset>/mp3/ —output_audio_folder <project_folder>/<dataset>/trunc_mp3/ —meta_output_folder <project_folder>/<dataset>/trunc_meta/ --mt ```
7. Generate captions: 
    - run: ```python generate_captions.py --dataset_folder <project_folder>/<dataset>/ --split <project_folder>/<dataset>/train_uri.pkl --output_path <project_folder>/<dataset>/caption_sets/```
8. Generate interaction matrix for downstream evaluation: 
    - run: ```python -u build_interactions.py --project_folder <project_folder>/<dataset>/```

# Using your own dataset:
For seamless data loading, files should have the following structure: 
```
project_folder 

| -- MPD/ 
		| -- clean_org/  
					| -- train_uri.pkl 
					| -- valid_uri.pkl 
					| -- test_uri.pkl 
		| -- trunc_mp3/ 
					| -- <uri>/ 
							| -- <uri>_<sample_index>.mp3 
							| -- ...
					| ... 
		| -- caption_sets/ 
					| -- train_captions.pkl 
					| -- valid_captions.pkl 
					| -- test_captions.pkl 
		| -- evaluation_sets
					| -- bundle
							| -- bi_full.npy 
							| -- ... 
        | -- pair_sets/
					| ... 
		
| -- LFM/ 
		| ... 
```
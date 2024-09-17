This is the official repository for KDD 2024 submission *LARP: Language Audio Relational Pre-training for Cold-Start Playlist Continuation* <URL> 

If you use this codebase please cite us using: 
```
@inproceedings{LARP2024,
  author    = {Rebecca Salganik and
               Xiaohao Liu and
               Jian Kang and
               Yunshan Ma and
               Tat{-}Seng Chua},
  title     = {LARP: Language Audio Relational Pre-training 
                for Cold-Start Playlist Continuation},
  booktitle = {{KDD}},
  publisher = {{ACM}},
  year      = {2024}
}
```

# Table of Contents 
- Building environment 
- Running pre-trained model on your data 
- Building our datasets from scratch  
 

# Building Environment
We find that installing in two phases was most effective for package compatibility. 
1. Run ``` conda env create --name envname --file=/KDD2024-LARP/environment_setup/conda.yml```
2. Activate env. Run ```pip install  /KDD2024-LARP/environment_setup/pip.txt ``` 
 
# Running pre-trained model on your data  
1. We have added two sample files to provide examples of the format required by our model. Audio files should be stored in .mp3 format and captions should be saved in a .json file that contains all the stri
1. Start by downloading the pre-trained HTSAT checkpoint at https://drive.google.com/drive/folders/1SMQyzJvc6DwJNuhQ_WI8tlCFL5HG2vk6. Then go into the ```/KDD2024-LARP/configs/config2.yaml``` file and manually change the path under ```MODELS>AUDIO_MODELS>HTSAT>checkpoint``` entry. 
2. Download our checkpoints from: https://huggingface.co/Rsalga/LARP/tree/main 
3. Manually input your audio and caption folders inside the ```run_pretrained.py``` file 
4. Launch: ```python run_pretrained.py --checkpoint <checkpoint path>```




# Dataset Building 
Unfortunately, due to legal restrictions we cannot post the mp3 files that were used during our training. However, we provide all of the elements necessary to recreate our code exactly. If you have any issues please feel free to reach out to our primary author, Rebecca Salganik. 

We store the exact song-caption pairs in the ```datasets/``` folder of this repository. These uris can be used in tandem with the code in the ```databuild_utils/``` to download the mp3. Furthermore, to replicate the truncation procedure we store the exact start and end of each 10 second sample using the keys ```start``` and ```end```. 

In order to facilitate your audio loading, we provide two versions of the code - one that can be batched using a scheduler such as SLURM and another which can be done slowly on a personal computer. 
 
## Full service script 
- you can run ```KDD2024-LARP/databuild_utils/runner.sh```
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
            - run ```KDD2024-LARP/databuild_utils/extractions/tsv_to_csv.py -t <track path> -i <interaction path> -o <data_folder>```
    - MPD 
        - Download data from: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files 
            - NOTE: you will need to make an account. 
            - NOTE: download both train and test sets. 
        - Rename files to follow propeor structure: 
            - run ```KDD2024-LARP/databuild_utils/extractions/json_to_csv.py```
2. Create dataset splits:  
    - run ```KDD2024-LARP/dataset_build/generate_splits.py --train_size 100_000 —valid_size 1_000 —test_size 1_000 —min_playlist_length 10 —project_folder <output_folder from previous step>```
3. Setup spotify credentials: 
    - Make an account here: https://developer.spotify.com/dashboard
    - Create a file ```/KDD2024-LARP/databuild_utils/secret.py``` where you store your ```client secret``` and ```client id```: 
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


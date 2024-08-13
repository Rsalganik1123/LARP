import argparse 
# from dataset_build.fastchat.serve.inference import add_model_args


def parse_feature_gen_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type='str')





import argparse 
def parse_feature_gen_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type='str')


def parse_args(no_cmd=False): 
    parser = argparse.ArgumentParser()
    #Basic Run Commands --> changed with each experiment
    parser.add_argument('--dataset', type=str, default='MPD', choices=['MPD', 'LFM'], help='dataset for training/eval')
    parser.add_argument('--gpu', type=str, default="0,", help='specifying which gpu nodes to use, see README for more details')
    parser.add_argument('--world_size', type=int, default=1, help='number of distributed nodes to run, see README for more details')
    parser.add_argument('--save_every', default = 1, type=int, help='epoch interval at which to save checkpoints')
    parser.add_argument('--validation_interval', default=1, type=int, help='epoch interval at which to run validation loop')
    parser.add_argument('--verbose', action="store_true", help='print statements during training')
    parser.add_argument('--info', required=False, nargs="+", default=[], help='comments to add to experiment names')
    parser.add_argument('--output_path', required=False, help='path for storing intermediate embeddings [intended for debug use only]')
    parser.add_argument('--early_stop', type=int, default=5, help='number of epochs without improvement before terminating with early stopping')
    parser.add_argument('--ablation_loss', type=str, default='itc,itm,lm', help='loss with which to train model, see README')
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_truncate', type=int, default=10, help='number of tracks used to generate playlist encodings')

    #Predefined Hyperparameters --> change once 
    parser.add_argument('--log_base', default = '/home/rebecca/rebecca_workplace/TENSORBOARD', help='folder path for storing logs')
    parser.add_argument('--dataset_path', type=str, default='/mnt/ssd1/rebecca/', help='location where data is stored, see README for required file structure')
    parser.add_argument('--config_path', type=str, default='/home/xhliu/KDD2024-LARP/configs/config2.yaml', help='hyperparameter settings')
    parser.add_argument('--checkpoint', type=str, default=None, help='path for storing checkpoints')
    parser.add_argument('--master_port', required=False, type=str, default="12355", help='if you launch multiple distributed runs, you will need to change this')
    
    #Required For Run --> don't recommend changing 
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'TTC', 'TPC'], help='which loss to train with') 
    parser.add_argument('--base_model', type=str, default='LARP', choices=['LARP'])
    parser.add_argument('--audio_model', default='HTSAT', help='backbone model used for audio modality')
    parser.add_argument('--text_model', default='BERT', help='backbone model used for text modality')
    parser.add_argument('--caption_mode', type=str, default='meta', choices=['meta'], help='caption style, future work to extend options')
    parser.add_argument('--audio_mode', type=str, default='file_path_ncl', choices=['file_path_ncl'], help='key used to lookup mp3 file locations during dataset build')
    parser.add_argument('--eval_mode', type=str, nargs = "+", default=['bundle'], choices = ['bundle'], help='function used for early stopping and validation')
    parser.add_argument('--decision_metric', type=str, default='bundle', choices = ['bundle'], help='decision metric to use for early stopping evaluation') 
    parser.add_argument('--feat_fusion', type=str, default='avg', choices=['avg', 'concat'])
    parser.add_argument('--seed', default=42, type=int, help='seed for reproducibility')
    parser.add_argument('--fusion_method', type=str, default="self_attn", choices=["average", "soft_weight", "self_attn"])
    
    # parser.add_argument('--distributed', action="store_true", help='whether to run distributed training (improved speed)')
    # parser.add_argument('--dataset_size', type=str, default='TOY', choices=['TOY', 'FULL'], help='dataset for training/eval')
    # parser.add_argument('--soft_weight', action='store_true')
    
    # parser.add_argument('--session', action='store_true')
    # parser.add_argument('--trunc', action='store_true')
    
    
    # can delte it later. [2024,01,16]
    if no_cmd:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    
    return args

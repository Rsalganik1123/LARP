import argparse 
# from dataset_build.fastchat.serve.inference import add_model_args


def parse_feature_gen_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type='str')


def vicuna_parse_args(): 
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--message", type=str, default="Hello! Who are you?")

    parser.add_argument('--input_path', type=str, default='/home/rebecca/vicuna_caption_info.json',
                        help='input json')
    parser.add_argument('--output_path', type=str, default='/ssd1/ccye/vicuna-event/',
                        help='output data directory')
    parser.add_argument('--split', type=str, default='all')


    args = parser.parse_args()
    return args


import argparse 
def parse_feature_gen_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type='str')


def parse_args(no_cmd=False): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'TTC', 'TPC']) #old: 'CF_train', 'CF_ego_train',
    parser.add_argument('--base_model', type=str, default='BLAP', choices=['BLAP', 'CLAP', 'jukebox', 'mule', 'HTSAT', 'BERT', 'RESNET'])
    parser.add_argument('--audio_model', default='HTSAT', help='backbone model used for audio modality')
    parser.add_argument('--text_model', default='BERT', help='backbone model used for text modality')
    parser.add_argument('--dataset', type=str, default='MPD', choices=['MPD', 'MUSIC_CAPS', "EMO", 'LFM'], help='dataset for training/eval')
    parser.add_argument('--dataset_size', type=str, default='TOY', choices=['TOY', 'FULL'], help='dataset for training/eval')
    parser.add_argument('--caption_mode', type=str, default='meta', choices=['meta', 'genre', 'audio', 'combo', 'shuffle', 'None'], help='caption style, please read docs for more details')
    parser.add_argument('--audio_mode', type=str, default='file_path_ncl', choices=['full', 'random_slice', 'first_slice', 'file_path_ncl', 'short_path'], help='format of audio sample for generating audio emb')
    parser.add_argument('--eval_mode', type=str, nargs = "+", default=['bundle', 'retrieval'], choices = ['bundle', 'retrieval'], help='which function to run for evaluating')
    parser.add_argument('--decision_metric', type=str, default='bundle', choices = ['bundle', 'retrieval'], help='which decision metric to use for early stopping evaluation') 
    parser.add_argument('--feat_fusion', type=str, default='avg', choices=['avg', 'concat'])
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path for BLIP model')
    parser.add_argument('--log_base', default = '/home/rebecca/rebecca_workplace/TENSORBOARD', help='folder path for storing logs')
    parser.add_argument('--seed', default=42, type=int, help='seed for reproducibility')
    parser.add_argument('--gpu', type=str, default="0,", help='specifying which gpu nodes to use')
    parser.add_argument('--world_size', type=int, default=2, help='number of distributed nodes to run')
    parser.add_argument('--save_every', default = 1, type=int, help='epoch interval at which to save checkpoints')
    parser.add_argument('--validation_interval', default=1, type=int, help='epoch interval at which to run validation loop')
    parser.add_argument('--distributed', action="store_true", help='whether to run distributed training (improved speed)')
    parser.add_argument('--verbose', action="store_true", help='print statements during training')
    parser.add_argument('--info', required=False, nargs="+", default=[], help='comments to add to experiment names')
    parser.add_argument('--output_path', required=False)
    parser.add_argument('--master_port', required=False, type=str, default="12355", help='if you launch multiple distributed runs, you will need to change this')
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--ablation_loss', type=str, default='itc,itm,lm') 
    parser.add_argument('--soft_weight', action='store_true')
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--session', action='store_true')
    parser.add_argument('--trunc', action='store_true')
    parser.add_argument('--num_truncate', type=int, default=10)
    parser.add_argument('--fusion_method', type=str, default="self_attn", choices=["average", "soft_weight", "self_attn"])
    parser.add_argument('--dataset_path', type=str, default='/mnt/ssd1/rebecca/')

    # can delte it later. [2024,01,16]
    if no_cmd:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    
    return args

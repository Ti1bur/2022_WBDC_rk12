import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')

    # ========================= Data Configs ==========================
    parser.add_argument('--train_annotation', type=str, default='/opt/ml/input/data/annotations/labeled.json')
    parser.add_argument('--test_annotation', type=str, default='/opt/ml/input/data/annotations/test.json')
    parser.add_argument('--train_zip_frames', type=str, default='/opt/ml/input/data/zip_frames/labeled/')
    parser.add_argument('--test_zip_frames', type=str, default='/opt/ml/input/data/zip_frames/test/')
    parser.add_argument('--test_output_csv', type=str, default='/opt/ml/output/result.csv')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=32, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=128, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=32, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=8, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='./save/vit-14-ema-pgd3-pretrain6-origin-dataset')
    parser.add_argument('--ckpt_file', type=str, default='./save/vit-14-ema-pgd3-pretrain6-origin-dataset/double_model.bin')
    parser.add_argument('--best_score', default=0.5, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=5, help='How many epochs')
    parser.add_argument('--max_steps', default=17000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1700, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument('--bert_learning_rate', default=7e-5, type=float, help='initial learning rate')
    parser.add_argument('--cross_learning_rate', default=8e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    # ========================== Swin ===================================
    parser.add_argument('--swin_pretrained_path', type=str, default='../../opensource_models/swin_tiny_patch4_window7_224.pth')

    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='../../opensource_models/roberta-wwm-ext')
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    parser.add_argument('--bert_seq_length', type=int, default=258)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)

    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--max_frames', type=int, default=14)
    parser.add_argument('--vlad_cluster_size', type=int, default=64)
    parser.add_argument('--vlad_groups', type=int, default=4)
    parser.add_argument('--vlad_hidden_size', type=int, default=1024, help='nextvlad output size using dense')
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')

    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=768, help="linear size before final linear")
    
    # ========================== trick =============================
    parser.add_argument('--confrontation', type=str, default='pgd', help="confrontation way")
    parser.add_argument('--use_ema', type=str, default='True', help="use ema")
    parser.add_argument('--use_fp16', type=str, default='True', help="use fp16")
    parser.add_argument('--Gradient_Accumulation_step', type=int, default=1, help="use_Gradient_Accumulation")
    parser.add_argument('--use_kfold', type=str, default='False', help="use kfold")
    parser.add_argument('--kfold', type=int, default=5, help="fold num")
    parser.add_argument('--freeze', type=bool, default='True', help="Freeze visiual_backbone")
    parser.add_argument('--distill', type=bool, default=False, help="work distill")
    parser.add_argument("--alpha", default=0.3, type=float, help="distill loss")
    
    # ========================== Pretrain =============================
    parser.add_argument('--ispretrain', type=bool, default=False)
    parser.add_argument('--bert_config', type=str, default='../../opensource_models/roberta_wwm_ext/config.json')
    parser.add_argument('--image_res', default=224, type=int)
    parser.add_argument('--vision_width', default=512, type=int)
    parser.add_argument('--embed_dim', default=768, type=int)
    parser.add_argument("--temp", default=0.07, type=float)    
    parser.add_argument("--mlm_probability", default=0.15, type=float)  
    # parser.add_argument('--queue_size', default=10368, type=int)
    parser.add_argument('--queue_size', default=8192, type=int)
    parser.add_argument('--momentum', default=0.995, type=float)
    parser.add_argument('--local_rank', default=-1, type=int) 

    return parser.parse_args()
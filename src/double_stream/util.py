import logging
import random

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from category_id_map import lv2id_to_lv1id


def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    if args.ispretrain:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


def build_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    
    bert = []
    cross = []
    other = []
    
    for name, param in model.named_parameters():
        space = name.split('.')
        if space[0] == 'bert':
            bert.append((name, param))
        elif space[0] == 'multi_head_decoder':
            cross.append((name, param))
        else:
            other.append((name, param))
      
    optimizer_grouped_parameters = [
        {'params' : [p for n, p in bert if not any(nd in n for nd in no_decay)],
         'weight_decay':args.weight_decay, 'lr':args.bert_learning_rate},
        {'params' : [p for n, p in bert if any(nd in n for nd in no_decay)],
         'weight_decay':0.0, 'lr':args.bert_learning_rate},
        
        {'params' : [p for n, p in cross if not any(nd in n for nd in no_decay)],
         'weight_decay':args.weight_decay, 'lr':args.cross_learning_rate},
        {'params' : [p for n, p in cross if any(nd in n for nd in no_decay)],
         'weight_decay':0.0, 'lr':args.cross_learning_rate},
        
        {'params' : [p for n, p in other if not any(nd in n for nd in no_decay)],
         'weight_decay':args.weight_decay, 'lr':args.learning_rate},
        {'params' : [p for n, p in other if any(nd in n for nd in no_decay)],
         'weight_decay':0.0, 'lr':args.learning_rate},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    return optimizer, scheduler


def evaluate(predictions, labels):
    # prediction and labels are all level-2 class ids

    lv1_preds = [lv2id_to_lv1id(lv2id) for lv2id in predictions]
    lv1_labels = [lv2id_to_lv1id(lv2id) for lv2id in labels]

    lv2_f1_micro = f1_score(labels, predictions, average='micro')
    lv2_f1_macro = f1_score(labels, predictions, average='macro')
    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

    eval_results = {'lv1_acc': accuracy_score(lv1_labels, lv1_preds),
                    'lv2_acc': accuracy_score(labels, predictions),
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    'mean_f1': mean_f1}

    return eval_results

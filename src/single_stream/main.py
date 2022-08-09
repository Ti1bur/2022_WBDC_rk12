import logging
import os
import time
import torch
from tqdm import tqdm
from model import MultiModal, FGM, PGD, EMA
from config import parse_args
from data_helper import create_dataloaders
#from helper_frames import create_dataloaders
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from pretrain_model import MultiModalForPretrain
import warnings
from collections import OrderedDict
import pdb
import torch.nn.functional as F

def load_param(args, model):
    pretrain = MultiModalForPretrain(model_path=args.bert_dir)
    chkpoint = torch.load('./save/pretrain/pretrain_epoch30.bin', map_location='cpu')

    name_params = OrderedDict()
    for name, param in chkpoint.items():
        key = '.'.join(name.split('.')[1:])
        name_params[key] = param

    pretrain.load_state_dict(name_params)
    cur_name_param_bert = OrderedDict()
    cur_name_param_fc = OrderedDict()
    for name, param in pretrain.named_parameters():
        if name.split('.')[0] == 'roberta':
            key = '.'.join(name.split('.')[2:])
            cur_name_param_bert[key] = param
        if name.split('.')[0] == 'video_fc':
            key = '.'.join(name.split('.')[1:])
            cur_name_param_fc[key] = param
    model.bert.load_state_dict(cur_name_param_bert, strict=False)
    model.video_fc.load_state_dict(cur_name_param_fc, strict=False)
    return model

def cal_loss(prediction, label):
    label = label.squeeze(dim=1)
    loss = F.cross_entropy(prediction, label)
    with torch.no_grad():
        pred_label_id = torch.argmax(prediction, dim=1)
        accuracy = (label == pred_label_id).float().sum() / label.shape[0]
    return loss, accuracy, pred_label_id, label

def validate(model, val_dataloader, args, autocast):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            with autocast():
                # loss, _, pred_label_id, label = model(batch)
                prediction = model(batch['title_input'], batch['title_mask'], batch['frame_input'], batch['frame_mask'])
                loss, _, pred_label_id, label = cal_loss(prediction, batch['label'].cuda())
                loss = loss.mean()
                predictions.extend(pred_label_id.cpu().numpy())
                labels.extend(label.cpu().numpy())
                losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args):
# def train_and_validate(args,train_dataloader, val_dataloader, fold_num):
    # 1. load data
    os.makedirs(args.savedmodel_path, exist_ok=True)
    args.ispretrain = False
    args.train_annotation = '/opt/ml/input/data/annotations/labeled.json'
    args.train_zip_frames = '/opt/ml/input/data/zip_feats/labeled_vit.zip'
    train_dataloader, val_dataloader = create_dataloaders(args)
    args.max_steps = len(train_dataloader) * args.max_epochs
    args.warmup_steps = args.max_steps * 0.06
    # 2. build model and optimizers
    model = MultiModal(args)
    model = load_param(args, model)
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    # fgm = FGM(model)
    pgd = PGD(model)
    pgd_k = 3
    ema = EMA(model, 0.995)
    ema.register()
    # 3. training
    step = 0
    best_score = args.best_score
    # start_time = time.time()
    for epoch in range(args.max_epochs):
        for batch in tqdm(train_dataloader):
            model.train()
            with autocast():
                # print(batch['title_input'].shape, batch['title_mask'].shape, batch['frame_input'].shape, batch['frame_mask'].shape)
                prediction = model(batch['title_input'], batch['title_mask'], batch['frame_input'], batch['frame_mask'])
                loss, accuracy, _, _ = cal_loss(prediction, batch['label'].cuda())
                loss = loss.mean()
                accuracy = accuracy.mean()
            scaler.scale(loss).backward()
            # with autocast():
            #     fgm.attack()
            #     # loss_adv, acc, _, _ = model(batch)
            #     prediction = model(batch['title_input'], batch['title_mask'], batch['frame_input'], batch['frame_mask'])
            #     loss_adv, acc, _, _ = cal_loss(prediction, batch['label'].cuda())
            #     loss_adv = loss_adv.mean()
            # scaler.scale(loss_adv).backward()
            # fgm.restore()
            pgd.backup_grad()
            # 对抗训练
            for t in range(pgd_k):
                pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != pgd_k-1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                with autocast():
                    prediction_adv = model(batch['title_input'], batch['title_mask'], batch['frame_input'], batch['frame_mask'])
                    loss_adv, accuracy_adv, _, _ = cal_loss(prediction_adv, batch['label'].cuda())
                loss_adv = loss_adv.mean()
                scaler.scale(loss_adv).backward()
            pgd.restore() # 恢复embedding参数

            
            scaler.step(optimizer)
            optimizer.zero_grad()
            scheduler.step()
            scaler.update()
            ema.update()
            
            step += 1
            if step % 500 == 0 and step >= 3000:
                ema.apply_shadow()
                loss, results = validate(model, val_dataloader, args, autocast)
                results = {k: round(v, 4) for k, v in results.items()}
                logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
                mean_f1 = results['mean_f1']
                state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                torch.save({'step': step, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                       f'{args.savedmodel_path}/model_step_{step}_mean_f1_{mean_f1}.bin')
                ema.restore()
        
#         ema.apply_shadow()
#         # 4. validation
#         loss, results = validate(model, val_dataloader, autocast)
#         results = {k: round(v, 4) for k, v in results.items()}
#         logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

#         # 5. save checkpoint
#         mean_f1 = results['mean_f1']
#         best_score = mean_f1
#         state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
#         torch.save({'fold_num': fold_num, 'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
#                     f'{args.savedmodel_path}/model_fold_{fold_num}_epoch_{epoch}_mean_f1_{mean_f1}.bin')
        
#         ema.restore()

def val(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)
    # 2. build model and optimizers
    model = MultiModal(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    
    if 'model.bin' not in args.ckpt_file:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    autocast = torch.cuda.amp.autocast
    # 4. validation
    model.eval()
    with torch.no_grad():
        loss, results = validate(model, val_dataloader, args, autocast)
        results = {k: round(v, 4) for k, v in results.items()}
    # 5. save checkpoint
    mean_f1 = results['mean_f1']
    logging.info(f"validate mean_f1 = {mean_f1}")

def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)
    
    # args.train_annotation = '/home/tione/notebook/data/annotations/1train.json'
    # args.valid_annotation = '/home/tione/notebook/data/annotations/1valid.json'
    # train_dataloader, val_dataloader = create_dataloaders(args)
    # train_and_validate(args, train_dataloader, val_dataloader, 1)
    # args.train_annotation = '/home/tione/notebook/data/annotations/2train.json'
    # args.valid_annotation = '/home/tione/notebook/data/annotations/2valid.json'
    # train_dataloader1, val_dataloader1 = create_dataloaders(args)
    # train_and_validate(args, train_dataloader1, val_dataloader1, 2)
    # args.train_annotation = '/home/tione/notebook/data/annotations/3train.json'
    # args.valid_annotation = '/home/tione/notebook/data/annotations/3valid.json'
    # train_dataloader2, val_dataloader2 = create_dataloaders(args)
    # train_and_validate(args, train_dataloader2, val_dataloader2, 3)
    # args.train_annotation = '/home/tione/notebook/data/annotations/4train.json'
    # args.valid_annotation = '/home/tione/notebook/data/annotations/4valid.json'
    # train_dataloader3, val_dataloader3 = create_dataloaders(args)
    # train_and_validate(args, train_dataloader3, val_dataloader3, 4)
    train_and_validate(args)
    # val(args)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()

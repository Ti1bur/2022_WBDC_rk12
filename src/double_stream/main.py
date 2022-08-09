import logging
import os
import time
import torch
from tqdm import tqdm
from model import MultiModal, FGM, PGD, EMA
from pretrain_ALBEF import ALBEF
from config import parse_args
from data_helper import create_dataloaders
from collections import OrderedDict
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
import pdb
import torch.nn.functional as F

def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        # loss = CrossEntropyLoss_label_smooth(prediction, label)
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
            if args.use_fp16 == 'True':
                with autocast():
                    prediction = model(batch['title_input'], batch['title_mask'], batch['frame_input'], batch['frame_mask'])
                    
                    loss, accuracy, pred_label_id, label = cal_loss(prediction, batch['label'].cuda())
                    # loss = (1-args.alpha)*loss - args.alpha*torch.sum(F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1)
            else:
                prediction = model(batch['title_input'], batch['title_mask'], batch['frame_input'], batch['frame_mask'])
                    
                loss, accuracy, pred_label_id, label = cal_loss(prediction, batch['label'].cuda())
                # loss = (1-args.alpha)*loss - args.alpha*torch.sum(F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    # import pdb
    # pdb.set_trace()
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results

def freeze(args, model):
    if args.freeze == True:
        for name, param in model.named_parameters():
            print(name)
            if 'visual_backbone' in name:
                param.requires_grad=False
    return model

def load_param(args, model):
    pretrain = ALBEF(config=args)
    chkpoint = torch.load('./save/pretrain/pretrain_epoch6.bin', map_location='cpu')

    name_params = OrderedDict()
    for name, param in chkpoint.items():
        key = '.'.join(name.split('.')[0:])
        name_params[key] = param

    pretrain.load_state_dict(name_params)
    cur_name_param_bert = OrderedDict()
    cur_name_param_fc = OrderedDict()
    cur_name_param_mhlayer = OrderedDict()
    cur_name_param_mh = OrderedDict()
    for name, param in pretrain.named_parameters():
        if name.split('.')[0] == 'text_encoder':
            key = '.'.join(name.split('.')[2:])
            cur_name_param_bert[key] = param
        if name.split('.')[0] == 'visual_encoder':
            key = '.'.join(name.split('.')[1:])
            cur_name_param_fc[key] = param
        if name.split('.')[0] == 'multi_head_decoderlayer':
            key = '.'.join(name.split('.')[1:])
            cur_name_param_mhlayer[key] = param
        if name.split('.')[0] == 'multi_head_decoder':
            key = '.'.join(name.split('.')[1:])
            cur_name_param_mh[key] = param
    model.bert.load_state_dict(cur_name_param_bert, strict=False)
    model.visual_encoder.load_state_dict(cur_name_param_fc, strict=False)
    model.multi_head_decoderlayer.load_state_dict(cur_name_param_mhlayer, strict=False)
    model.multi_head_decoder.load_state_dict(cur_name_param_mh, strict=False)
    
    return model
        
# def train_and_validate(args, train_dataloader, val_dataloader, epoch_num):
def train_and_validate(args):
    # 1. load data
    args.train_annotation = '/opt/ml/input/data/annotations/labeled.json'
    args.train_zip_frames = '/opt/ml/input/data/zip_feats/labeled_vit.zip'
    train_dataloader, val_dataloader = create_dataloaders(args)
    args.max_steps = args.max_epochs * len(train_dataloader)
    args.warmup_steps = args.max_steps * 0.06
    # 2. build model and optimizers
    model = MultiModal(args)
    model = load_param(args,model)
    # model = freeze(args, model)

    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
        if args.confrontation == 'fgm':
            fgm = FGM(model)
        if args.confrontation == 'pgd':
            pgd = PGD(model)
            pgd_k = 3
        if args.use_ema == 'True':
            ema = EMA(model, 0.995)
            ema.register()
        if args.use_fp16 == 'True':
            scaler = torch.cuda.amp.GradScaler()
            autocast = torch.cuda.amp.autocast
    # 3. training
    step = 0
    best_score = args.best_score
    # 开始训练
    for epoch in range(args.max_epochs):
        for i, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            if args.use_fp16 == 'True':
                with autocast():
                    prediction = model(batch['title_input'], batch['title_mask'], batch['frame_input'], batch['frame_mask'])
                    
                    loss, accuracy, _, _ = cal_loss(prediction, batch['label'].cuda())
                    # loss = (1-args.alpha)*loss - args.alpha*torch.sum(F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1)
            else:
                prediction = model(batch['title_input'], batch['title_mask'], batch['frame_input'], batch['frame_mask'])
                loss, accuracy, _, _ = cal_loss(prediction, batch['label'].cuda())
                # loss = (1-args.alpha)*loss - args.alpha*torch.sum(F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1)
            loss = loss.mean()
            accuracy = accuracy.mean()
            if args.use_fp16 == 'True':
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if epoch != 0 and args.confrontation == 'fgm':
                # 对抗训练
                fgm.attack() # 在embedding上添加对抗扰动
                if args.use_fp16 == 'True':
                    with autocast():
                        prediction_adv = model(batch['title_input'], batch['title_mask'], batch['frame_input'], batch['frame_mask'])
                    
                        loss_adv, accuracy_adv, _, _ = cal_loss(prediction_adv, batch['label'].cuda())
                        # loss_adv = (1-args.alpha)*loss_adv - args.alpha*torch.sum(F.log_softmax(prediction_adv, dim=1)*F.softmax(prediction_m, dim=1),dim=1)
                else:
                    prediction_adv = model(batch['title_input'], batch['title_mask'], batch['frame_input'], batch['frame_mask'])
                    
                    loss_adv, accuracy_adv, _, _ = cal_loss(prediction_adv, batch['label'].cuda())
                    # loss_adv = (1-args.alpha)*loss_adv - args.alpha*torch.sum(F.log_softmax(prediction_adv, dim=1)*F.softmax(prediction_m, dim=1),dim=1)
                loss_adv = loss_adv.mean()
                accuracy_adv = accuracy_adv.mean()
                if args.use_fp16 == 'True':
                    scaler.scale(loss_adv).backward()
                else:
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore() # 恢复embedding参数
                accuracy = (accuracy + accuracy_adv)/2
            if epoch != 0 and args.confrontation == 'pgd':
                pgd.backup_grad()
                # 对抗训练
                accuracy_adv_all=0
                for t in range(pgd_k):
                    pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != pgd_k-1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    if args.use_fp16 == 'True':
                        with autocast():
                            prediction_adv = model(batch['title_input'], batch['title_mask'], batch['frame_input'], batch['frame_mask'])
                            loss_adv, accuracy_adv, _, _ = cal_loss(prediction_adv, batch['label'].cuda())
                    else:
                        prediction_adv = model(batch['title_input'], batch['title_mask'], batch['frame_input'], batch['frame_mask'])
                        loss_adv, accuracy_adv, _, _ = cal_loss(prediction_adv, batch['label'].cuda())
                    loss_adv = loss_adv.mean()
                    # accuracy_adv = accuracy_adv.mean()
                    # accuracy_adv_all += accuracy_adv
                    if args.use_fp16 == 'True':
                        scaler.scale(loss_adv).backward()
                    else:
                        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                pgd.restore() # 恢复embedding参数
                # accuracy = (accuracy + accuracy_adv_all/pgd_k)/2

            if args.use_fp16 == 'True':
                scaler.step(optimizer)
            else:
                optimizer.step()
            if args.use_ema == 'True':
                ema.update()
            optimizer.zero_grad()
            scheduler.step()
            if args.use_fp16 == 'True':
                scaler.update()

            step += 1
            if step % 500 == 0 and step >= 3000:
                if args.use_ema == 'True':
                    ema.apply_shadow()
                loss, results = validate(model, val_dataloader, args, autocast)
                results = {k: round(v, 4) for k, v in results.items()}
                logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
                mean_f1 = results['mean_f1']
                state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                torch.save({'step': step, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                       f'{args.savedmodel_path}/model_step_{step}_mean_f1_{mean_f1}.bin')
                # torch.save({'step': step, 'model_state_dict': state_dict},
                #        f'{args.savedmodel_path}/model_step_{step}.bin')
                if args.use_ema == 'True':
                    # 恢复
                    ema.restore()

        # 4. validation
        # if args.use_ema == 'True':
        #     ema.apply_shadow()
        # loss, results = validate(model, val_dataloader, args, autocast)
#         # round() 方法返回浮点数x的四舍五入值，精确到小数点后四位
#         results = {k: round(v, 4) for k, v in results.items()}
        
#         logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

        # 5. save checkpoint
        # mean_f1 = results['mean_f1']
        # if mean_f1 > best_score:
        #     state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
        #     torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
        #                f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')
        # state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
        # torch.save({'epoch':epoch, 'model_state_dict' : state_dict}, f'{args.savedmodel_path}/model_epoch_{epoch}.bin')
        

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
    loss, results = validate(model, val_dataloader, args, autocast)
    results = {k: round(v, 4) for k, v in results.items()}
    
    # 5. save checkpoint
    mean_f1 = results['mean_f1']
    logging.info(f"validate mean_f1 = {mean_f1}")
            
def main():
    args = parse_args()
    args.ispretrain = False
    setup_logging()
    setup_device(args)
    setup_seed(args)
    args.savedmodel_path = './save/vit-14-ema-pgd3-pretrain6-origin-dataset'
    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)
    
    # args.train_annotation = '/home/tione/notebook/data/annotations/1train.json'
    # args.valid_annotation = '/home/tione/notebook/data/annotations/1valid.json'
    # train_dataloader, val_dataloader = create_dataloaders(args)
    # train_and_validate(args, train_dataloader, val_dataloader, 1)
    # args.train_annotation = '/home/tione/notebook/data/annotations/2train.json'
    # args.valid_annotation = '/home/tione/notebook/data/annotations/2valid.json'
    # train_dataloader1, val_dataloader1 = create_dataloaders(args)
    # # train_and_validate(args, train_dataloader1, val_dataloader1, 2)
    # args.train_annotation = '/home/tione/notebook/data/annotations/3train.json'
    # args.valid_annotation = '/home/tione/notebook/data/annotations/3valid.json'
    # train_dataloader2, val_dataloader2 = create_dataloaders(args)
    # train_and_validate(args, train_dataloader2, val_dataloader2, 3)
    # args.train_annotation = '/home/tione/notebook/data/annotations/4train.json'
    # args.valid_annotation = '/home/tione/notebook/data/annotations/4valid.json'
    # train_dataloader3, val_dataloader3 = create_dataloaders(args)
    # train_and_validate(args, train_dataloader3, val_dataloader3, 4)
    # args.train_annotation = '/home/tione/notebook/data/annotations/5train.json'
    # args.valid_annotation = '/home/tione/notebook/data/annotations/5valid.json'
    # train_dataloader4, val_dataloader4 = create_dataloaders(args)
    # train_and_validate(args, train_dataloader4, val_dataloader4, 5)
    
    train_and_validate(args)
    # val(args)

if __name__ == '__main__':
    main()

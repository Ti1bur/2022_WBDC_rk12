import logging
import os
import time
import torch
from tqdm import tqdm
from config import parse_args
from data_helper import create_dataloaders
from pretrain_model import MultiModalForPretrain
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
import numpy as np
import warnings
from imp import reload
import pdb
reload(logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(f"./log/train_{time.strftime('%m%d_%H%M', time.localtime())}.log"),
        logging.StreamHandler()
    ]
)

def get_pred_and_loss(model, item, task=None):
    """Get pred and loss for specific task"""
    video_feature = item['frame_input'].to('cuda')
    input_ids = item['title_input'].to('cuda')
    attention_mask = item['title_mask'].to('cuda')
    video_mask = item['frame_mask'].to('cuda')

    target = None
    if 'label' in item:
        target = item['label'].to('cuda')

    loss, mlm, mfm, itm = model(video_feature, video_mask, input_ids, attention_mask, target, task)
    return loss, mlm, mfm, itm

def train(model, model_path,
          train_loader, train_loader1, optimizer, get_pred_and_loss, scheduler=None,
          num_epochs=5):
    os.makedirs('./save/pretrain', exist_ok=True)
    best_loss, best_epoch, step = None, 0, 0
    loss_list = []
    mlm_list, mfm_list, itm_list = [], [], []
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    for epoch in range(num_epochs):
        best_loss = None
        for item in tqdm(train_loader):
            model.train()
            # optimizer.zero_grad()
            with autocast():
                loss, mlm, mfm, itm = get_pred_and_loss(model, item)
                loss = loss.mean()
                mlm = mlm.mean()
                mfm = mfm.mean()
                itm = itm.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            if scheduler:
                scheduler.step()
            scaler.update()
            loss_list.append(loss.to('cpu').item())
            mlm_list.append(mlm.to('cpu').item())
            mfm_list.append(mfm.to('cpu').item())
            itm_list.append(itm.to('cpu').item())
            if step % 200 == 0 and step != 0:
                improve_str = ''
                loss = np.mean(loss_list)
                mlm = np.mean(mlm_list)
                mfm = np.mean(mfm_list)
                itm = np.mean(itm_list)
                loss_list = []
                mlm_list, mfm_list, itm_list = [], [], []
                if not best_loss or loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), f'./save/pretrain/pretrain_epoch{epoch + 1}.bin')
                    improve_str = f"|New best_loss={best_loss:6.4}"

                logging.info(f"Epoch={epoch + 1}/{num_epochs}|step={step:3}|mlm={mlm:6.4}|mfm={mfm:6.4}|itm={itm:6.4}|loss={loss:6.4}" + improve_str)
            step += 1
        for item in tqdm(train_loader1):
            model.train()
            # optimizer.zero_grad()
            with autocast():
                loss, mlm, mfm, itm = get_pred_and_loss(model, item)
                loss = loss.mean()
                mlm = mlm.mean()
                mfm = mfm.mean()
                itm = itm.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            if scheduler:
                scheduler.step()
            scaler.update()
            loss_list.append(loss.to('cpu').item())
            mlm_list.append(mlm.to('cpu').item())
            mfm_list.append(mfm.to('cpu').item())
            itm_list.append(itm.to('cpu').item())
            if step % 200 == 0 and step != 0:
                improve_str = ''
                loss = np.mean(loss_list)
                mlm = np.mean(mlm_list)
                mfm = np.mean(mfm_list)
                itm = np.mean(itm_list)
                loss_list = []
                mlm_list, mfm_list, itm_list = [], [], []
                if not best_loss or loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), f'./save/pretrain/pretrain_epoch{epoch + 1}.bin')
                    improve_str = f"|New best_loss={best_loss:6.4}"

                logging.info(f"Epoch={epoch + 1}/{num_epochs}|step={step:3}|mlm={mlm:6.4}|mfm={mfm:6.4}|itm={itm:6.4}|loss={loss:6.4}" + improve_str)
            step += 1
        
        

def train_and_validate(args):
    # 1. load data
    args.ispretrain = True
    args.val_ratio = 0.0
    args.batch_size = 128
    args.max_frames =32
    args.learning_rate=5e-05
    args.bert_learning_rate=5e-05
    args.cross_learning_rate=7e-05
    
    args.train_annotation = '/opt/ml/input/data/annotations/unlabeled.json'
    args.train_zip_frames = '/opt/ml/input/data/zip_feats/unlabeled_vit.zip'
    train_dataloader, val_dataloader = create_dataloaders(args)
    args.train_annotation = '/opt/ml/input/data/annotations/labeled.json'
    args.train_zip_frames = '/opt/ml/input/data/zip_feats/labeled_vit.zip'
    train_dataloader1, val_dataloader1 = create_dataloaders(args)
    # 2. build model and optimizers
    model = MultiModalForPretrain(model_path=args.bert_dir)
    epoch = 30
    args.max_steps = len(train_dataloader) * epoch
    args.warmup_steps = args.max_steps * 0.06
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    # 3. training
    train(model, args.bert_dir, train_dataloader, train_dataloader1, optimizer, get_pred_and_loss, scheduler, epoch)


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)
    train_and_validate(args)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
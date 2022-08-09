import logging
import os
import time
import torch
from tqdm import tqdm
from config import parse_args
from data_helper import create_dataloaders

from pretrain_ALBEF import ALBEF
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
import numpy as np
from torch import distributed as dist
import warnings
from imp import reload
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

def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def get_pred_and_loss(model, item, arg):
    """Get pred and loss for specific task"""
    video_feature = item['frame_input'].cuda(arg.local_rank, non_blocking=True)
    input_ids = item['title_input'].cuda(arg.local_rank, non_blocking=True)
    attention_mask = item['title_mask'].cuda(arg.local_rank, non_blocking=True)
    video_mask = item['frame_mask'].cuda(arg.local_rank, non_blocking=True)

    target = None
    if 'label' in item:
        target = item['label'].cuda(arg.local_rank, non_blocking=True)

    loss_mlm, loss_ita, loss_itm  = model(video_feature, video_mask, input_ids, attention_mask,alpha=arg.alpha)
    # loss_mlm = reduce_mean(loss_mlm, dist.get_world_size())
    # loss_ita = reduce_mean(loss_ita, dist.get_world_size())
    # loss_itm = reduce_mean(loss_itm, dist.get_world_size())
    
    loss = loss_mlm + loss_ita + loss_itm
    
    return loss, loss_mlm, loss_ita, loss_itm

def train(model,
          arg,
          train_loader, 
          train_loader1,
          optimizer, 
          get_pred_and_loss, 
          train_sampler, 
          train_sampler1,
          scheduler=None,
          num_epochs=5,
          alpha=0.4):
    os.makedirs('./save/pretrain', exist_ok=True)
    best_loss, best_epoch, step = None, 0, 0
    loss_list = []
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    for epoch in range(num_epochs):
        best_loss = None
        train_sampler.set_epoch(epoch)
        model.train()
        for i, item in enumerate(tqdm(train_loader)):
            if epoch>0:
                arg.alpha = 0.3
            else:
                arg.alpha = arg.alpha*min(1, i/(len(train_loader)+len(train_loader1)))
            # optimizer.zero_grad()
            with autocast():
                loss, loss_mlm, loss_ita, loss_itm = get_pred_and_loss(model, item, arg)
                # loss, loss_mlm, loss_ita, loss_itm = loss.mean(),vm_loss.mean(),itm_loss.mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                if scheduler:
                    scheduler.step()
                scaler.update()
            loss_mlm = reduce_mean(loss_mlm, dist.get_world_size())
            loss_ita = reduce_mean(loss_ita, dist.get_world_size())
            loss_itm = reduce_mean(loss_itm, dist.get_world_size())
            loss = loss_mlm + loss_ita + loss_itm
            loss_list.append(loss.to('cpu').item())
            if step % 200 == 0 and step != 0:
                improve_str = ''
                loss = np.mean(loss_list)
                loss_list = []
                if not best_loss or loss < best_loss:
                    best_loss = loss
                    if arg.local_rank==0:
                        torch.save(model.module.state_dict(), f'./save/pretrain/pretrain_epoch{epoch + 1}.bin')
                        improve_str = f"|New best_loss={best_loss:6.4}"
                if arg.local_rank==0:
                    logging.info(f"Epoch={epoch + 1}/{num_epochs}|step={step:3}|mlm_loss={loss_mlm:6.4}|ita_loss={loss_ita:6.4}|itm_loss={loss_itm:6.4}|loss={loss:6.4}" + improve_str)
            step += 1
        
        train_sampler1.set_epoch(epoch)
        model.train()
        for i, item in enumerate(tqdm(train_loader1)):
            # optimizer.zero_grad()
            if epoch>0:
                arg.alpha = 0.3
            else:
                len_unlabel = len(train_loader)
                arg.alpha = arg.alpha*min(1, (i+len_unlabel)/(len_unlabel+len(train_loader1)))
            with autocast():
                loss, loss_mlm, loss_ita, loss_itm = get_pred_and_loss(model, item, arg)
                # loss, vm_loss, itm_loss = loss.mean(),vm_loss.mean(),itm_loss.mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                if scheduler:
                    scheduler.step()
                scaler.update()
            loss_mlm = reduce_mean(loss_mlm, dist.get_world_size())
            loss_ita = reduce_mean(loss_ita, dist.get_world_size())
            loss_itm = reduce_mean(loss_itm, dist.get_world_size())
            loss = loss_mlm + loss_ita + loss_itm
            loss_list.append(loss.to('cpu').item())
            if arg.local_rank==0:
                if step % 200 == 0 and step != 0:
                    improve_str = ''
                    loss = np.mean(loss_list)
                    loss_list = []
                    if not best_loss or loss < best_loss:
                        best_loss = loss
                        if arg.local_rank==0:
                            torch.save(model.module.state_dict(), f'./save/pretrain/pretrain_epoch{epoch + 1}.bin')
                            improve_str = f"|New best_loss={best_loss:6.4}"
                    if arg.local_rank==0:
                        logging.info(f"Epoch={epoch + 1}/{num_epochs}|step={step:3}|mlm_loss={loss_mlm:6.4}|ita_loss={loss_ita:6.4}|itm_loss={loss_itm:6.4}|loss={loss:6.4}" + improve_str)
            step += 1
        

def train_and_validate(args):
    # print(torch.cuda.device_count())
    # print(torch.cuda.device_count())
    # print(torch.cuda.device_count())
    # print(torch.cuda.device_count())
    
    args.val_ratio = 0.0
    args.batch_size = 32
    args.max_frames =32
    args.learning_rate=6e-05
    args.bert_learning_rate=6e-05
    args.cross_learning_rate=7e-05
    
    # 1. build model and optimizers
    # model = MultiModalForPretrain(model_path=args.bert_dir)
    model = ALBEF(config=args)
    
    model = model.cuda(args.local_rank)  # 将模型拷贝到每个gpu上.直接.cuda()也行，因为多进程时每个进程的device号是不一样的
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # 设置多个gpu的BN同步
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True, broadcast_buffers=False)
    
    # 2. load data
    args.train_annotation = '/opt/ml/input/data/annotations/unlabeled.json'
    args.train_zip_frames = '/opt/ml/input/data/zip_feats/unlabeled_vit.zip'
    train_dataloader, val_dataloader, train_sampler= create_dataloaders(args)
    args.train_annotation = '/home/tione/notebook/data/annotations/labeled.json'
    args.train_zip_frames = '/home/tione/notebook/data/zip_feats/labeled_vit.zip'
    train_dataloader1, val_dataloader1, train_sampler1= create_dataloaders(args)
    
    epoch = 15
    args.max_steps = (len(train_dataloader)+len(train_dataloader1)) * epoch
    args.warmup_steps = args.max_steps * 0.08
    optimizer, scheduler = build_optimizer(args, model)
    # if args.device == 'cuda':
    #     model = torch.nn.parallel.DataParallel(model.to(args.device))
    # 3. training
    train(model, args, train_dataloader, train_dataloader1, optimizer, get_pred_and_loss, train_sampler, train_sampler1, scheduler, epoch)


def main():
    args = parse_args()
    args.ispretrain = True
    setup_logging()
    setup_device(args)
    setup_seed(args)
    

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)
    train_and_validate(args)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
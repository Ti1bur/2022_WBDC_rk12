import os
import time
import json
import torch
import random
import zipfile
import logging
import numpy as np
from VIT import vit
from torch import nn
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from config import parse_args
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import BertTokenizer
from data_helper import MultiModalDataset
from model_single import MultiModal_single
from model_double import MultiModal_double
from torch.utils.data import SequentialSampler, DataLoader
from category_id_map import lv2id_to_category_id, category_id_to_lv2id
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate

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

    return loss, results

class InferenceModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.video_backbone = vit()
        self.model_double = MultiModal_double(args, False)
        # self.model_double_2 = MultiModal_double(args, False)
        # self.model_double_3 = MultiModal_double(args, False)
        self.model_single = MultiModal_single(args, False)

    def forward(self, title_input, title_mask, frame_input, frame_mask):
        # frame_input = self.video_fc(self.video_backbone(frame_input))
        frame_input = self.video_backbone(frame_input)
        pred1 = self.model_double(title_input, title_mask, frame_input, frame_mask)
        # pred2 = self.model_double_2(title_input, title_mask, frame_input, frame_mask)
        # pred3 = self.model_double_3(title_input, title_mask, frame_input, frame_mask)
        pred4 = self.model_single(title_input, title_mask, frame_input, frame_mask)
        pred = pred1 * 0.5  + pred4 * 0.5
        # pred = torch.argmax(pred, dim = 1)
        return pred

def inference():
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_frames, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)
    

    # 2. load model
    model = InferenceModal(args)
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    checkpoint1 = torch.load('../double_stream/save/vit-14-ema-pgd3-pretrain6-origin-dataset/double_model.bin', map_location='cpu')
    # checkpoint2 = torch.load('save/double_model2.bin', map_location='cpu')
    # checkpoint3 = torch.load('save/double_model3.bin', map_location='cpu')
    checkpoint4 = torch.load('../single_stream/save/vit-14-pretrain30-ema-pgd3-origin-dataset/single_model.bin', map_location='cpu')
    model.model_double.load_state_dict(checkpoint1, strict=False)
    # model.model_double_2.load_state_dict(checkpoint2, strict=False)
    # model.model_double_3.load_state_dict(checkpoint3, strict=False)
    model.model_single.load_state_dict(checkpoint4, strict=False)
    # model.load_state_dict(checkpoint, strict=False)
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()

    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            with autocast():
                pred_label_id = model(batch['title_input'], batch['title_mask'], batch['frame_input'], batch['frame_mask'])
                pred_label_id = torch.argmax(pred_label_id, dim=1)
                predictions.extend(pred_label_id.cpu().numpy())

    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    inference()
# def inference():
#     args = parse_args()
#     # 1. load data
#     dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_frames, test_mode=True)
#     sampler = SequentialSampler(dataset)
#     dataloader = DataLoader(dataset,
#                             batch_size=args.test_batch_size,
#                             sampler=sampler,
#                             drop_last=False,
#                             pin_memory=True,
#                             num_workers=args.num_workers,
#                             prefetch_factor=args.prefetch)

#     # 2. load model
#     model = InferenceModal(args)
#     scaler = torch.cuda.amp.GradScaler()
#     autocast = torch.cuda.amp.autocast
#     checkpoint1 = torch.load('save/fold5/model1.bin', map_location='cpu')
#     checkpoint2 = torch.load('save/fold5/model2.bin', map_location='cpu')
#     checkpoint3 = torch.load('save/fold5/model3.bin', map_location='cpu')
#     checkpoint4 = torch.load('save/fold5/model4.bin', map_location='cpu')
#     model.model1.load_state_dict(checkpoint1, strict=False)
#     model.model2.load_state_dict(checkpoint2, strict=False)
#     model.model3.load_state_dict(checkpoint3['model_state_dict'], strict=False)
#     model.model4.load_state_dict(checkpoint4['model_state_dict'], strict=False)
#     # model.load_state_dict(checkpoint, strict=False)
#     if torch.cuda.is_available():
#         model = torch.nn.parallel.DataParallel(model.cuda())
#     model.eval()

#     # 3. inference
#     predictions = []
#     with torch.no_grad():
#         for batch in tqdm.tqdm(dataloader):
#             with autocast():
#                 pred_label_id = model(batch['title_input'], batch['title_mask'], batch['frame_input'], batch['frame_mask'])
#                 pred_label_id = torch.argmax(pred_label_id, dim=1)
#                 predictions.extend(pred_label_id.cpu().numpy())

#     # 4. dump results
#     with open(args.test_output_csv, 'w') as f:
#         for pred_label_id, ann in zip(predictions, dataset.anns):
#             video_id = ann['id']
#             category_id = lv2id_to_category_id(pred_label_id)
#             f.write(f'{video_id},{category_id}\n')


# if __name__ == '__main__':
#     inference()

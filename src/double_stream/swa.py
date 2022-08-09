import os
import pdb
import torch
import copy
from model import MultiModal
from config import parse_args

def get_model_path_list(base_dir):
    """    从文件夹中获取 model.pt 的路径    """
    model_lists = []
    for root, dirs, files in os.walk(base_dir):
        for _file in files:
            if 'epoch_0' not in _file:
                model_lists.append(os.path.join(root, _file))
            # model_lists.append(os.path.join(root, _file))
    # sorted(model_lists, key = lambda x: x.split('_')[-1])
    return model_lists
def swa(model, model_dir, swa_start=5):
    """    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA    """
    model_path_list = get_model_path_list(model_dir)
    model_path_list = sorted(model_path_list, key=(lambda x: x.split('_')[-1].split('.')[1]),reverse=True)
    # print(model_path_list)
    # assert 1 <= swa_start < len(model_path_list) - 1    # Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 0
    swa_model = copy.deepcopy(model)
    swa_n = 0.
    with torch.no_grad():
        for _ckpt in model_path_list[:swa_start]:  # 从第 swa_start 个模型开始进行swa融合
            print(_ckpt.split('/')[-1])
            model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu'))['model_state_dict'])
            tmp_para_dict = dict(model.named_parameters())
            alpha = 1. / (swa_n + 1.)
            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))
            swa_n += 1    # use 100000 to represent swa to avoid clash
        swa_model_dir = os.path.join(model_dir, f'./')
        if not os.path.exists(swa_model_dir):
            os.mkdir(swa_model_dir)
        swa_model_path = os.path.join(swa_model_dir, 'double_model.bin')
        torch.save(swa_model.state_dict(), swa_model_path)
    return swa_model
if __name__ == '__main__':
    dir = "./save/vit-14-ema-pgd3-pretrain6-origin-dataset/"   # swa_dir = "E:/code\微信大数据挑战赛/weixin/save/SWA/
    args = parse_args()
    model = MultiModal(args)
    swa(model, dir)
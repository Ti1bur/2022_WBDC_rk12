import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

args = {}
args['train_annotation'] = '/home/tione/notebook/data/annotations/labeled.json'
with open(args['train_annotation'], 'r', encoding='utf8') as f:
    anns = json.load(f)

skf = StratifiedKFold(10, shuffle=True, random_state=2022)
y = [anns[i]['category_id'] for i in range(len(anns))]
x = anns
x = np.array(x)
y = np.array(y)
fold_num = 1
for index in skf.split(x, y):  # 因为是3等分，即3折交叉，一共循环3次
    x_train = x[index[0]]
    y_train = y[index[0]]
    x_valid = x[index[1]]
    y_valid = y[index[1]]

    train_fold_name = '/home/tione/notebook/data/annotations/fold10/' + str(fold_num) + "train" + ".json"
    valid_fold_name = '/home/tione/notebook/data/annotations/fold10/' + str(fold_num) + "valid" + ".json"
    with open(train_fold_name, "w", encoding="utf-8") as f:
        json.dump([tr for tr in x_train], fp=f)
    with open(valid_fold_name, "w", encoding="utf-8") as f:
        json.dump([tr for tr in x_valid], fp=f)
    fold_num += 1
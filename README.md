## 比赛
微信大数据挑战赛(小样本多模态分类任务)
rank ：12  (Student Rank : 8, Score : 0.727216)
### 赛题描述和页面
[2022微信大数据挑战赛赛题介绍和描述](https://algo.weixin.qq.com/problem-description)

## 思路

利用开源Bert和开源Visual模型权重来初始化文本和图片, 再送入自己的model中对文本进行交互, 主要构建了两个模型.

(1) 单流(Single_Stream)

    文本过Bert(roberta-wwm-ext)的embedding, 图片过Vit(Clip-Vit-B/16)做成embedding, 拼接在一起后送入Bert的12层Transformer块.
    裸模型未预训练成绩: 0.67 - 0.68, 预训练任务: ITM, MFM, MLM, 预训练后(Trick)成绩: 0.720
    
(2) 双流(Double_stream)

    文本过Bert(roberta-wwm-ext)的12层Transformer块, 图片过Vit(Clip-Vit-B/16)做成(Batch_size, Max_frame, Vision_dim), 送入3层Transformer-Attention块中(图片做Q, 文本做K, V)
    裸模型未预训练成绩: 0.700，      预训练任务: ITM, ITA, MLM, 预训练后(Trick)成绩: 0.715

模型融合后达到了最高分

Trick: EMA, SWA, FGM, PGD

## 经验与总结

(1)经验: 

多模态任务最好采用经过了大规模多模态数据预训练的模型权重(特指Clip), (2022年7月Clip开源中文语料库预训练权重). 

多做多模态信息的交互任务, 比如单流中预训练的ITM, 双流中的Cross_attention

(2)赛后学习(大佬发言)

   (1) 在限制Inference时间的比赛中都要尝试利用大模型(Bert-large, Clip-large)来对预训练数据打伪标, 一是为了能更好地发挥经过预训练的大模型的性能, 二是更加充分的利用预训练数据.
            
       伪标流程: 1. 利用Large模型在训练集上训练好高分model, 并对无标注数据进行伪标标注
                     
                2. 两种思路: 在进行预训练的时候多一个对标签的任务, 平衡loss, 相当于多任务学习.
                                 
                            在训练时先对伪标数据进行训练, 再用训练集进行修正.
        
   (2) 在小样本的情况下, 多分类任务中出现的小类, 会由于F1_marco而导致对小类错分极为敏感, 故可观察模型在验证集中, 对小类分类的正确率, 强行提高小类的预测概率.
            
(3) 其他大佬的总结
![image](https://github.com/Ti1bur/2022WeChatBigDataChallenge/blob/main/Summarize.jpg)

## 代码说明

------

### 环境配置

Python版本：3.6.9    PyTorch版本：1.7.0    CUDA版本：11.0

所需环境在`requirements.txt`中定义

### 文件目录结构

```
./
├── README.md
├── requirements.txt      # Python包依赖文件 
├── init.sh               # 初始化脚本，用于准备环境
├── train.sh              # 模型训练脚本
├── inference.sh          # 模型测试脚本 
├── opensource_models/                           # 开源模型
│   ├── swin_base_patch4_window7_224_22k.pth     # Swin Transformer Base开源模型
│   ├── ViT-B-16.pt                              # ViT开源模型
│   ├── roberta-wwm-ext/                         # RoBERTa开源模型
│   │   ├── config.json                          # 配置文件
│   │   ├── pytorch_model.bin                    # 模型文件
│   │   ├── vocab.txt                            # 词库
├── src/                          # 核心代码
│   ├── single_stream/
│   │   ├── category_id_map.py    # ID-类别映射
│   │   ├── config.py             # 参数配置
│   │   ├── data_helper.py        # 文件读取
│   │   ├── extract_feature.py    # 提取视频特征
│   │   ├── evaluate.py           # 结果评估
│   │   ├── main.py               # 主函数
│   │   ├── masklm.py             # 预训练中的MLM、MFM、ITM函数
│   │   ├── model.py              # 单流模型
│   │   ├── pretrain.py           # 预训练主函数
│   │   ├── pretrain_model.py     # 预训练模型框架
│   │   ├── swa.py                # SWA函数
│   │   ├── swin.py               # Swin模型框架
│   │   ├── util.py               # 相关函数
│   │   ├── VIT.py                # ViT模型框架
│   │   ├── save/
│   │   │   ├── vit-14-pretrain30-ema-pgd3-origin-dataset/        # 存放微调后的模型
│   │   │   ├── pretrain/                       # 存放预训练模型   
│   │   ├── log/                  # 日志文件
│   ├── double_stream/
│   │   ├── category_id_map.py    # ID-类别映射
│   │   ├── config.py             # 参数配置
│   │   ├── data_helper.py        # 文件读取
│   │   ├── evaluate.py           # 结果评估
│   │   ├── main.py               # 主函数
│   │   ├── model.py              # 双流模型
│   │   ├── pretrain_ALBEF.py     # 预训练模型框架
│   │   ├── pretrain.py           # 预训练主函数
│   │   ├── swa.py                # SWA函数
│   │   ├── swin.py               # Swin模型框架
│   │   ├── tokenization_bert.py  # 分词器
│   │   ├── util.py               # 相关函数
│   │   ├── VIT.py                # ViT模型框架
│   │   ├── save/
│   │   │   ├── vit-14-ema-pgd3-pretrain6-origin-dataset/        # 存放微调后的模型
│   │   │   ├── pretrain/                       # 存放预训练模型   
│   │   ├── log/                  # 日志文件
│   ├── single_and_double/
│   │   ├── category_id_map.py    # ID-类别映射
│   │   ├── config.py             # 参数配置
│   │   ├── data_helper.py        # 文件读取
│   │   ├── evaluate.py           # 结果评估
│   │   ├── inference.py          # 推理函数
│   │   ├── model_double.py       # 双流模型
│   │   ├── model_single.py       # 单流模型
│   │   ├── util.py               # 相关函数
│   │   ├── VIT.py                # ViT模型框架
```

### 数据

- 大赛提供的无标注数据（100万）加有标注数据（10万）用于预训练
- 大赛提供的有标注数据（10万）用于微调
- 未使用任何额外数据

### 预训练模型

- 使用了 huggingface 上提供的 `hfl/chinese-roberta-wwm-ext` 模型。链接为： https://huggingface.co/hfl/chinese-roberta-wwm-ext，包含三个文件：config.json、pytorch_model.bin、vocab.txt。config.json的md5值为4d1944648b1d2098dd3ffddc95a86015，pytorch_model.bin的md5值为bd7f0d8b1c326bb2dd4eb185344df789，vocab.txt的md5值为3b5b76c4aef48ecf8cb3abaafe960f09
- 使用了 SwinTransformer 官方提供的`swin-base`模型。链接为：https://github.com/microsoft/Swin-Transformer，md5值为：bf9cc182ae5e417f97390e2b21a0eb09
- 使用了OpenAI官方提供的`CLIP-ViT`模型。链接为：https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt，md5值为：44c3d804ecac03d9545ac1a3adbca3a6

### 算法描述

1、双流模型：

- 对于视觉特征，首先使用开源模型`CLIP-ViT`提取视频帧特征，将其作为视频embedding，之后需要过一个线性层将维度转变为768
- 对于文本特征，代入文本数据到预训练模型`chinese-roberta-wwm-ext`中，取输出last_hidden_state作为文本embedding。文本由title、asr、ocr三个字段中直接拼接得到，长度限制为258
- 以文本embedding为K、V，视频embedding为Q，代入基于Transformer的decoder层实现的cross-attention网络结构
- 对最后的结果进行mean-pool，并使用简单的 MLP 结构进行类别预测
- 优化操作包括：`EMA`、`FGM`、`PGD`、`SWA`，进行预训练，预训练任务包括`MLM`、`ITC`、`ITM`

2、单流模型：

- 对于视觉特征，首先使用开源模型`CLIP-ViT`提取视频帧特征，将其作为视频embedding，之后需要过一个线性层将维度转变为768，并过`BertEmbeddings`得到视频embedding
- 对于文本特征，直接过`BertEmbeddings`（视频、文本共用）得到文本embedding。文本由title、asr、ocr三个字段直接拼接得到，长度限制为258
- 将视频embedding与文本embedding代入`BertEncoder`，取输出last_hidden_state过简单的 MLP 结构得到分类结果。
- 优化操作包括：`EMA`、`FGM`、`PGD`、`SWA`，进行了预训练，预训练任务包括`MLM`、`MFM`、`ITM`

### 性能

B榜测试性能：

单流+双流（加权融合）：0.727216

### 训练流程

- 使用大赛提供的无标注数据（100万）加有标注数据（10万）共110万数据进行预训练
- 单流预训练任务包括`MLM`、`MFM`、`ITM`，对文本进行`MLM`，对视频进行`MFM`，对文本-视频pair进行`ITM`，训练30个epoch，保存每个epoch训练的模型
- 双流预训练任务包括`MLM`、`ITC`、`ITM`，对文本进行`MLM`，对文本-视频pair进行`ITC`、`ITM`，训练15个epoch，保存每个epoch训练的模型

### 微调与推理流程

- 对有标注数据划分10%的数据作为验证集，剩下90%的数据作为训练集，读取预训练后的模型参数进行微调

- 取在验证集上评估得分较高的多个模型进行SWA融合，分别得到单、双流的最终模型

- 直接对单流与双流模型预测得出的各类别的概率进行加权融合，处理融合结果为指定类别，保存预测结果


### 代码贡献

- single_stream文件夹中，自身贡献部分包括model文件的单流模型架构代码和`EMA`、`FGM`、`PGD`等trick代码、pretrain_model和pretrain和masklm文件的预训练代码、main文件的微调流程代码、swa文件内的代码、util文件中分层学习率的代码

- double_stream文件夹中，自身贡献部分包括model文件的双流流模型架构代码和`EMA`、`FGM`、`PGD`等trick代码、pretrain_ALBEF和pretrain文件的预训练代码、main文件的微调流程代码、swa文件内的代码、util文件中分层学习率的代码

- single_and_double文件夹中，自身贡献部分包括model_double和model_single文件中的模型架构代码、inference文件的模型融合代码

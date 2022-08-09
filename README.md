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
1. 关于比赛后处理方案？
    - 陪跑：我们的后处理的思路是把少类别的概率强行放大，这样初赛复赛都有比较稳定的提升（3k）左右.具体做法是 1/value_count,每个类别得到一个放大的概率值，大类是1，小类>1, 具体的值设置是通过初赛的线下五折调出来的参数。

2. mixup 是否有效？
    - 辉：初赛尝试了mixup,复赛并没有用上，模型抖的厉害。
    - 陪跑：我这里尝试了很多次mixup，但是没啥用。

3. 伪标签是否有效？
    - 陪跑：有效果，但取决于生成的标签的精度。在损失函数上的选择，可以考虑使用kd loss,也可以使用交叉熵。
    - 郭大：这题的关键其实还是伪标签，提升是2个百分点，0.720--0.741.伪标签不是为了利用数据，而是为了利用大模型（估计是为了使用clip large）。在训练伪标签上，是否使用soft label或者使用hard label，区别并不大。如果soft 和hard 分别都用，融合还能提升分数。本赛题受限于规则上进行限时，所以需要伪标签，不限时的话，都不需要伪标签（猜测是直接用多个large融合）。
    - chizhu：伪标签在我们这里提升是2个百。0.689--0.710，限于挥霍的stacking细节，可能也影响了模型效果。具体使用了8帧的单折。我们采用了两阶段训练小模型，先用100万无标签数据训练第一阶段，然后第二阶段用9万真实标签修正，（可不可以描述的更细致一点呢？）。
    - 挥霍的人生：受限于我们组的打伪标签的base模型精度太低问题，导致我们后期运用伪标签提升未达到预期（主要是上班太忙，没时间做题才是真相）。
    - UA:伪标签在本题中可以理解为蒸馏的思路，伪标签可以套娃，训练出好的模型，再去标注，再训练，再标注，形成一个循环过程。评估的标注是测试集上无法再次上分为止。当出现大量的unlabel数据的时候，就可以考虑使用伪标签，例如本赛题，特征均出现对齐的情况，唯一的区别是缺失label。如果使用kl loss做软标签，就可以看做是知识蒸馏。
    - Lawliet：我们一开始用的swin-base上分，后来看到大家都在传clip，就产生了敏锐的洞察力。
    - 虚着点和气：关于chizhu描述先用100万无标签数据训练第一阶段，然后第二阶段用9万真实标签修正这部分，完全可以边打标，边修正。


4. 本赛题限时的原因？
    - starry:以我的理解，主办方限时更多的是考虑限制大量的模型融合，而更希望更少量的模型的精度提升，能更好的做工业化应用参考，由于这个场景下内容理解可以离线做，并不需要做到高度实时，所以和视频推荐场景中的召回和精排仍然是有区别的。

5. 特征提取的backbone用谁好？
    - 各队伍都使用各种类型的视频特征预训练模型作为提取的主要方法，例如swin, tiny, clip, vit等，但从本赛题来看，最有效的方法仍然是clip。具体可以参考的代码地址如下：
        - [openai/CLIP](https://github.com/openai/CLIP)
        - 具体采用的预训练模型文件之一参考如下：
        ```shell
        {"ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"}

    - Lawliet, 陪跑：几乎所有的论文都是用clip的，这一点可以作为选择backbone有更好效果的一种指导方向。
    - 高等数学：初赛我们就在使用clip，由于初赛已经明确了视频特征提取结果，因此是在文本上用clip提取的文本信息。（这一点的使用方法没有理解到）。
    - lawliet: 2022年7月29日clip开始上线了中文clip模型。

6. 伪标签使用后是否会导致训练结果leak？
    - UA:伪标签确实会存在leak,这时候验证集可能存在无法作为线上预期分数的参考的问题。
    - Tibur:为了评估自己的模型效果，当出现leak的时候，可以通过提交线上来查看是否自己的整体流程是有效的。


7. clip的学习率要设置成多少？
    - 陪跑：bert的1/10作为参考，例如bert的学习率为1e-4,那么clip的学习率就是1e-5. 在这里需要配置分层学习率。

8. clip是否要冻结不参与微调训练？
    - Tibur: 需要冻结，这里应该是指只需要做推理得到特征。
    - 两面包夹芝士: 可以一起参与训练，提升是0.7个百分点。在学习率设置上是5e-6.具体的学习率设置如下参考：
        ```shell
        -- other_learning_rate 5e-4
        -- bert_learning_rate 5e-5
        -- clip_learning_rate 5e-6
        ```

9. 哪些预训练任务是有效的？
    - Tibur: ita, itm 在我这边是最有效的。这是通过初赛消融得到的结论，复赛没时间。我的双流就是图片过backbone, 文本过bert,然后两个做cross_attentiony以后直接输出.
    - 陪跑：我这边clip模型最重要，双流里，算text vision的相似度可以明显上分。我们的双流就是UniVL(3个transformer 编码器) + LOUPE.在双流里用UniVL,具体操作上使用clip替代了itm任务.在双流里，是文本过tfs,图片embedding过tfs,合并起来继续tfs.相当于是三个独立的tfs.这样的模型比单流要慢一点。我使用ALBEF的模型，如果用中间分层的话会出现过拟合，而且会预训练的时候没法在微调上产生效果，暂时未排查到原因。在设计MLM任务的时候，可以考虑使用ngram mask，也比普通的MLM任务有明显提升。
    - 一只大海龟：ALBEF在这个场景下预训练不上分。


10. 融合阶段受用logits融合好还是使用prob? prob在此处应该是指对logits做softmax后的结果。
    - 陪跑：使用prob融合出现掉分。考虑单流和双流模型如果差异大的话，Logits不一定在一个向量空间内。融合方法上就是直接加权。
    - 两面包夹芝:初赛上融合logits不如prob，复赛默认使用prob。
    - Tibur:我们对两种方法都做了尝试，但是没有区别。我的理解是应该融prob。
    - UA: 如果模型中存在很多的Norm,不同模型的预测结果的值域应该不会差太远。


11. 数据上特别是asr ocr存在很多脏数据，是否可以做特征工程清洗呢？（玄学，清洗更可能的是掉分）
    - Lawliet:初赛做了清洗，掉分。
    - Tibur:清洗文本有用，我洗了涨了3k。但是预训练后反而没用了。


12. 还有哪些数据EDA存在很高的价值呢？
    - 陪跑：top1的预测准确率acc是0.8左右，但是top5的hit就有0.95，这表明一般ground truth就在top5内。所以对小类乘以权重，变大一点，就可以让top2的到top1了。初赛上涨了3k,复赛更多上分5-6k。具体代码如下：
    ```python
    from collections import Counter
    a = pd.DataFrame(Counter(train_data["category_id"].map(CATEGORY_ID_TO_LV2ID)), index=[0]).T.sort_index()
    a[0] = (10 / a[0] + 1)
    ```

13. swa 应该在全量数据训练中如何参与做提升？
    - 陪跑：我们swa了top5个。开着ema训练，最后再手动swa.（代码也非常简单，就是把字典里的参数的weight和bias提取出来加权）。


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

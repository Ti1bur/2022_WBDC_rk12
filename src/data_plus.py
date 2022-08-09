import random
import os
import jieba
import numpy as np
import json
from gensim.models import KeyedVectors, TfidfModel
from gensim.corpora import Dictionary
from util import read_samples, write_samples, isChinese
from gensim import matutils
from itertools import islice
from gensim import models
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from tqdm import tqdm


class DataPlus():
    def __init__(self, sample_path=None, wv_path=None):
        with open('/home/tione/notebook/data/annotations/labeled.json', 'r', encoding='utf-8') as f:
            self.labeled = json.load(f)
        # with open('/home/tione/notebook/data/annotations/unlabeled.json', 'r', encoding='utf-8') as f:
        #     self.unlabeled = json.load(f)
        self.labeled = self.labeled[:30000]
        self.labeled_samples = []
        for a in self.labeled:
            text = a['title'] + a['asr']
            for b in a['ocr']:
                text += b['text']
            self.labeled_samples.append(text)

        # self.unlabeled_samples = []
        # for a in self.unlabeled:
        #     text = a['title'] + a['asr']
        #     for b in a['ocr']:
        #         text += b['text']
        #     self.unlabeled_samples.append(text)

        self.labeled_samples = [list(jieba.cut(sample)) for sample in self.labeled_samples]
        # self.unlabeled_samples = [list(jieba.cut(sample)) for sample in self.unlabeled_samples]

        self.wv = KeyedVectors.load_word2vec_format('word2vec.bin', binary=False)

        if os.path.exists('./tfidf_word2vec/tfidf.model'):
            self.tfidf_model = TfidfModel.load('tfidf_word2vec/tfidf.model')
            self.dct = Dictionary.load('tfidf_word2vec/tfidf.dict')
            # doc2bow将文档转换为单词袋(BoW)格式= (token_id, token_count)元组的列表
            self.corpus = [self.dct.doc2bow(doc) for doc in self.labeled_samples]
        else:
            self.dct = Dictionary(self.labeled_samples)
            self.corpus = [self.dct.doc2bow(doc) for doc in self.labeled_samples]
            self.tfidf_model = TfidfModel(self.corpus)
            self.dct.save('./tfidf_word2vec/tfidf.dict')
            self.tfidf_model.save('./tfidf_word2vec/tfidf.model')
            self.vocab_size = len(self.dct.token2id)

    def vectorize(self, docs, vocab_size):

        return matutils.corpus2dense(docs, vocab_size)

    def extract_keywords(self, dct, tfidf, threshold=0.2, topk=5):
        """ 提取关键词
        :param dct (Dictionary): gensim.corpora.Dictionary
        :param tfidf (list):
        :param threshold: tfidf的临界值
        :param topk: 前 topk 个关键词
        :return: 返回的关键词列表
        """
        tfidf = sorted(tfidf, key=lambda x: x[1], reverse=True)

        return list(islice([dct[w] for w, score in tfidf if score > threshold], topk))

    def replace(self, sample, doc):
        """用wordvector的近义词来替换，并避开关键词
        :param sample (list): reference token list
        :param doc (list): A reference represented by a word bag model
        :return: 新的文本
        """
        keywords = self.extract_keywords(self.dct, self.tfidf_model[doc])
        #
        num = int(len(sample) * 0.3)
        new_tokens = sample.copy()
        indexes = np.random.choice(len(sample), num)
        for index in indexes:
            token = sample[index]
            if isChinese(token) and token not in keywords and token in self.wv:
                new_tokens[index] = self.wv.most_similar(positive=token, negative=None, topn=1)[0][0]

        return ''.join(new_tokens)

    def generate_samples(self):
        """得到用word2vector词表增强后的数据
        :param write_path:
        """
        for id in tqdm(range(len(self.labeled_samples))):
            # 替换同义词
            text = self.replace(self.labeled_samples[id], self.corpus[id])
            self.labeled[id]['text_plus'] = text

    def save(self, wr_path):
        with open(wr_path, 'w', encoding='utf-8') as f:
            json.dump(self.labeled, f)


if __name__ == '__main__':
    a = DataPlus()
    a.generate_samples()
    a.save('./data/labeled_plus.json')
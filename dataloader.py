import re
import os
import time
import emoji
import random
import numpy as np
import pandas as pd
from soynlp.normalizer import repeat_normalize

import torch
from torch.utils.data import DataLoader, TensorDataset
from score import BERTScore
from transformers import BertTokenizer


class make_dataloader:
    def __init__(self, options):
        super().__init__()
        self.args = options
        self.tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-base')

    def read_data(self, path):
        if path.endswith('xlsx'):
            return pd.read_excel(path)
        elif path.endswith('csv'):
            return pd.read_csv(path)
        elif path.endswith('tsv') or path.endswith('txt'):
            return pd.read_csv(path, sep='\t')
        else:
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')

    def preprocess_dataframe(self, df, mix=None, DEVICE=None):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

        def clean(x):
            x = pattern.sub(' ', x)
            x = url_pattern.sub('', x)
            x = x.strip()
            x = repeat_normalize(x, num_repeats=2)
            return x

        df['label'] = df['label'].replace(['none', 'offensive', 'hate'],[0,1,2])
        df['comments'] = df['comments'].map(lambda x: clean(str(x)))

        if mix==None:
            df['comments'] = df['comments'].map(lambda x: self.tokenizer.encode(
                x,
                pad_to_max_length=True,
                max_length=self.args.max_length,
                truncation=True,
            ))

        if mix:
            label = df['label'].to_list()
            comments = df['comments'].to_list()
            candidates = df['comments'].to_list()
            mix_comments = []
            mix_label = []

            model_name = "beomi/kcbert-base"
            bertscore = BERTScore(model_name, best_layer=4, device=DEVICE)

            for i in np.arange(len(comments)//2):
                c = random.sample(candidates, 1)[0]
                l = label[comments.index(c)]
                candidates.remove(c)
                
                references = [c]*len(candidates)
                score = bertscore(references, candidates, batch_size=1024)
                if mix=='low':
                    mix_c = candidates[score.index(np.min(score))]
                if mix=='high':
                    mix_c = candidates[score.index(np.max(score))]
                candidates.remove(mix_c)
                mix_l = label[comments.index(mix_c)]

                mix_comments.append(list(map(lambda x: self.tokenizer.encode(
                    x,
                    pad_to_max_length=True,
                    max_length=self.args.max_length,
                    truncation=True,
                ), [c, mix_c])))
                mix_label.append([l, mix_l])
            
            df = pd.DataFrame({
                    "comments": [sum(i, []) for i in mix_comments],
                    "label": mix_label
                })
        
        return df
    

    def dataloader(self, path, mix=None, DEVICE=None, no_label=False):
        df = self.read_data(path)
        df = self.preprocess_dataframe(df, DEVICE=DEVICE, mix=mix)

        if no_label:
            dataset = TensorDataset(
                torch.tensor(df['comments'].to_list(), dtype=torch.long),
                )

        else:
            dataset = TensorDataset(
                torch.tensor(df['comments'].to_list(), dtype=torch.long),
                torch.tensor(df['label'].to_list(), dtype=torch.long),
                )
            
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
        )

# Standard
import os
import sys
from datetime import date
import re

# PIP
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import BertTokenizer


# Custom



class CustomDataset(Dataset):
    def __init__(
        self,
        cfg,
        start_date='2014-01-01',
        end_date='2016-03-31',
        time_delta=1,
        is_filtering = False,
    ):
        self.cfg = cfg
        self.start_date = start_date
        self.end_date = end_date
        self.time_delta = time_delta
        self.is_filtering = is_filtering

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        special_tokens_dict = {'additional_special_tokens': ['at_user','$ticker','$target_ticker']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        xs, ys = self.gather_data()

        print(self.tokenizer.decode(xs[0]))
        print(self.tokenizer.decode(xs[2]))
        print(self.tokenizer.decode(xs[1]))

        self.X = torch.tensor(xs, dtype=torch.long)
        self.y = torch.tensor(ys, dtype=torch.long)

        count = 0

    def gather_data(self):
        xs = []
        ys = []

        if self.is_filtering:
            file_list = os.listdir(self.cfg.TWITTER_DATASET_DIR_FOR_FILTERING)
            file_list = sorted(file_list)
            file_list = [f[:-4] for f in file_list if f.endswith('.csv')]

        else:
            file_list = os.listdir(self.cfg.TWITTER_DATASET_DIR)
            file_list = sorted(file_list)

            if 'preprocessed' in self.cfg.TWITTER_DATASET_DIR:
                file_list = [f[:-4] for f in file_list if f.endswith('.ftr')]
            else:
                file_list = [f[:-4] for f in file_list if f.endswith('.csv')]
        
        ticker_list = ['$'+f.lower() for f in file_list]

        for file_name in file_list:
            df = self.load_data_from_file(file_name)
            xs.extend(df['text'].values)
            ys.extend(df['label'].values)

        print(f'TOTAL: {len(ys)}')
        print(f'UP: {sum(ys)}')
        print(f'DOWN: {len(ys) - sum(ys)}')

        return xs, ys

    @staticmethod
    def is_in_term(d, start_date, end_date):
        if start_date <= d and d <= end_date:
            return True

        return False

    @staticmethod
    def preprocess_text(text,symbol):
        new_str = text.lower()
        new_str = new_str.replace(f'${symbol}','target_ticker')
        new_str = re.sub('[$][a-z]+','$ticker',new_str)
        new_str = new_str.replace('target_ticker','$target_ticker')
        return new_str

    def load_data_from_file(self, symbol):
        '''
        df info
        columns: target_date, date, text, label, time_delta
        target_date (date): target date
        date (date): tweet's publish date
        text (string): tweet
        label (int): 가격 변동 up=1, down=0 
        time_delta (int): target_date와 날짜 차이
        '''

        if self.is_filtering:
            df = pd.read_csv(f'{self.cfg.TWITTER_DATASET_DIR_FOR_FILTERING}/{symbol}.csv',sep='\t')
        
        else:
            if 'preprocessed' in self.cfg.TWITTER_DATASET_DIR:
                df = pd.read_feather(f'{self.cfg.TWITTER_DATASET_DIR}/{symbol}.ftr')
                print(df.head())
            else:
                df = pd.read_csv(f'{self.cfg.TWITTER_DATASET_DIR}/{symbol}.csv',sep='\t')
                

        start_date_in_datetime = date.fromisoformat(self.start_date)
        end_date_in_datetime = date.fromisoformat(self.end_date)

        date_list = df['date'].tolist()
        date_list = [date.fromisoformat(d) for d in date_list]
        date_list_in_datetime = [d for d in date_list if self.is_in_term(d, start_date_in_datetime, end_date_in_datetime)]
        date_list = [d.isoformat() for d in date_list_in_datetime]
        df = df[df['date'].isin(date_list)]

        df = df[df['time_delta'] == self.time_delta]
        df = df.reset_index()

        df = df.drop_duplicates(['date','text'])

        if self.is_filtering:
            df['text'] = df['text'].map(lambda x: self.tokenizer.encode(
            x,
            padding='max_length',
            truncation=True,
            ))

        else:
            if 'preprocessed' in self.cfg.TWITTER_DATASET_DIR:
                df['text'] = df['text'].map(lambda x: self.tokenizer.encode(
                x,
                padding='max_length',
                truncation=True,
                ))

            else:
                df['text'] = df['text'].map(lambda x: self.tokenizer.encode(
                self.preprocess_text(x,symbol.lower()),
                padding='max_length',
                truncation=True,
                ))

        

        return df

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        batch_size=1,
        num_workers=0,
        time_delta=1,
        option='all'
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.time_delta = time_delta
        self.option = option
        self.set_datasets()

    def set_datasets(self):
        if self.option == 'train' or self.option == 'all':
            self.train_dataset = CustomDataset(
                cfg=self.cfg,
                start_date='2014-01-01',
                # end_date='2015-07-31',
                # end_date='2015-09-30',
                end_date='2015-12-31',
                time_delta=self.time_delta,
            )
        # if self.option == 'valid' or self.option == 'all':
        #     self.valid_dataset = CustomDataset(
        #         cfg=self.cfg,
        #         start_date='2015-08-01',
        #         end_date='2015-09-30',
        #         time_delta=self.time_delta,
        #     )

        if self.option == 'test' or self.option == 'all':
            self.test_dataset = CustomDataset(
                cfg=self.cfg,
                start_date='2015-10-01',
                end_date='2015-12-31',
                time_delta=self.time_delta,
            )

        if self.option == 'filter':
            self.test_dataset = CustomDataset(
                cfg=self.cfg,
                start_date='2014-01-01',
                end_date='2015-12-31',
                time_delta=self.time_delta,
            )

            self.filter_dataset = CustomDataset(
                cfg=self.cfg,
                start_date='2014-01-01',
                end_date='2015-12-31',
                time_delta=self.time_delta,
                is_filtering=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def filter_dataloader(self):
        return DataLoader(
            self.filter_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

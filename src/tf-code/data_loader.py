import os
import pandas as pd
import numpy as np
import tensorflow as tf

class DataLoader:
    def __init__(self, dataset_host, dataset_file, all_cols):
        self.dataset_host = dataset_host
        self.dataset_file = dataset_file
        self.all_cols = all_cols
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.train_mean = None
        self.train_std = None

    @staticmethod
    def is_gzip_file(filepath):
        with open(filepath, 'rb') as f:
            return f.read(2) == b'\x1f\x8b'

    def download_and_load(self):
        csv_path = tf.keras.utils.get_file(origin=self.dataset_host + self.dataset_file)
        if self.is_gzip_file(csv_path):
            df = pd.read_csv(csv_path, compression='gzip')
        else:
            df = pd.read_csv(csv_path)
        self.df = df
        return df

    def preprocess(self):
        df = self.df
        if df.get('Datetime') is not None:
            if df.index.dtype != 'datetime64[ns]':
                df.index = pd.to_datetime(df['Datetime'])
                df.pop('Datetime')
        # Zero row detection
        zero_columns = [col for col in self.all_cols]
        valid_columns = [col for col in zero_columns if col in df.columns]
        mask_zeros = df[valid_columns].eq(0).all(axis=1)
        zero_rows = df[mask_zeros].copy()
        # Add cyclical time features
        timestamp_s = df.index.map(pd.Timestamp.timestamp)
        day = 24*60*60
        year = (365.2425)*day
        df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        self.df = df
        return df

    def split_and_normalize(self):
        df = self.df
        n = len(df)
        self.train_df = df[0:int(n*0.7)]
        self.val_df = df[int(n*0.7):int(n*0.9)]
        self.test_df = df[int(n*0.9):]
        self.train_mean = self.train_df.mean()
        self.train_std = self.train_df.std()
        self.train_df = (self.train_df - self.train_mean) / self.train_std
        self.val_df = (self.val_df - self.train_mean) / self.train_std
        self.test_df = (self.test_df - self.train_mean) / self.train_std
        return self.train_df, self.val_df, self.test_df
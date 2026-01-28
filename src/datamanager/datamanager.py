import os,sys
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_PATH)

from src.config.config import Config

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np

class UserDataset(Dataset):
    def __init__(self, user_groups, user_labels, num_items=29502):
        self.user_ids = list(user_groups.keys())
        # input data : list of user history tensor
        self.data = [torch.tensor(user_groups[uid].values) for uid in self.user_ids]
        # label data: list of purchased item by user
        self.labels = [user_labels.get(uid, []) for uid in self.user_ids]
        self.num_items = num_items

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        # 입력 히스토리
        history = self.data[idx]
        
        # Convert label list to vector
        # ex: when user purchases 1, 3 item [0, 1, 0, 1, 0, ...]
        label_vector = torch.zeros(self.num_items)
        if self.labels[idx]:
            label_vector[self.labels[idx]] = 1.0
            
        return history, label_vector
        
def collate_fn(batch):
    '''
    batch is tuple list of (history, label).
    '''
    histories = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch]) # [B, num_items]
    
    # 히스토리만 최대 길이에 맞춰 패딩 [B, L_max, 6]
    histories_padded = pad_sequence(histories, batch_first=True, padding_value=0.0)
    
    return histories_padded, labels

class Datamanager:
    def __init__(self, config) :
        self.config = config

    def load_data(self, path: str) -> tuple[pd.DataFrame, pd.DataFrame] : 
        '''
        This function loads data from a parquet file.
        It creates index mappings for users and items,
        adds the corresponding indices to the DataFrame,
        and converts the event_time column to datetime,
        sorts the data by event_time,
        and splits the dataset into a 9:1 ratio.

        :param path: data path of parquet
        :type path: str
        :return: loaded train data
        '''
        df = pd.read_parquet(path)
        print("[datamanager] Data load success.")

        user2idx = {v: k for k, v in enumerate(df['user_id'].unique())}
        self.idx2user = {k: v for k, v in enumerate(df["user_id"].unique())}
        item2idx = {v: k for k, v in enumerate(df['item_id'].unique())}
        self.idx2item = {k: v for k, v in enumerate(df['item_id'].unique())}
        brand2idx = {v: k for k, v in enumerate(df['brand'].unique())}
        self.idx2brand = {k: v for k, v in enumerate(df['brand'].unique())}
        cat2idx = {v: k for k, v in enumerate(df['category_code'].unique())}
        self.idx2cat = {k: v for k, v in enumerate(df['category_code'].unique())}
        event2idx = {v: k for k, v in enumerate(df['event_type'].unique())}
        self.idx2event = {k: v for k, v in enumerate(df['event_type'].unique())}

        df['user_idx'] = df['user_id'].map(user2idx)
        df['item_idx'] = df['item_id'].map(item2idx)
        df['brand_idx'] = df['brand'].map(brand2idx)
        df['category_idx'] = df['category_code'].map(cat2idx)
        df['event_type_idx'] = df['event_type'].map(event2idx)
        print("[datamanager] Append user-index data in df")
        print("[datamanger] df : \n", df[['user_idx', 'item_idx', 'brand_idx', 'category_idx']].head())

        df['event_time'] = pd.to_datetime(df['event_time'], format='%Y-%m-%d %H:%M:%S %Z')
        ref_time = df["event_time"].min()
        df["event_hour_float"] = (df["event_time"] - ref_time).dt.total_seconds() / 3600
        df = df.sort_values('event_time', ascending=True)
        print("[datamanager] Event time Sorting success")

        # Split train, test df by date
        start_date = pd.to_datetime('2020-02-22 00:00:00', utc=True)
        end_date   = pd.to_datetime('2020-02-29 23:59:59', utc=True)

        train_df = df[df['event_time'] < start_date]
        test_df = df[(df['event_time'] >= start_date) & (df['event_time'] <= end_date)]

        test_df = test_df[test_df["event_type"] == "purchase"]
        test_df = test_df[["user_idx", "item_idx"]]

        # Create labels by applying list {user_idx : [item0, item1, item2...]}
        user_labels = test_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
        all_train_users = train_df['user_idx'].unique()
        # If there is no purchasing, return empty list
        final_labels = {u: user_labels.get(u, []) for u in all_train_users}

        print(f"[datamanager] Label mapping success. Total labeled users: {len(user_labels)}")

        return train_df, final_labels
    
    def analysis_data(self, train_df: pd.DataFrame):
        '''
        This function analysis df for target column
        Set target_column like "purchase", "view" ...

        :param train_df: Description
        '''
        target_column = "view"
        print(f"[datamanager] Analysising data about {target_column}")
        target_df = train_df[train_df["event_type"] == target_column]
        user_target_counts = (
        target_df
        .groupby("user_idx")
        .size()
        )
        min_target = user_target_counts.min()
        mean_target = user_target_counts.mean()
        max_target = user_target_counts.max()

        print(f"[datamanager] Min {target_column} per user:", min_target)
        print(f"[datamanager] Mean {target_column} per user:", mean_target)
        print(f"[datamanager] Max {target_column} per user:", max_target)

    def split_data_per_user(self, df: pd.DataFrame, max_len: int = 100) -> dict[int, pd.DataFrame] :
        '''
        This function Split data per user
        
        :param df: train_df which contains target_cols
        :type df: pd.DataFrame

        :return: Description
        :rtype: dict[int, DataFrame]
        '''
        target_cols = ['user_idx', 'item_idx', 'brand_idx', 'category_idx', 
                    'price', 'event_hour_float', 'event_type_idx']
        user_groups = {}

        for idx, group in df[target_cols].groupby('user_idx'):
            if len(group) > max_len:
                group = group.iloc[-max_len:]
            # Exclude user_idx because of user_idx for key
            user_groups[idx] = group.drop(columns=['user_idx'])
            
        return user_groups
    
    def prepare_dataloader(self) -> DataLoader:
        train_df, final_labels = self.load_data(self.config.data["data_path"])
        user_groups = self.split_data_per_user(train_df, max_len=100)
        num_items = len(datamanager.idx2item) 
        dataset = UserDataset(user_groups, final_labels, num_items)

        dataloader = DataLoader(
            dataset,
            batch_size= self.config.train['batch_size'],
            shuffle=True,
            collate_fn=collate_fn
        )
        for histories, labels in dataloader:
            print("[datamanager]Input Shape:", histories.shape) # [32, 100, 6]
            print("[datamanager]Label Shape:", labels.shape)    # [32, 29502]
            break
        return dataloader


if __name__ == "__main__" :
    config = Config()
    datamanager = Datamanager(config)
    # train_df, test_df = datamanager.load_data(config.data["data_path"])
    # datamanager.split_data_per_user(train_df)
    # datamanager.analysis_data(train_df)
    datamanager.prepare_dataloader()
    # print("[datamanger] df max : \n", train_df["event_hour_float"].max())
    # print("[datamanager] dtype : ", type(train_df))

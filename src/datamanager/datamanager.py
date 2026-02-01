import os,sys
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_PATH)

from src.config.config import Config

import torch
import pickle
from typing import List
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd
import numpy as np

class UserDataset(Dataset):
    def __init__(self, user_groups, user_labels=None, num_items=29502):
        self.user_ids = list(user_groups.keys())
        self.user_groups = user_groups
        self.user_labels = user_labels
        self.num_items = num_items

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]

        history_df = self.user_groups[uid]
        history_tensor = torch.tensor(history_df.values, dtype=torch.float32)
        # Training mode
        if self.user_labels is not None:
            target_items = self.user_labels[uid]
            multi_hot = torch.zeros(self.num_items+1, dtype=torch.float32)
            if len(target_items) > 0:
                multi_hot[target_items] = 1.0
            return history_tensor, multi_hot
        # Inference mode
        else :
            return history_tensor, uid
        
class RecommandedDataset(Dataset):
    """
    :param history_list: 추출된 history DataFrame들의 리스트
    :param labels: 추출된 target labels DataFrame
    """
    def __init__(self, user_idx:List ,history: List[pd.DataFrame], labels: pd.DataFrame=None, max_len: int = 100, num_items=29502):
        self.user_idx = user_idx
        self.history = history
        self.labels = labels
        self.max_len = max_len
        self.num_items = num_items

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        hist_df = self.history[idx]
        uid = self.user_idx[idx]

        values = hist_df.values
        curr_len = len(values)
        num_features = values.shape[1]
        # Padding 
        padded_hist = np.zeros((self.max_len, num_features), dtype=np.float32)

        if curr_len > 0:
            # max_len보다 길면 자르고, 짧으면 있는 만큼만 채움
            actual_len = min(curr_len, self.max_len)
            padded_hist[-actual_len:] = values[-actual_len:]

        hist_tensor = torch.from_numpy(padded_hist)

        if self.labels is None:
            return torch.tensor(uid, dtype=torch.long), hist_tensor

        else:
            label = self.labels.iloc[idx]['item_idx']
            multi_hot = torch.zeros(self.num_items+1, dtype=torch.float32)
            multi_hot[label] = 1.0
            return (
                    torch.tensor(uid, dtype=torch.long), 
                    hist_tensor, 
                    multi_hot
            )
        
def collate_fn(batch):
    '''
    batch: 
      - 학습 시: [(history, label), (history, label), ...]
      - 추론 시: [(history, uid), (history, uid), ...]
    '''
    histories = [item[0] for item in batch]
    
    # 히스토리 패딩 (학습/추론 공통)
    histories_padded = pad_sequence(histories, batch_first=True, padding_value=0)
    
    # 두 번째 요소가 label(Tensor)인지 uid(Scalar)인지 확인
    second_elements = [item[1] for item in batch]
    
    if isinstance(second_elements[0], torch.Tensor):
        # 학습 모드: label들을 stack하여 [B, num_items] 반환
        targets = torch.stack(second_elements)
    else:
        # 추론 모드: uid들을 Tensor로 변환하여 반환
        targets = torch.tensor(second_elements, dtype=torch.long)
    
    return histories_padded, targets

class Datamanager:
    def __init__(self, config) :
        self.config = config

    def load_data(self, path: str) -> pd.DataFrame : 
        '''
        This function loads data from a parquet file.
        It creates index mappings for users and items,
        adds the corresponding indices to the DataFrame,
        and converts the event_time column to datetime,
        sorts the data by event_time,

        :param path: data path of parquet
        :type path: str
        :return: loaded train data
        '''
        df = pd.read_parquet(path)
        print("[datamanager] Data load success.")
        total_purchase = df[df['event_type'] == "view"]
        print("[datamanager] Total purchase : ", len(total_purchase))

        # Popular Item
        popular_top10 = total_purchase['item_id'].value_counts().head(10)

        print("[datamanager]=== Real World Popular Items (Top 10) ===")
        print(popular_top10)

        # 만약 비율로 보고 싶다면 (전체 구매 중 차지하는 비중)
        popular_ratio = (popular_top10 / len(total_purchase)) * 100
        print("\n[datamanager]=== Popularity Ratio (%) ===")
        print(popular_ratio)
                
        df['brand'] = df['brand'].fillna('unknown')
        df['category_code'] = df['category_code'].fillna('unknown')

        self.user2idx = {v: k+1 for k, v in enumerate(df['user_id'].unique())}
        self.idx2user = {k+1: v for k, v in enumerate(df["user_id"].unique())}
        self.item2idx = {v: k+1 for k, v in enumerate(df['item_id'].unique())}
        self.idx2item = {k+1: v for k, v in enumerate(df['item_id'].unique())}
        self.brand2idx = {v: k+1 for k, v in enumerate(df['brand'].unique())}
        self.idx2brand = {k+1: v for k, v in enumerate(df['brand'].unique())}
        self.cat2idx = {v: k+1 for k, v in enumerate(df['category_code'].unique())}
        self.idx2cat = {k+1: v for k, v in enumerate(df['category_code'].unique())}
        self.event2idx = {v: k+1 for k, v in enumerate(df['event_type'].unique(), start=1)}
        self.idx2event = {k+1: v for k, v in enumerate(df['event_type'].unique(), start=1)}

        df['user_idx'] = df['user_id'].map(self.user2idx)
        df['item_idx'] = df['item_id'].map(self.item2idx)
        df['brand_idx'] = df['brand'].map(self.brand2idx)
        df['category_idx'] = df['category_code'].map(self.cat2idx)
        df['event_type_idx'] = df['event_type'].map(self.event2idx)
        print("[datamanager] Append user-index data in df")
        print("[datamanager] user, item, brand, category len : ", len(self.user2idx),len(self.item2idx),len(self.brand2idx),len(self.cat2idx),len(self.event2idx))
        # print("[datamanger] df : \n", df[['user_idx', 'item_idx', 'brand_idx', 'category_idx']].head())

        df['event_time'] = pd.to_datetime(df['event_time'], format='%Y-%m-%d %H:%M:%S %Z')
        ref_time = df["event_time"].min()
        df["event_hour_float"] = (df["event_time"] - ref_time).dt.total_seconds() / 3600
        df = df.sort_values('event_time', ascending=True)
        print("[datamanager] Event time Sorting success")

        return df
    
    def split_data_per_user(self, df: pd.DataFrame, max_len: int = 100) -> dict[int, pd.DataFrame] :
        '''
        This function Split data per user
        
        :param df: train_df which contains target_cols
        :type df: pd.DataFrame
    

        :return: Description
        :rtype: dict[int, DataFrame], dict[int, DataFrame]
        '''
        target_cols = ['user_idx', 'item_idx', 'brand_idx', 'category_idx', 
                    'price', 'event_hour_float', 'event_type_idx']
        user_groups = {}

        grouped = df[target_cols].groupby('user_idx')
        for idx, group in tqdm(grouped, desc="Splitting data per user", total=grouped.ngroups):
            
            if len(group) > max_len:
                group = group.iloc[-max_len:]
            # Exclude user_idx because of user_idx for key
            user_groups[idx] = group.drop(columns=['user_idx'])

        return user_groups


    
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

    
    def prepare_dataloader(self,
                           train_groups=None,
                           train_labels=None,
                           val_groups=None,
                           val_labels=None,
                           all_user_dict=None,
                           num_items=29502,
                           prepare_infer=False
                           ) -> tuple[DataLoader, DataLoader]:
        if(train_groups is None):
            df = self.load_data(self.config.data["data_path"])
            max_len = self.config.train['max_len']
            df = df.sort_values(['user_idx', 'event_hour_float']).reset_index(drop=True)
            # print("[datamanager] : df sort values", df.head())

            mask = df["event_type_idx"].isin([
                self.event2idx['purchase'], 
                self.event2idx['cart']
            ])
            # print("[datamanager] mask : ", mask.head())

            all_labels = df[mask].copy()
            target_indices = all_labels.index
            # print("[datamanager] target_indices : ", target_indices)

            all_history = []
            all_user_ids = []
            target_cols = ['user_idx', 'item_idx', 'brand_idx', 'category_idx', 
                    'price', 'event_hour_float', 'event_type_idx']
            
            # infer_data = df.iloc[-max_len:]
            if prepare_infer:
                grouped_infer = df[target_cols].groupby('user_idx')
                infer_data = []
                infer_user_ids = []

                for user_id, group in tqdm(grouped_infer, desc="Preparing infer data"):
                    # 유저 식별을 위해 ID 저장
                    infer_user_ids.append(user_id)
                    
                    # 마지막 max_len개 추출 및 user_idx 드롭 (Dataset 내부 로직에 맞춰 일관성 유지)
                    history = group.tail(max_len).drop(columns=['user_idx'])
                    infer_data.append(history)

            for idx in tqdm(target_indices, desc="Extracting history sequences"):
                user_id = df.at[idx, 'user_idx']
                
                # 이전 max_len만큼 자르기
                start_idx = max(0, idx - max_len)
                history = df.iloc[start_idx:idx][target_cols].copy()
                
                # Another user filtering
                history = history[history['user_idx'] == user_id]
                history = history.drop(columns="user_idx")

                # Extract target_cols
                all_history.append(history)
                all_user_ids.append(user_id)

            all_history_idx = list(range(len(all_history)))
                
            train_indices, val_indices = train_test_split(
                    all_history_idx, 
                    test_size=0.2, 
                    random_state=42, 
                    shuffle=True
                )
            train_user_idx = [all_user_ids[i] for i in train_indices]
            train_x = [all_history[i] for i in train_indices]
            train_label = all_labels.iloc[train_indices]

            val_user_idx = [all_user_ids[i] for i in val_indices]
            val_x = [all_history[i] for i in val_indices]
            val_label = all_labels.iloc[val_indices]


            # self.save_data("infer_data.pkl", user_groups=all_user_dict)
            

        train_dataset = RecommandedDataset(train_user_idx, train_x, train_label, self.config.train['max_len'])
        val_dataset = RecommandedDataset(val_user_idx, val_x, val_label, self.config.train['max_len'])
        if prepare_infer:
            infer_dataset = RecommandedDataset(user_idx=infer_user_ids, history=infer_data, labels=None, max_len=self.config.train['max_len'])

        train_dataloader = DataLoader(
            train_dataset,
            batch_size= self.config.train['batch_size'],
            shuffle=True,
        )

        valid_dataloader = DataLoader(
            val_dataset,
            batch_size= self.config.train['batch_size'],
            shuffle=False,
        )
        if prepare_infer:
            infer_dataloader = DataLoader(
                infer_dataset,
                batch_size= self.config.train['batch_size'],
                shuffle=False,
            )
        else :
            infer_dataloader = None

        for _, histories, labels in train_dataloader:
            print("[datamanager]Input Shape:", histories.shape) # [32, 100, 6]
            print("[datamanager]Label Shape:", labels.shape)    # [32, 29502+1]
            break

        print(f"[datamanager] Train samples: {len(train_dataset)}, Valid samples: {len(val_dataset)}")

        return train_dataloader, valid_dataloader, infer_dataloader
    
    def save_data(self, file_name, user_groups, labels=None, num_items=29502):
        path = os.path.join(self.config.data['pickle_data_path'], f"{file_name}")
        if labels is not None:
            data_to_save = {
                'user_groups': user_groups,
                'labels': labels,
                'num_items': num_items
            }
            with open(path, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"Successfully saved to {path}")
        else:
            data_to_save = {
                'user_groups': user_groups,
                'num_items': num_items
            }
            with open(path, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"Successfully saved to {path}")    

    def load_pickle_data(self, file_name):
        with open(os.path.join(self.config.data['pickle_data_path'], f"{file_name}"), 'rb') as f:
            loaded_data = pickle.load(f)
        return loaded_data

if __name__ == "__main__" :
    config = Config()
    datamanager = Datamanager(config)
    # train_df, test_df = datamanager.load_data(config.data["data_path"])
    # datamanager.split_data_per_user(train_df)
    # datamanager.analysis_data(train_df)
    datamanager.prepare_dataloader()
    # print("[datamanger] df max : \n", train_df["event_hour_float"].max())
    # print("[datamanager] dtype : ", type(train_df))

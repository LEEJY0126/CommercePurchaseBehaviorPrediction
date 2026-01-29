import os,sys
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_PATH)

from src.config.config import Config

import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd
import numpy as np

class UserDataset(Dataset):
    def __init__(self, user_groups, user_labels, num_items=29502):
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

        target_items = self.user_labels[uid]
        multi_hot = torch.zeros(self.num_items, dtype=torch.float32)
        if len(target_items) > 0:
            multi_hot[target_items] = 1.0
        return history_tensor, multi_hot
        
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

    def load_data(self, path: str, date: str) -> tuple[pd.DataFrame, pd.DataFrame] : 
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
        
        df['brand'] = df['brand'].fillna('unknown')
        df['category_code'] = df['category_code'].fillna('unknown')

        user2idx = {v: k for k, v in enumerate(df['user_id'].unique())}
        self.idx2user = {k: v for k, v in enumerate(df["user_id"].unique())}
        item2idx = {v: k for k, v in enumerate(df['item_id'].unique())}
        self.idx2item = {k: v for k, v in enumerate(df['item_id'].unique())}
        brand2idx = {v: k for k, v in enumerate(df['brand'].unique())}
        self.idx2brand = {k: v for k, v in enumerate(df['brand'].unique())}
        cat2idx = {v: k for k, v in enumerate(df['category_code'].unique())}
        self.idx2cat = {k: v for k, v in enumerate(df['category_code'].unique())}
        event2idx = {v: k for k, v in enumerate(df['event_type'].unique(), start=1)}
        self.idx2event = {k: v for k, v in enumerate(df['event_type'].unique(), start=1)}

        df['user_idx'] = df['user_id'].map(user2idx)
        df['item_idx'] = df['item_id'].map(item2idx)
        df['brand_idx'] = df['brand'].map(brand2idx)
        df['category_idx'] = df['category_code'].map(cat2idx)
        df['event_type_idx'] = df['event_type'].map(event2idx)
        print("[datamanager] Append user-index data in df")
        print("[datamanager] user, item, brand, category len : ", len(user2idx),len(item2idx),len(brand2idx),len(cat2idx),len(event2idx))
        # print("[datamanger] df : \n", df[['user_idx', 'item_idx', 'brand_idx', 'category_idx']].head())

        df['event_time'] = pd.to_datetime(df['event_time'], format='%Y-%m-%d %H:%M:%S %Z')
        ref_time = df["event_time"].min()
        df["event_hour_float"] = (df["event_time"] - ref_time).dt.total_seconds() / 3600
        df = df.sort_values('event_time', ascending=True)
        print("[datamanager] Event time Sorting success")

        # Split train, test df by date
        start_date = pd.to_datetime(date, utc=True)
        end_date = start_date + pd.Timedelta(days=7)
        train_df = df[df['event_time'] < start_date]
        test_df = df[(df['event_time'] >= start_date) & (df['event_time'] <= end_date)]

        # test df has only purchase event
        test_df = test_df[test_df["event_type"] == "purchase"]
        test_df = test_df[["user_idx", "item_idx"]]

        # Rest train_df only purchased users
        purchased_users = test_df["user_idx"].unique()
        train_df = train_df[train_df["user_idx"].isin(purchased_users)]
        print(f"[datamanager] 전체 구매 이벤트 수: {len(test_df)}")
        print(f"[datamanager] 구매에 참여한 유저 수: {test_df['user_idx'].nunique()}")
        print(f"[datamanager] 유저당 평균 구매 아이템 수: {len(test_df) / test_df['user_idx'].nunique():.2f}")

        # Create labels by applying list {user_idx : [item0, item1, item2...]}
        user_labels = test_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
        all_train_users = train_df['user_idx'].unique()
        # If there is no purchasing, return empty list
        final_labels = {u: user_labels.get(u, []) for u in all_train_users}

        print(f"[datamanager] Label mapping success. Total labeled users: {len(user_labels)}")

        return train_df, final_labels
    
    def load_combined_data(self, path: str, dates: list) -> tuple[pd.DataFrame, dict]:
        all_train_dfs = []
        combined_labels = {}

        # 1. 일단 전체 DF 로드 및 전처리는 한 번만 수행 (속도 최적화)
        df = pd.read_parquet(path)
        print("[datamanager] Data load success.")
        
        df['brand'] = df['brand'].fillna('unknown')
        df['category_code'] = df['category_code'].fillna('unknown')

        user2idx = {v: k for k, v in enumerate(df['user_id'].unique())}
        self.idx2user = {k: v for k, v in enumerate(df["user_id"].unique())}
        item2idx = {v: k for k, v in enumerate(df['item_id'].unique())}
        self.idx2item = {k: v for k, v in enumerate(df['item_id'].unique())}
        brand2idx = {v: k for k, v in enumerate(df['brand'].unique())}
        self.idx2brand = {k: v for k, v in enumerate(df['brand'].unique())}
        cat2idx = {v: k for k, v in enumerate(df['category_code'].unique())}
        self.idx2cat = {k: v for k, v in enumerate(df['category_code'].unique())}
        event2idx = {v: k for k, v in enumerate(df['event_type'].unique(), start=1)}
        self.idx2event = {k: v for k, v in enumerate(df['event_type'].unique(), start=1)}

        df['user_idx'] = df['user_id'].map(user2idx)
        df['item_idx'] = df['item_id'].map(item2idx)
        df['brand_idx'] = df['brand'].map(brand2idx)
        df['category_idx'] = df['category_code'].map(cat2idx)
        df['event_type_idx'] = df['event_type'].map(event2idx)
        print("[datamanager] Append user-index data in df")
        print("[datamanager] user, item, brand, category len : ", len(user2idx),len(item2idx),len(brand2idx),len(cat2idx),len(event2idx))
        # print("[datamanger] df : \n", df[['user_idx', 'item_idx', 'brand_idx', 'category_idx']].head())

        df['event_time'] = pd.to_datetime(df['event_time'], format='%Y-%m-%d %H:%M:%S %Z')
        ref_time = df["event_time"].min()
        df["event_hour_float"] = (df["event_time"] - ref_time).dt.total_seconds() / 3600
        df = df.sort_values('event_time', ascending=True)
        print("[datamanager] Event time Sorting success")

        for str_start in dates:
            print(f"[datamanager] Processing window starting at: {str_start}")
            
            start_date = pd.to_datetime(str_start, utc=True)
            end_date = start_date + pd.Timedelta(days=7)
            
            # 해당 윈도우 분리
            temp_train = df[df['event_time'] < start_date]
            temp_test = df[(df['event_time'] >= start_date) & (df['event_time'] <= end_date)]
            
            # 테스트 유저 필터링
            temp_test = temp_test[temp_test["event_type"] == "purchase"]
            purchased_users = temp_test["user_idx"].unique()
            
            # 학습 데이터 필터링 (해당 시점 구매자만)
            temp_train = temp_train[temp_train["user_idx"].isin(purchased_users)]
            
            # 정답지 생성 {user_idx: [items]}
            temp_labels = temp_test.groupby('user_idx')['item_idx'].apply(list).to_dict()

            # 데이터 보관 (각 윈도우의 유저 구분을 위해 식별자를 줄 수도 있지만, 
            # 단순히 패턴을 늘리는 거라면 그대로 합쳐도 무방합니다.)
            all_train_dfs.append(temp_train)
            
            # 레이블 통합 (중복 유저의 경우 리스트를 합침)
            for u, items in temp_labels.items():
                if u in combined_labels:
                    combined_labels[u].extend(items)
                    combined_labels[u] = list(set(combined_labels[u])) # 중복 아이템 제거
                else:
                    combined_labels[u] = items

        # 최종 데이터 통합
        final_train_df = pd.concat(all_train_dfs).drop_duplicates()
        
        # 실제 통합된 유저들만 레이블 생성
        final_train_users = final_train_df['user_idx'].unique()
        final_labels = {u: combined_labels[u] for u in final_train_users}

        print(f"[datamanager] Combined Data Success. Total Samples: {len(final_train_df)}")
        return final_train_df, final_labels
    
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

    def split_data_per_user(self, df: pd.DataFrame, max_len: int = 100) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]] :
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
    
    def prepare_dataloader(self,
                           date: str,
                           train_groups=None,
                           train_labels=None,
                           val_groups=None,
                           val_labels=None,
                           num_items=29502
                           ) -> tuple[DataLoader, DataLoader]:
        if(train_groups is None):
            train_df, final_labels = self.load_data(self.config.data["data_path"], date)
            data = self.split_data_per_user(train_df, max_len=100)
            num_items = len(self.idx2item) 
            train_dataset = UserDataset(data, final_labels, num_items)
            common_user_ids = list(data.keys())

            train_ids, val_ids = train_test_split(
                    common_user_ids, 
                    test_size=0.2, 
                    random_state=42, 
                    shuffle=True
                )
            train_groups = {uid: data[uid] for uid in train_ids}
            train_labels = {uid: final_labels[uid] for uid in train_ids}
            self.save_data("train_data.pkl",train_groups, train_labels)

            val_groups = {uid: data[uid] for uid in val_ids}
            val_labels = {uid: final_labels[uid] for uid in val_ids}
            self.save_data("val_data.pkl",val_groups, val_labels)

        train_dataset = UserDataset(train_groups, train_labels, num_items)
        val_dataset = UserDataset(val_groups, val_labels, num_items)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size= self.config.train['batch_size'],
            shuffle=True,
            collate_fn=collate_fn
        )

        valid_dataloader = DataLoader(
            val_dataset,
            batch_size= self.config.train['batch_size'],
            shuffle=True,
            collate_fn=collate_fn
        )


        for histories, labels in train_dataloader:
            print("[datamanager]Input Shape:", histories.shape) # [32, 100, 6]
            print("[datamanager]Label Shape:", labels.shape)    # [32, 29502]
            break

        print(f"[datamanager] Train samples: {len(train_dataset)}, Valid samples: {len(val_dataset)}")

        return train_dataloader, valid_dataloader
    
    def save_data(self, file_name, user_groups, labels, num_items=29502):
        path =os.path.join(self.config.data['pickle_data_path'], f"{file_name}")
        data_to_save = {
            'user_groups': user_groups,
            'labels': labels,
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

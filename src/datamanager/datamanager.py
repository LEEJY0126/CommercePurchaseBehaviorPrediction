import os,sys
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_PATH)

from src.config.config import Config

import pandas as pd
import numpy as np


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
        idx2user = {k: v for k, v in enumerate(df["user_id"].unique())}
        item2idx = {v: k for k, v in enumerate(df['item_id'].unique())}
        idx2item = {k: v for k, v in enumerate(df['item_id'].unique())}

        df['user_idx'] = df['user_id'].map(user2idx)
        df['item_idx'] = df['item_id'].map(item2idx)
        print("[datamanager] Append user-index data in df")

        df['event_time'] = pd.to_datetime(df['event_time'], format='%Y-%m-%d %H:%M:%S %Z')
        ref_time = df["event_time"].min()
        df["event_hour_float"] = (df["event_time"] - ref_time).dt.total_seconds() / 3600
        df = df.sort_values('event_time', ascending=True)
        print("[datamanager] Event time Sorting success")

        offset = int(0.9 * len(df))
        train_df, test_df = np.split(df, [offset])
        test_df = test_df[test_df["event_type"] == "purchase"]
        test_df = test_df[["user_idx", "item_idx"]]
        train_df["label"] = 1
        # user_item_matrix = train_df.groupby(["user_idx", "item_idx"])["label"].sum().reset_index()

        return train_df, test_df
    
    def analysis_data(self, train_df):
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


if __name__ == "__main__" :
    config = Config()
    datamanager = Datamanager(config)
    train_df, test_df = datamanager.load_data(config.data["data_path"])
    datamanager.analysis_data(train_df)
    # print("[datamanger] df max : \n", train_df["event_hour_float"].max())
    # print("[datamanager] dtype : ", type(train_df))

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
        print("[datamanager] Append user-index data in df]")

        df['event_time'] = pd.to_datetime(df['event_time'], format='%Y-%m-%d %H:%M:%S %Z')
        df = df.sort_values('event_time', ascending=True)
        print("[datamanager] Event time Sorting success")

        offset = int(0.9 * len(df))
        train_df, test_df = np.split(df, [offset])
        test_df = test_df[test_df["event_type"] == "purchase"]
        test_df = test_df[["user_idx", "item_idx"]]
        train_df["label"] = 1
        user_item_matrix = train_df.groupby(["user_idx", "item_idx"])["label"].sum().reset_index()

        return user_item_matrix, test_df


if __name__ == "__main__" :
    config = Config()
    datamanager = Datamanager(config)
    train_df, test_df = datamanager.load_data(config.data["data_path"])
    print("[datamanger] df : \n", train_df.head())
    print("[datamanager] dtype : ", type(train_df))

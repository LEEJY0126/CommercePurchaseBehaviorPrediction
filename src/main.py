import sys,os
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(PROJECT_PATH)

from src.model.model2 import PurchasePred
from src.config.config import Config

if __name__ == "__main__":
    config = Config()
    _model = PurchasePred(config)

    dates = [config.data["start_date"][0], config.data["start_date"][1], config.data["start_date"][2]]


    train_dataloader, valid_dataloader, infer_dataloader = _model.prepare_dataloader()
    
    # print("[main] Data loading from pickle...")
    # train_pickle_name = "train_data.pkl"
    # val_pickle_name = "val_data.pkl"
    # infer_pickle_name = "infer_data.pkl"
    # train_pickle = _model.datamanager.load_pickle_data(train_pickle_name)
    # val_pickle = _model.datamanager.load_pickle_data(val_pickle_name)
    # infer_pickle = _model.datamanager.load_pickle_data(infer_pickle_name)
    # train_dataloader, valid_dataloader, infer_dataloader = _model.datamanager.prepare_dataloader(
    #     date=dates[2],
    #     train_groups=train_pickle['user_groups'],
    #     train_labels=train_pickle['labels'],
    #     val_groups=val_pickle['user_groups'],
    #     val_labels=val_pickle['labels'],
    #     all_user_dict=infer_pickle['user_groups']
    # )
    
    _model.train(train_dataloader, valid_dataloader)
    _model.infer(infer_dataloader, "output1.csv")

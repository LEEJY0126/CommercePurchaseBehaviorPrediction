import os,sys
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_PATH)
from src.config.config import Config
from src.datamanager.datamanager import Datamanager

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

class ItemEncoder(nn.Module):
    '''
    Data will come by history unit [item_idx, brand_idx, category_idx, price]
    
    return: item embedding
    '''
    def __init__(self,
                 num_items=29502,
                 num_brands=1859,
                 num_categories=24,
                 d_item=64,
                 d_brand=16,
                 d_categories=16,
                 d_price=16
    ):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, d_item)
        self.brand_emb = nn.Embedding(num_brands, d_brand)
        self.cat_emb = nn.Embedding(num_categories, d_categories)

        self.price_proj = nn.Linear(1, d_price)

        self.item_fusion = nn.Linear(d_item + d_brand + d_categories + d_price, d_item)


    def forward(self, x): # x: [B, 4]
        item_idx  = x[:, 0].long()
        brand_idx = x[:, 1].long()
        cat_idx   = x[:, 2].long()
        price     = x[:, 3].float().unsqueeze(-1)

        item_e  = self.item_emb(item_idx)
        brand_e = self.brand_emb(brand_idx)
        cat_e   = self.cat_emb(cat_idx)
        price_e = self.price_proj(price)

        item_repr = torch.cat([item_e, brand_e, cat_e, price_e], dim=-1)
        item_repr = self.item_fusion(item_repr)

        return item_repr
    

class EventEncoder(nn.Module):
    '''
    Data will come like 5 user purchase histoy data 
    [item_idx, brand_idx, category_idx, price, event_time(float, hours), event_type]
    [B, 6] -> [B,64 + 16] -> [B, 128]
    '''
    def __init__ (self, num_event_types = 3):
        super().__init__()
        self.item_encoder = ItemEncoder()
        self.event_type_emb = nn.Embedding(num_event_types + 1, 8, padding_idx=0)
        self.time_proj = nn.Linear(1, 8)
        self.layer = nn.Linear(64 + 16, 128)

    def forward (self, x):
        item_property = x[:, :4]
        event_time = x[:, 4].float().unsqueeze(-1)
        event_type = x[:, 5].long()
        item_e = self.item_encoder(item_property) # [B, 64]
        event_type_e = self.event_type_emb(event_type) # [B, 8]
        event_time_e = self.time_proj(event_time)
        emb_cat = torch.cat([item_e, event_type_e, event_time_e], dim=-1) # [B, 80]
        logits = self.layer(emb_cat) # [B, 128]

        return logits
    
class UserBehaviorModel(nn.Module):
    def __init__(self, num_items=29502, d_model=80):
        super().__init__()
        self.event_encoder = EventEncoder() 
        self.rnn = nn.GRU(input_size=128, hidden_size=256, batch_first=True) # EventEncoder output shape : [B, 128]
        
        # Output layer produces Linear whcih has num_items
        self.predictor = nn.Linear(256, num_items)

    def forward(self, x):
        # x shape: [Batch, Seq_len, 6]
        b, l, f = x.shape

        # 1. Convert event to vector within all sequence and batch
        x_flat = x.view(-1, f)
        e_flat = self.event_encoder(x_flat) # [B*L, 128]
        e_seq = e_flat.view(b, l, -1)       # [B, L, 128]
        if torch.isnan(e_seq).any() or torch.isinf(e_seq).any():
            print("Input to RNN contains NaN or Inf!")
        
        # 2. Summary user's purchase pattern to vector
        _, hn = self.rnn(e_seq)             # hn: [1, B, 256]
        user_vector = hn.squeeze(0)         # [B, 256]
        
        # 3. output item probability
        logits = self.predictor(user_vector) # [B, 29502]
        return logits
    
class PurchasePred :
    def __init__(self, config):
        self.config = config
        self.model = UserBehaviorModel()
        self.datamanager = Datamanager(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
    
    def train_one_epoch(self, dataloader,optimizer) -> float:
        self.model.train()
        pos_weight = torch.tensor([10000.0]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        total_loss = 0.0

        pbar = tqdm(dataloader, desc="Training", leave=False)
        for histories, labels in pbar:
            histories = histories.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()

            logits = self.model(histories)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss
            l_min = logits.min().item()
            l_max = logits.max().item()
            s_mean = torch.sigmoid(logits).mean().item()

            pbar.set_postfix(loss=f"{current_loss:.8f}")
            pbar.set_postfix({
                'loss': f"{loss.item():.8f}",
                'range': f"[{l_min:.1f}, {l_max:.1f}]"
                # 'sig_m': f"{s_mean:.4f}"
            })
        avg_loss = total_loss / len(dataloader)

        return avg_loss
    
    def train(self, train_dataloader, valid_dataloader):
        num_epoch = self.config.train['num_epoch']
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.train['lr'])
        for epoch in range(num_epoch):
            train_loss = self.train_one_epoch(train_dataloader, optimizer)
            print(f"[Epoch {epoch+1}/{num_epoch}] Train loss = {train_loss:.8f}")
            valid_loss, ndcg10 = self.validate(valid_dataloader)
            print(f"\n[model1] Validate end \nvalid_loss: {valid_loss:.8f} ndcg10: {ndcg10:.6f}")
    
    @torch.no_grad()
    def validate(self, dataloader) -> tuple[float, float]:
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        total_ndcg = 0.0
        match_count = 0  # 실제로 맞춘 유저 수 카운트

        for histories, labels in tqdm(dataloader, desc="validate", leave=False):
            histories = histories.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(histories)
            
            # Loss 계산
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # NDCG 계산을 위한 Top-K
            _, top_indices = torch.topk(logits, k=10, dim=-1)
            top_indices = top_indices.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(labels_np)):
                # 1. 해당 유저의 정답 아이템 세트
                true_indices = np.where(labels_np[i] == 1)[0]
                if len(true_indices) == 0:
                    continue

                true_indices_set = set(true_indices)

                # 2. DCG 계산
                dcg = 0.0
                for rank, idx in enumerate(top_indices[i]):
                    if idx in true_indices_set:
                        dcg += 1.0 / np.log2(rank + 2)

                # 3. IDCG 계산 (이상적인 정답 순위)
                idcg = 0.0
                num_hits = min(len(true_indices_set), 10)
                for rank in range(num_hits):
                    idcg += 1.0 / np.log2(rank + 2)

                # 4. NDCG 합산 및 Bingo 출력
                if idcg > 0:
                    current_ndcg = dcg / idcg
                    total_ndcg += current_ndcg
                    
                    if current_ndcg > 0:
                        match_count += 1
                        # 너무 많이 찍히면 지저분하므로 첫 5명까지만 Bingo 출력
                        # if match_count <= 5:
                        #     print(f" [Bingo!] User Match Found - NDCG@10: {current_ndcg:.4f}")

        avg_loss = total_loss / len(dataloader)
        # 전체 유저 수로 나눠서 평균 NDCG 산출
        avg_ndcg = total_ndcg / len(dataloader.dataset)
        
        print(f" >> Validation Finished. Total Matches: {match_count}/{len(dataloader.dataset)}")
        return avg_loss, avg_ndcg


    def prepare_dataloader(self, date: str):
        train_dataloader, valid_dataloader = self.datamanager.prepare_dataloader(date)
        return train_dataloader, valid_dataloader


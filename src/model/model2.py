import os,sys
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_PATH)
from src.config.config import Config
from src.datamanager.datamanager import Datamanager

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.manifold import TSNE
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
        self.item_emb = nn.Embedding(num_items+1, d_item, padding_idx=0)
        self.brand_emb = nn.Embedding(num_brands+1, d_brand, padding_idx=0)
        self.cat_emb = nn.Embedding(num_categories+1, d_categories, padding_idx=0)

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
    def __init__ (self, num_event_types = 4):
        super().__init__()
        self.item_encoder = ItemEncoder()
        self.event_type_emb = nn.Embedding(num_event_types + 1, 8, padding_idx=0)
        self.time_proj = nn.Linear(1, 8)
        self.layer = nn.Linear(64 + 16, 256)

    def forward (self, x):
        item_property = x[:, :4]
        event_time = x[:, 4].float().unsqueeze(-1)
        event_type = x[:, 5].long()
        
        item_e = self.item_encoder(item_property) # [B, 64]
        event_type_e = self.event_type_emb(event_type) # [B, 8]
        event_time_e = self.time_proj(event_time)
        emb_cat = torch.cat([item_e, event_type_e, event_time_e], dim=-1) # [B, 80]
        logits = self.layer(emb_cat) # [B, 256]

        return logits
    
class TransformerRecommender(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=2, max_len=100):
        super().__init__()
        self.d_model=d_model
        self.event_encoder = EventEncoder() # 기존에 만드신 것 활용
        
        # 트랜스포머는 위치 정보를 모르기 때문에 포지셔널 임베딩 추가 필요
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model)) 
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.predictor = nn.Linear(d_model, 29502+1)

    def forward(self, x):
        # x: [B, L, 6]
        b, l, f = x.shape

        has_data = (x[:, :, 0] > 0.5).any(dim=1)
        is_empty_seq = ~has_data
        # Padding mask
        padding_mask = (x[:, :, 0] == 0)

        x_flat = x.reshape(-1, f).contiguous()
        e_flat = self.event_encoder(x_flat)
        e_seq = e_flat.contiguous().view(b, l, -1)

        last_vector = torch.zeros((b, self.d_model), device=x.device)
        if has_data.any():
            # 데이터가 있는 샘플들만 골라냅니다.
            real_e_seq = e_seq[has_data]
            real_mask = (x[has_data, :, 0] < 0.5)
            
            # 포지셔널 임베딩 더하기
            real_e_seq = real_e_seq + self.pos_emb[:, :real_e_seq.size(1), :]
            
            # 트랜스포머 통과
            real_attn_out = self.transformer(real_e_seq, src_key_padding_mask=real_mask)
            
            # 결과 저장 (마지막 타임스텝)
            last_vector[has_data] = real_attn_out[:, -1, :]

        output = self.predictor(last_vector)

        return output
    
class PurchasePred :
    def __init__(self, config):
        self.config = config
        self.model = TransformerRecommender()
        self.datamanager = Datamanager(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_ndcg = 0
        self.best_epoch = 0

        self.model.to(self.device)
    
    def train_one_epoch(self, dataloader,optimizer) -> float:
        self.model.train()
        pos_weight = torch.tensor([15000.0]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        total_loss = 0.0

        pbar = tqdm(dataloader, desc="Training", leave=False)
        for ids, histories, labels in pbar:
            histories = histories.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()

            logits = self.model(histories)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss
            l_min = logits.min().item()
            l_max = logits.max().item()

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
            if ndcg10 > self.best_ndcg:
                print(f"[Epoch {epoch+1}/{num_epoch}]New best NDCG@10 : {ndcg10}")
                self.best_ndcg = ndcg10
                self.best_epoch = epoch+1
            print(f"\n[model1] Epoch {epoch+1}/{num_epoch} Validate End \nvalid_loss: {valid_loss:.8f} ndcg10: {ndcg10:.6f}")
        print(f"[model1] End training. \nBest NDCG@10 : {self.best_ndcg} / Epoch : {self.best_epoch}")
    
    @torch.no_grad()
    def validate(self, dataloader) -> tuple[float, float]:
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        total_ndcg = 0.0
        match_count = 0  # 실제로 맞춘 유저 수 카운트

        for ids, histories, labels in tqdm(dataloader, desc="validate", leave=False):
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
    
    @torch.no_grad
    def infer(self, dataloader, output_name, k=10):
        self.model.eval()
        all_recommendations = {}

        for ids, histories in tqdm(dataloader, desc="Inference"):
            histories = histories.to(self.device)
            logits = self.model(histories) # [batch_size, num_items]
            
            # 가장 높은 확률을 가진 아이템 인덱스 K개 추출
            top_k_probs, top_k_indices = torch.topk(logits, k=k, dim=-1)
            uids = ids.numpy() if torch.is_tensor(ids) else ids
            top_k_indices_cpu = top_k_indices.cpu().numpy()
            for batch_idx, uid in enumerate(uids):
                    # 인덱스를 실제 유저 ID와 아이템 ID로 변환
                    user_id = self.datamanager.idx2user[uid]
                    rec_items = [self.datamanager.idx2item[item_idx] for item_idx in top_k_indices_cpu[batch_idx]]

                    all_recommendations[user_id] = rec_items

        flattened_data = []
        for user_id, item_list in all_recommendations.items():
            for item_id in item_list:
                flattened_data.append([user_id, item_id])

        df_res = pd.DataFrame(flattened_data, columns=['user_id', 'item_id'])
        infer_path = os.path.join(self.config.data["output_path"], output_name)
        df_res.to_csv(infer_path, index=True)

        print(f"[model1] Saved {len(df_res)} recommendation rows to {infer_path}")
        
        return all_recommendations

    def prepare_dataloader(self):
        train_dataloader, valid_dataloader, infer_dataloader = self.datamanager.prepare_dataloader()
        return train_dataloader, valid_dataloader, infer_dataloader
    


    @torch.no_grad()
    def get_all_item_embeddings(self, item_metadata):
        """
        item_metadata: [num_items, 4] 형태의 numpy/tensor 
                    (item_idx, brand_idx, cat_idx, price) 포함
        """
        self.model.eval()
        # 전체 아이템 정보를 모델의 item_encoder에 넣음
        # 메모리 문제를 피하기 위해 배치 단위로 처리하는 것이 좋습니다.
        x = torch.tensor(item_metadata).to(self.device)
        item_embeddings = self.model.event_encoder.item_encoder(x) # [29502, 64]

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        item_tsne = tsne.fit_transform(item_embeddings)
        
        return item_embeddings.cpu().numpy()



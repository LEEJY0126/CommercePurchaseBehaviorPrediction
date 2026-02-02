import os,sys
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_PATH)
from src.config.config import Config
from src.datamanager.datamanager import Datamanager

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from tqdm import tqdm

D_ITEM = 64 # Item embedding
D_OTHER = 16 # Other Embedding, brands, categories, price
D_EVENT_TYPE = 8 # Event type Embedding
D_EVENT_TIME = 16 # Event time Embedding 
D_MODEL = 256 # Event output and Transformer input




class ItemEncoder(nn.Module):
    '''
    Data will come by history unit [item_idx, brand_idx, category_idx, price]
    
    return: item embedding
    '''
    def __init__(self,
                 num_items=29502,
                 num_brands=1859,
                 num_categories=24,
                 d_item=D_ITEM,
                 d_brand=D_OTHER,
                 d_categories=D_OTHER,
                 d_price=D_OTHER
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
    [B, 6] -> [B,64 + 16] -> [B, D_MODEL]
    '''
    def __init__ (self, 
                 d_item,
                 d_brand,
                 d_categories,
                 d_price,
                 d_event_type,
                 d_event_time,
                 num_event_types = 4, 
                 ):
        super().__init__()
        self.item_encoder = ItemEncoder(
                                        d_item=d_item,
                                        d_brand=d_brand,
                                        d_categories=d_categories,
                                        d_price=d_price
                            )
        self.event_type_emb = nn.Embedding(num_event_types + 1, d_event_type, padding_idx=0)
        self.time_proj = nn.Linear(1, d_event_time)
        self.layer = nn.Linear(D_ITEM + d_event_type + d_event_time, D_MODEL)

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
    def __init__(self,  
                 d_item,
                 d_other,
                 d_event_type, 
                 d_event_time,
                 d_model,
                 nhead, num_layers,
                 max_len, num_items=29502):
        super().__init__()
        self.d_model=d_model
        self.event_encoder = EventEncoder(
                             d_item=d_item,
                            d_brand=d_other,
                            d_categories=d_other,
                            d_price=d_other,
                            d_event_type=d_event_type, 
                            d_event_time=d_event_time,                     
                            )
        self.num_items = num_items
        
        # 트랜스포머는 위치 정보를 모르기 때문에 포지셔널 임베딩 추가 필요
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model)) 
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.predictor = nn.Linear(d_model, (self.num_items+1)*2)

    def forward(self, x):
        # x: [B, L, 6]
        b, l, f = x.shape

        has_data = (x[:, :, 0] > 0.5).any(dim=1)

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

        return output.view(-1, self.num_items+1, 2)
    
class PurchasePred :
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.model = TransformerRecommender(d_item=self.config.model["d_item"],
                                            d_other=self.config.model["d_other"],
                                            d_event_type=self.config.model["d_event_type"],
                                            d_event_time=self.config.model["d_event_time"],
                                            d_model=self.config.model["d_model"],
                                            nhead=self.config.model["nhead"],
                                            num_layers=self.config.model["num_layers"],
                                            max_len=self.config.train["max_len"]
                                            )
        self.datamanager = Datamanager(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.train['lr'])

        self.model.to(self.device)

    def calculate_loss(self, logits, targets, view_weight=1.0, purchase_weight=10.0):
        """
        logits: [Batch, Items, 2] -> index 0: view, index 1: purchase
        targets: [Batch, Items, 2]
        """
        pos_weight = torch.tensor([1000.0], device=logits.device)

        view_loss = F.binary_cross_entropy_with_logits(logits[:, :, 0], targets[:, :, 0], pos_weight)
        purchase_loss = F.binary_cross_entropy_with_logits(logits[:, :, 1], targets[:, :, 1], pos_weight=pos_weight)
        
        total_loss = (view_weight * view_loss) + (purchase_weight * purchase_loss)
        return total_loss
    
    def train_one_epoch(self, train_dataloader, val_dataloader,optimizer) -> float:
        self.model.train()
        # pos_weight = torch.tensor([15000.0]).to(self.device)
        total_loss = 0.0

        pbar = tqdm(train_dataloader, desc="Training", leave=False)
        for i, (ids, histories, labels) in enumerate(pbar):
            histories = histories.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()

            logits = self.model(histories)
            loss = self.calculate_loss(logits, labels)

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

            if i % 5000 == 0 and i > 0:
                # We use tqdm.write so the print doesn't break the progress bar
                tqdm.write(f"\n[Step {i}] Running mid-epoch validation...")
                val_loss, val_view_ndcg, val_purchase_ndcg = self.validate(val_dataloader)
                tqdm.write(f"[Step {i}] Val Loss: {val_loss:.4f} | Val View NDCG: {val_view_ndcg:.4f} | Val Purchase NDCG: {val_purchase_ndcg:.4f}")
                
                self.model.train()
        avg_loss = total_loss / len(train_dataloader)

        return avg_loss
    
    def train(self, train_dataloader, valid_dataloader):
        num_epoch = self.config.train['num_epoch']
        optimizer = self.optimizer
        best_ndcg = 0.0
        for epoch in range(num_epoch):
            train_loss = self.train_one_epoch(train_dataloader, valid_dataloader, optimizer)
            print(f"[Epoch {epoch+1}/{num_epoch}] Train loss = {train_loss:.8f}")
            val_loss, v_ndcg, p_ndcg = self.validate(valid_dataloader)

            is_best =  p_ndcg > best_ndcg
            if is_best :
                best_ndcg = p_ndcg
            self.save_checkpoint(self.config.data["output_path"], epoch, p_ndcg, is_best=is_best)

        print(f"[model1] End training.\nBest Validation Loss : {val_loss:.4f} \nBest View NDCG@10 : {v_ndcg:.4f}  \nBest Purchase NDCG@10 : {p_ndcg:.4f}")
    
    @torch.no_grad()
    def validate(self, dataloader) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        view_total_ndcg = 0.0
        purchase_total_ndcg = 0.0
        view_match_count = 0
        purchase_match_count = 0
        view_eligible_users = 0
        purchase_eligible_users = 0

        for ids, histories, labels in tqdm(dataloader, desc="validate", leave=False):
            histories = histories.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(histories)
            
            loss = self.calculate_loss(logits, labels)
            total_loss += loss.item()

            # Separating logits for the two tasks
            v_logits = logits[:, :, 0]
            p_logits = logits[:, :, 1]

            _, top_v = torch.topk(v_logits, k=10, dim=-1)
            _, top_p = torch.topk(p_logits, k=10, dim=-1)
            
            labels_np = labels.cpu().numpy()
            top_v = top_v.cpu().numpy()
            top_p = top_p.cpu().numpy()

            for i in range(len(labels_np)):
                # Helper logic for View
                view_trues = np.where(labels_np[i, :, 0] == 1)[0]
                if len(view_trues) > 0:
                    view_eligible_users += 1
                    ndcg = self._calculate_single_ndcg(top_v[i], view_trues)
                    view_total_ndcg += ndcg
                    if ndcg > 0: view_match_count += 1

                # Helper logic for Purchase
                purch_trues = np.where(labels_np[i, :, 1] == 1)[0]
                if len(purch_trues) > 0:
                    purchase_eligible_users += 1
                    ndcg = self._calculate_single_ndcg(top_p[i], purch_trues)
                    purchase_total_ndcg += ndcg
                    if ndcg > 0: purchase_match_count += 1

        avg_loss = total_loss / len(dataloader)
        view_avg_ndcg = view_total_ndcg / max(1, view_eligible_users)
        purchase_avg_ndcg = purchase_total_ndcg / max(1, purchase_eligible_users)
        print(f" >> Validation Finished. Total View Matches: {view_match_count}/{view_eligible_users} | Total Purchase Matches: {purchase_match_count}/{purchase_eligible_users}")
        return avg_loss, view_avg_ndcg, purchase_avg_ndcg
    
    def _calculate_single_ndcg(self, top_indices, true_indices):
        true_set = set(true_indices)
        dcg = 0.0
        for rank, idx in enumerate(top_indices):
            if idx in true_set:
                dcg += 1.0 / np.log2(rank + 2)
        
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(true_set), 10)))
        return dcg / idcg if idcg > 0 else 0.0
    
    @torch.no_grad()
    def infer(self, dataloader, output_name, k=10):
        self.model.eval()
        all_vrecommendations = {}
        all_precommendations = {}

        for ids, histories in tqdm(dataloader, desc="Inference"):
            histories = histories.to(self.device)
            logits = self.model(histories) # [batch_size, num_items, 2]
            v_logits = logits[:, :, 0]
            p_logits = logits[:, :, 1]
            
            # 가장 높은 확률을 가진 아이템 인덱스 K개 추출
            _, top_v = torch.topk(v_logits, k=10, dim=-1)
            _, top_p = torch.topk(p_logits, k=10, dim=-1)
            uids = ids.numpy() if torch.is_tensor(ids) else ids
            top_v = top_v.cpu().numpy()
            top_p = top_p.cpu().numpy()
            for batch_idx, uid in enumerate(uids):
                    # 인덱스를 실제 유저 ID와 아이템 ID로 변환
                    user_id = self.datamanager.idx2user[uid]
                    rec_vitems = [self.datamanager.idx2item[item_idx] for item_idx in top_v[batch_idx]]
                    rec_pitems = [self.datamanager.idx2item[item_idx] for item_idx in top_p[batch_idx]]
                    all_vrecommendations[user_id] = rec_vitems
                    all_precommendations[user_id] = rec_pitems

        flattened_vdata = []
        flattened_pdata = []
        # View
        for user_id, item_list in all_vrecommendations.items():
            for item_id in item_list:
                flattened_vdata.append([user_id, item_id])
        # Purchase
        for user_id, item_list in all_precommendations.items():
            for item_id in item_list:
                flattened_pdata.append([user_id, item_id])

        df_vres = pd.DataFrame(flattened_vdata, columns=['user_id', 'item_id'])
        df_pres = pd.DataFrame(flattened_pdata, columns=['user_id', 'item_id'])
        infer_path = os.path.join(self.config.data["output_path"], f"output({self.name})")
        v_path = os.path.join(infer_path, f"View_{output_name}.csv")
        p_path = os.path.join(infer_path, f"Purc_{output_name}.csv")
        df_vres.to_csv(v_path, index=True)
        df_pres.to_csv(p_path, index=True)

        print(f"[model2] Saved {len(df_vres)} recommendation rows to {v_path}")
        print(f"[model2] Saved {len(df_pres)} recommendation rows to {p_path}")
        return all_vrecommendations, all_precommendations

    def prepare_dataloader(self):
        train_dataloader, valid_dataloader, infer_dataloader = self.datamanager.prepare_dataloader()
        return train_dataloader, valid_dataloader, infer_dataloader
    
    def save_checkpoint(self, path, epoch, val_ndcg, is_best=False):
        # Create directory if it doesn't 
        checkpoint_dir = os.path.join(path, f'checkpoints({self.name})')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_ndcg': val_ndcg,
        }
        
        # Save the regular epoch checkpoint
        model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
        torch.save(checkpoint, model_path)
        
        # Save the "Best" version separately for easy access
        if is_best:
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
            print(f" [!] New Best Model Saved (NDCG: {val_ndcg:.4f})")

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f" [!] No checkpoint found at: {checkpoint_path}")
            return 0  # Return epoch 0 if nothing found
        
        print(f" [*] Loading checkpoint from {checkpoint_path}...")
        
        # map_location ensures it loads on the current device (CPU or GPU)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore the model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore the optimizer's "memory" (AdamW moments)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        val_ndcg = checkpoint.get('val_ndcg', 0.0)
        
        print(f" [+] Resuming from Epoch {start_epoch} (Last Val NDCG: {val_ndcg:.4f})")
        return start_epoch
    


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



import torch.nn as nn
import torch

class ItemEncoder(nn.Module):
    '''
    Data will come by history unit [item_idx, brand_idx, category_idx, price]
    
    return: item embedding
    '''
    def __init__(self,
                 num_items=29502,
                 num_brands=1860,
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

class PurchasePred(nn.Module):
    '''
    Data will come like 5 user purchase histoy data 
    [item_idx, brand_idx, category_idx, price, event_time(float, hours), event_type]
    [B, 6] -> [B,64 + 16] -> [B, 10]
    '''
    def __init__ (self, num_event_types = 3):
        super().__init__()
        self.item_encoder = ItemEncoder()
        self.event_type_emb = nn.Embedding(num_event_types, 8)
        self.time_proj = nn.Linear(1, 8)
        self.layer = nn.Linear(64 + 16, 10)
        # self.sigmoid = nn.Sigmoid()

    def forward (self, x):
        item_property = x[:, :4]
        event_type = x[:, 4].long()
        event_time = x[:, 5].float().unsqueeze(-1)
        item_e = self.item_encoder(item_property) # [B, 64]
        event_type_e = self.event_type_emb(event_type) # [B, 8]
        event_time_e = self.time_proj(event_time)
        emb_cat = torch.cat([item_e, event_type_e, event_time_e], dim=-1) # [B, 80]
        logits = self.layer(emb_cat)
        # output = self.sigmoid(logits)

        return logits



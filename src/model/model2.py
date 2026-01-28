import torch.nn as nn

class WeekModel(nn.Module):
    '''
    7 values per user data will come like
    [item_idx, brand, category_code, price, event_type, event_time(float, hour unit)]
    [B, 7, 6]  -> [B, 29502(items number)]
    '''
    def __init__(self, num_items = 638257, d_item = 64):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, d_item)
        self.item_emb_proj = nn.Linear(4, d_item)
        self.history_emb_proj = nn.Linear(6, d_item)
        self.encoder = nn.Sequential(  # input shape : [B,7, 64]
            nn.Linear
        )

    def forward(self, x): # x = [B, 7, 6]
        item_emb1 = x[:, :, :4]
        item_emb2 = self.item_emb_proj(item_emb1)
        item_e = self.item_emb(x[:, :, 0])
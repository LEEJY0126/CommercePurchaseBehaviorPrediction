import pandas as pd

path = "/root/workspace/CommercePurchaseBehaviorPrediction/output/recommendations.csv"

df_res = pd.read_csv(path)

print(df_res['item_id'].value_counts().head(10))
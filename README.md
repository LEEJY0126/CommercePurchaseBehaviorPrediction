# Commerce Purchase Behavior Prediction

A high-performance recommendation system powered by **Transformer architecture** to predict user purchase behavior based on large-scale e-commerce interaction logs (8M+ rows). This project focuses on solving extreme class imbalance and ranking rare purchase events using Multi-Task Learning.

---

## 🚀 Key Features

* **Transformer-based Sequential Modeling**: Captures complex temporal patterns in user browsing and purchase history.
* **Multi-Task Learning (MTL)**: Simultaneously optimizes for both `View` and `Purchase` events to leverage the "signal boost" from frequent view data.
* **Extreme Imbalance Handling**: Implements custom loss weighting (e.g., `purchase_weight=100`) to address the 4000:1 ratio between views and purchases.
* **Robust Experiment Tracking**: Automated metadata logging (`config.yaml`, `metadata.json`) and checkpoint management for reproducible AI research.

---

## 🏗 Architecture

The model utilizes a Transformer encoder to process event sequences, with specialized embedding layers for:
* **Item IDs** (Categorical)
* **Brands & Categories** (Metadata)
* **Event Types** (View vs. Purchase)
* **Temporal Features** (Event Time)



---

## 📊 Technical Insights & Strategy

### 1. Handling Sparsity
With only **2,000 purchases** out of **8,000,000 interactions**, the system uses **AdamW** with a specialized learning rate strategy to prevent the optimizer from "forgetting" rare purchase signals during long periods of non-purchase data.

### 2. Evaluation Metric: NDCG
We prioritize **NDCG (Normalized Discounted Cumulative Gain)** over simple Accuracy. This ensures that the items a user is most likely to buy appear at the top of the recommendation list.

---

## 🛠 Project Structure

```text
.
├── src/
│   ├── config/          # YAML configurations
│   ├── data/            # Data processing & Datamanager
│   ├── models/          # TransformerRecommender architecture
│   └── trainer/         # PurchasePred class with MTL logic
├── output/              # Inference results (CSV) & Metadata (JSON)
└── checkpoints/         # Model weights & Optimizer states
```
output and checkpoints directory will create after train one epocch

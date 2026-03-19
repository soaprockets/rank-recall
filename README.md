#### The unofficial implementation of paper "RankMixer: Scaling Up Ranking Models in Industrial Recommenders"
#### Stacked Target-to-History Cross Attention(STCA) module of paper "Make It Long, Keep It Fast: End-to-End 10k-Sequence Modeling at Billion Scale on Douyin"
#### The unofficial implementation of paper "OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender"
#### The unofficial implementation of paper "MPFormer: Adaptive Framework for Industrial Multi-Task Personalized Sequential Retriever"
#### The unofficial implementation of paper "TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders"

---

## ImRec Code Documentation

A recommendation framework based on implicit feedback, supporting 17+ SOTA models (BPR, NGCF, LightGCN, LayerGCN, etc.).

### Key Features
- **Auto Cleanup**: Automatically clears old logs, model weights, and preprocessed data before each run
- **Auto Save**: Automatically exports user/item embeddings to `./ImRec/inter_embs/` after training
- **Hyperparameter Limit**: Limits hyperparameter combinations to a maximum of 7 to avoid excessive search
- **Data Preparation**: Training data should be placed in the `./ImRec/data/` directory

### How to Run
```bash
cd ImRec
python main.py -m LayerGCN -d food
```

### Configuration
- Model config: `configs/model/*.yaml`
- Dataset config: `configs/dataset/*.yaml`
- Global config: `configs/overall.yaml`

---

## Data Loading and Processing Pipeline

### Overview

The framework implements a complete data pipeline from raw interaction files to model training:

```
Raw Data → Dataset Loading → Data Processing → Data Splitting → DataLoader → Training
```

### 1. Raw Data Format

The framework expects interaction data in `.csv/txt/...` files (or multiple files in a directory) with the following format:

```
user_id:token item_id:token rating:float timestamp:float
1 1 4.0 964982703
1 3 4.0 964981179
...
```

Required columns (configurable in `configs/dataset/*.yaml`):
- `user_id`: User identifier
- `item_id`: Item identifier
- `timestamp`: Interaction time (for chronological splitting)

Data files should be placed in `./ImRec/data/{dataset_name}/` directory.

### 2. Dataset Loading (`utils/dataset.py`)

The `RecDataset` class handles data loading and preprocessing:

#### Key Steps:

**a) Loading from File** (`_from_scratch()`)
- Reads all files from the data directory
- Uses pandas to load and concatenate multiple files
- Validates that required columns exist

**b) Data Processing** (`_data_processing()`)
- **Removes NA values** and duplicate rows
- **K-core filtering**: Iteratively filters users/items with insufficient interactions (configurable via `min_user_inter_num` and `min_item_inter_num`)
- **ID Remapping**: Maps user/item IDs to continuous integer indices starting from 0

**c) Caching**
- Preprocessed data and ID mappings are saved to `./ImRec/preprocessed_data/`
- Subsequent runs can load from cache for faster startup (controlled by `load_preprocessed` config)

### 3. Data Splitting (`dataset.split()`)

The framework splits data chronologically to prevent data leakage:

**Chronological Splitting Process:**
1. Sort all interactions by timestamp
2. Calculate split timestamps based on `split_ratio` (default: [0.9, 0.05, 0.05] for train/val/test)
3. Extract unique users and items from the training set
4. **Remap IDs** using only the training set's users/items
5. Filter out interactions from validation/test sets that involve unseen users/items
6. Split the data into three `RecDataset` instances
7. Save the processed data and ID mappings to disk

### 4. DataLoader (`utils/dataloader.py`)

The framework provides specialized DataLoaders for training and evaluation:

#### TrainDataLoader
- Handles batch generation for training
- Implements negative sampling strategies:
  - **Uniform Negative Sampling**: Randomly samples negative items that the user hasn't interacted with
  - **Full Sampling** (optional): Uses all non-interacted items as negatives
- Maintains user interaction history to avoid sampling positive items as negatives

#### EvalDataLoader
- Handles batch generation for evaluation
- Supports ranking over all items or a sampled subset

### 5. Pipeline Execution Flow

```python
# 1. Load and preprocess dataset
dataset = RecDataset(config)

# 2. Split into train/val/test with chronological ordering
train_ds, val_ds, test_ds = dataset.split(config['split_ratio'])

# 3. Create DataLoaders with negative sampling
train_loader = TrainDataLoader(config, train_ds, batch_size, shuffle=True)
val_loader = EvalDataLoader(config, val_ds, batch_size)
test_loader = EvalDataLoader(config, test_ds, batch_size)

# 4. Train model
trainer.fit(train_loader, val_data=val_loader, test_data=test_loader)
```

### Key Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data_path` | Directory containing raw data files | `'data/'` |
| `preprocessed_data` | Directory for cached preprocessed data | `'preprocessed_data/'` |
| `load_preprocessed` | Whether to load from cache | `True` |
| `split_ratio` | Train/val/test split ratios | `[0.9, 0.05, 0.05]` |
| `min_user_inter_num` | Minimum interactions per user | `1` |
| `min_item_inter_num` | Minimum interactions per item | `1` |
| `train_batch_size` | Training batch size | `2048` |
| `eval_batch_size` | Evaluation batch size | `2048` |
| `use_neg_sampling` | Whether to use negative sampling | `True` |

This code is forked from the [ImRec](https://github.com/enoche/ImRec.git) repository and modified for implicit feedback datasets.

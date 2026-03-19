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

This code is forked from the [ImRec](https://github.com/enoche/ImRec.git) repository and modified for implicit feedback datasets.

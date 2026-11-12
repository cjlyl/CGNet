# Few-shot Knowledge Graph Completion via Prototype Augmentation with Conditional Generation Network

This repository implements **CGNet** with **CIAN** as the backbone for few-shot knowledge graph completion. CGNet generates contextualized entity features from the support set and combines them with the original support features to construct a relation prototype. The implementation supports 1-, 3-, and 5-shot evaluation and reports MRR and Hits@1, Hits@5, and Hits@10.

## Methods

| Method | Description |
| --- | --- |
| CIAN | Original CIAN scoring with query-dependent support aggregation. |
| CIAN + S-CGNet | Trains CGNet on all entity pairs from eligible training relations. |
| CIAN + R-CGNet | Trains CGNet on up to `M` representative entity pairs per eligible training relation, selected by proximity to the relation mean. |
| CIAN + Gaussian / Mixup | Support augmentation controls using Gaussian perturbation or within-relation Mixup. |

## Installation

Use Python 3.10 or later. For GPU runs, install a PyTorch build compatible with your CUDA environment.

```bash
python -m pip install -r requirements.txt
```

## Data preparation

Place the following files in a dataset directory using the CIAN data format:

| Files | Contents |
| --- | --- |
| `train_tasks_in_train.json`, `dev_tasks.json`, `test_tasks.json` | Relation-indexed lists of `[head, relation, tail]` triples. |
| `ent2ids`, `relation2ids`, `path_graph` | Entity and relation ID mappings and the background graph. |
| `rel2candidates_in_train.json`, `e1rel_e2_in_train.json` | Candidate tails by relation and known positives for filtered ranking. |
| `entity2vec.TransE`, `relation2vec.TransE` | Pretrained TransE embeddings. |

For each development and test relation, the first `FEW` triples in file order form the support set, and the remaining triples serve as queries. Supply the dataset files and pretrained embeddings separately.

## Training and evaluation

Run the following commands from the repository root:

```bash
# Train CIAN and R-CGNet, then evaluate both on the test split.
FEW=5 DEVICE=cuda:0 bash run_cgnet.sh /path/to/NELL r

# Train S-CGNet using an existing CIAN checkpoint.
FEW=5 DEVICE=cuda:0 CIAN_CHECKPOINT=/path/to/cian_best \
  bash run_cgnet.sh /path/to/FB15K s
```

To evaluate saved checkpoints:

```bash
python main.py --test --device cuda:0 --datapath /path/to/NELL --few 5 \
  --checkpoint /path/to/cian_best --cg_checkpoint /path/to/cgnet.pt
```

Use the same `--cg_draws` and `--cg_temperature` values used during checkpoint selection. Omit `--cg_checkpoint` for the CIAN baseline. For the augmentation controls, omit it and pass `--support_aug gaussian` or `--support_aug mixup`. Use `--eval_split dev` to evaluate on the development split.

<!-- ## Citation

If you use this implementation, please cite *Few-shot Knowledge Graph Completion via Prototype Augmentation with Conditional Generation Network*. -->

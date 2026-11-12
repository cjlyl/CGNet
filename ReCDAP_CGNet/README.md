# Few-shot Knowledge Graph Completion via Prototype Augmentation with Conditional Generation Network

This repository implements **CGNet** with **ReCDAP** as the backbone for few-shot knowledge graph completion. CGNet generates contextualized entity features from the support set and combines them with the original support features to construct a relation prototype. The implementation supports 1-, 3-, and 5-shot evaluation and reports MRR and Hits@1, Hits@5, and Hits@10.

## Methods

| Method | Description |
| --- | --- |
| ReCDAP | Original conditional diffusion and candidate scoring. |
| ReCDAP + S-CGNet | Trains CGNet on all unique entity pairs from eligible training relations. |
| ReCDAP + R-CGNet | Trains CGNet on up to `M` representative entity pairs per eligible training relation, selected by distance to the relation mean. |
| ReCDAP + C7 | Runs the supplied residual adapter through the separate C7 entry point. |

## Installation

Use Python 3.11 on Linux. To install the CPU dependencies:

```bash
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For GPU runs, install CUDA-compatible builds of both PyTorch and DGL and set `DEVICE=cuda:0`.

## Data preparation

Place the following files in a dataset directory using the ReCDAP data format:

| Files | Contents |
| --- | --- |
| `train_tasks_in_train.json`, `dev_tasks.json`, `test_tasks.json` | Relation-indexed lists of `[head, relation, tail]` triples. |
| `ent2ids`, `relation2ids` (or `rel2ids`), `path_graph` | Entity and relation ID mappings and the background graph. Include the inverse relation IDs used by the graph. |
| `rel2candidates_in_train.json`, `e1rel_e2_in_train.json` | Candidate tails by relation and known positive tails for filtered ranking. |
| `entity2vec.TransE`, `relation2vec.TransE` | Pretrained TransE embedding matrices in ID order. |

For each development and test relation, the first `FEW` triples in file order form the support set, and the remaining triples serve as queries. Supply the dataset files, pretrained embeddings, and checkpoints separately.

## Training and evaluation

Run the following commands from the repository root:

```bash
# Train the original ReCDAP backbone if a checkpoint is unavailable.
python main.py --step train --dataset NELL-One --data_path /path/to/NELL \
  --few 5 --device cpu --prefix recdap

# Train R-CGNet and evaluate it alongside frozen ReCDAP.
DATASET=NELL-One FEW=5 DEVICE=cpu \
  bash run_cgnet.sh /path/to/NELL /path/to/recdap/state_dict r

# Train S-CGNet with an existing FB15K-237 ReCDAP checkpoint.
DATASET=FB15k-237 FEW=5 DEVICE=cpu \
  bash run_cgnet.sh /path/to/FB15K /path/to/recdap/state_dict s
```

The backbone training command writes `state/<run>/state_dict`. Use a full original ReCDAP checkpoint with the matching dataset, ID ordering, shot count, embedding dimensions, LSTM configuration, GNN hop count, and diffusion schedule. For 1-shot or 3-shot evaluation, set `FEW=1` or `FEW=3` and use a checkpoint trained for that setting.

The runner selects the generator with the highest development MRR and evaluates it alongside frozen ReCDAP on the same test candidates. It writes the checkpoint and log to a unique directory under `runs/`.

To evaluate saved checkpoints:

```bash
python main_cgnet.py --step test --device cpu --dataset NELL-One \
  --data_path /path/to/NELL --few 5 \
  --baseline_checkpoint /path/to/recdap/state_dict \
  --cg_checkpoint /path/to/run/cgnet.pt
```

## Acknowledgments and Citation

The ReCDAP backbone in this repository is adapted from the
[original ReCDAP implementation](https://github.com/hou27/ReCDAP-FKGC).

If you use this code in your research, please cite both papers:

- **CGNet**. *Few-shot Knowledge Graph Completion via Prototype Augmentation with Conditional Generation Network*.
- **ReCDAP**. Jeongho Kim, Chanyeong Heo, and Jaehee Jung. 2025.
  *ReCDAP: Relation-Based Conditional Diffusion with Attention Pooling for Few-Shot Knowledge Graph Completion*.
  ACM SIGIR 2025. [DOI](https://doi.org/10.1145/3726302.3730241).
<!-- ## Citation

If you use this implementation, please cite *Few-shot Knowledge Graph Completion via Prototype Augmentation with Conditional Generation Network*.
-->

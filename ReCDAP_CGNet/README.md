# Few-shot Knowledge Graph Completion via Prototype Augmentation with Conditional Generation Network

This repository implements **CGNet** with **ReCDAP** as the backbone for few-shot knowledge graph completion. CGNet uses frozen ReCDAP graph features to generate synthetic support features and combine them with the original features into a relation prototype. The implementation supports 1-, 3-, and 5-shot evaluation and reports MRR, Hits@1, Hits@5, and Hits@10.

## Methods

| Method | Description |
| --- | --- |
| ReCDAP | Frozen backbone evaluated with its original conditional diffusion and scoring pipeline. |
| ReCDAP + S-CGNet | Trains CGNet on all unique entity pairs from eligible training relations. |
| ReCDAP + R-CGNet | Trains CGNet on up to `M` representative entity pairs per eligible training relation, selected by distance to the relation mean. |

## Installation

Use Python 3.11 on Linux. Install the CPU dependencies with:

```bash
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For GPU runs, install CUDA-compatible builds of both PyTorch and DGL and set `DEVICE=cuda:0`.

## Data preparation

Place the following files in a dataset directory using the ReCDAP data format.

| Files | Contents |
| --- | --- |
| `train_tasks_in_train.json`, `dev_tasks.json`, `test_tasks.json` | Relation-indexed lists of `[head, relation, tail]` triples. |
| `ent2ids`, `relation2ids` (or `rel2ids`), `path_graph` | Entity and relation ID mappings and the background graph. Include inverse relation IDs for the graph edges. |
| `rel2candidates_in_train.json`, `e1rel_e2_in_train.json` | Candidate tails by relation and known positive tails for filtered ranking. |
| `entity2vec.TransE`, `relation2vec.TransE` | Pretrained TransE embedding matrices in ID order. |

For each development and test relation, the first `FEW` triples in file order form the support set. The remaining triples serve as queries. Supply the dataset files and pretrained embeddings separately.

## Training and evaluation

Run the following commands from the repository root.

### Reusing a pretrained backbone checkpoint

The launcher requires the **full `state_dict` from the original ReCDAP model**, including the encoder, diffusion network, and scoring modules. Pass the checkpoint path as the second argument to `run_cgnet.sh`. If you do not have a compatible checkpoint, train ReCDAP using the original implementation and save its full model weights. This package uses those weights to freeze the backbone, train CGNet, and evaluate the original ReCDAP baseline on the same candidates.

Use a checkpoint trained with the same dataset, entity and relation ID ordering, shot count, embedding configuration, LSTM dimensions and layers, GNN hop count, and diffusion settings. Supply any nondefault settings as extra arguments to the launcher and the standalone evaluation command.

```bash
# Train R-CGNet using a pretrained 5-shot NELL-One ReCDAP model.
DATASET=NELL-One FEW=5 DEVICE=cpu \
  bash run_cgnet.sh /path/to/NELL /path/to/nell_recdap/state_dict r

# Train S-CGNet using a pretrained 5-shot FB15K-237 ReCDAP model.
DATASET=FB15k-237 FEW=5 DEVICE=cpu \
  bash run_cgnet.sh /path/to/FB15K /path/to/fb15k_recdap/state_dict s
```

Use `r` for R-CGNet or `s` for S-CGNet. Set `FEW=1`, `FEW=3`, or `FEW=5` and use the corresponding ReCDAP checkpoint. The launcher selects the generator with the highest development MRR, reloads it for test evaluation, and writes `cgnet.pt` and `run.log` to a unique directory under `runs/`.

### Evaluating saved checkpoints

```bash
python main_cgnet.py --step test --device cpu --dataset NELL-One \
  --data_path /path/to/NELL --few 5 \
  --baseline_checkpoint /path/to/nell_recdap/state_dict \
  --cg_checkpoint /path/to/run/cgnet.pt
```

Use `--step dev` for development evaluation. The CGNet checkpoint restores the scoring mode and prototype evaluation settings used during selection. Use the same ReCDAP checkpoint and architecture settings as during CGNet training.

## Acknowledgments and Citation

The ReCDAP backbone is adapted from the [original ReCDAP implementation](https://github.com/hou27/ReCDAP-FKGC).

If you use this code in your research, please cite both papers.

- **CGNet.** *Few-shot Knowledge Graph Completion via Prototype Augmentation with Conditional Generation Network*.
- **ReCDAP.** Jeongho Kim, Chanyeong Heo, and Jaehee Jung. 2025. *ReCDAP: Relation-Based Conditional Diffusion with Attention Pooling for Few-Shot Knowledge Graph Completion*. ACM SIGIR 2025. [DOI](https://doi.org/10.1145/3726302.3730241).

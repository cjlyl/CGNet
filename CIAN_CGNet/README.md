# Few-shot Knowledge Graph Completion via Prototype Augmentation with Conditional Generation Network

This repository implements **CGNet** with **CIAN** as the backbone for few-shot knowledge graph completion. CGNet generates contextualized entity features from the support set and combines them with the original support features to construct a relation prototype. The implementation supports 1-, 3-, and 5-shot evaluation and reports MRR, Hits@1, Hits@5, and Hits@10.

## Methods

| Method | Description |
| --- | --- |
| CIAN | Original CIAN scoring with query-dependent support aggregation. |
| CIAN + S-CGNet | Trains CGNet on all unique entity pairs from eligible training relations. |
| CIAN + R-CGNet | Trains CGNet on up to `M` representative entity pairs per eligible training relation, selected by distance to the relation mean. |
| CIAN + Gaussian / Mixup | Support augmentation controls using Gaussian perturbation or within-relation Mixup. |

## Installation

Use Python 3.10 or later. For GPU runs, install a PyTorch build compatible with your CUDA environment.

```bash
python -m pip install -r requirements.txt
```

## Data preparation

Place the following files in a dataset directory using the CIAN data format.

| Files | Contents |
| --- | --- |
| `train_tasks_in_train.json`, `dev_tasks.json`, `test_tasks.json` | Relation-indexed lists of `[head, relation, tail]` triples. |
| `ent2ids`, `relation2ids`, `path_graph` | Entity and relation ID mappings and the background graph. |
| `rel2candidates_in_train.json`, `e1rel_e2_in_train.json` | Candidate tails by relation and known positives for filtered ranking. |
| `entity2vec.TransE`, `relation2vec.TransE` | Pretrained TransE embeddings. |

For each development and test relation, the first `FEW` triples in file order form the support set. The remaining triples serve as queries. Supply the dataset files and pretrained embeddings separately, including when reusing a checkpoint.

## Training and evaluation

Run the commands from the repository root. Use `r` for R-CGNet or `s` for S-CGNet. Set `FEW=1`, `FEW=3`, or `FEW=5` for the desired shot setting, and use `DEVICE=cpu` for CPU runs.


### Reusing a pretrained backbone checkpoint

To skip CIAN training, set `CIAN_CHECKPOINT` to the path of a **full CIAN model `state_dict`**, such as `runs/cian_r_<run>/models/backbone_best` from a previous run. Use weights trained with the same dataset, entity and relation ID ordering, shot count, embedding configuration, and neighbor settings. The launcher loads and freezes the CIAN model before training CGNet.

```bash
# Train R-CGNet with a pretrained 5-shot NELL CIAN model.
FEW=5 DEVICE=cuda:0 CIAN_CHECKPOINT=/path/to/nell_cian_best \
  bash run_cgnet.sh /path/to/NELL r

# Train S-CGNet with a pretrained 5-shot FB15K CIAN model.
FEW=5 DEVICE=cuda:0 CIAN_CHECKPOINT=/path/to/fb15k_cian_best \
  bash run_cgnet.sh /path/to/FB15K s
```

Both workflows evaluate the baseline and CGNet with the same CIAN weights. The launcher selects the CGNet generator by development MRR and writes `cgnet.pt` and the logs to a separate directory under `runs/`.

### Evaluating saved checkpoints

```bash
python main.py --test --device cuda:0 --datapath /path/to/NELL --few 5 --seed 2022 \
  --checkpoint /path/to/nell_cian_best --cg_checkpoint /path/to/cgnet.pt
```

Use the same `--seed`, `--cg_draws`, and `--cg_temperature` values used during checkpoint selection. The launcher defaults are `2022`, `20`, and `1.0`, respectively. Also match any nondefault embedding or neighbor settings.

Omit `--cg_checkpoint` to evaluate the original CIAN baseline. For the augmentation controls, omit it and pass `--support_aug gaussian` or `--support_aug mixup`. Within-relation Mixup requires at least two support pairs. Use `--eval_split dev` for development evaluation.


## Acknowledgments and Citation

The CIAN backbone is adapted from the [original CIAN implementation](https://github.com/cjlyl/FKGC-CIAN).

If you use this code in your research, please cite both papers.

- **CGNet.** *Few-shot Knowledge Graph Completion via Prototype Augmentation with Conditional Generation Network*.
- **CIAN.** Yuling Li, Kui Yu, Xiaoling Huang, and Yuhong Zhang. 2022. *Learning Inter-Entity-Interaction for Few-Shot Knowledge Graph Completion*. EMNLP 2022. [ACL Anthology](https://aclanthology.org/2022.emnlp-main.524/).

# Few-shot Knowledge Graph Completion via Prototype Augmentation with Conditional Generation Network

CGNet trains a conditional generator on frozen backbone features and augments few-shot relation prototypes. CIAN and ReCDAP use the same S-CGNet and R-CGNet training and inference code.

## Quick start

Use Python 3.11. Install the common dependencies and run the synthetic CPU demo.

```bash
python -m pip install -r requirements.txt
python core.py
python example_adapter.py
```

For ReCDAP on Linux CPU, install PyTorch 2.2.2 from its CPU wheel index and then install `requirements-recdap.txt`. For CUDA, use compatible PyTorch and DGL builds.

## Use a pretrained backbone

Supply the full original backbone `state_dict` for the selected dataset, shot count, and architecture. CIAN uses contextualized head and tail vectors before its entity-pair MLP. ReCDAP uses the head and tail vectors from its relational path GNN. The loader restores the full state dict and freezes the backbone.

The data directory must contain `train_tasks_in_train.json`, `dev_tasks.json`, `test_tasks.json`, `rel2candidates_in_train.json`, `e1rel_e2_in_train.json`, `ent2ids`, `relation2ids` (or `rel2ids`), and `path_graph`. ReCDAP additionally needs `entity2vec.TransE` and `relation2vec.TransE`. Task files map relations to lists of `[head, relation, tail]` triples. Keep the original entity and relation dictionary ordering, graph, and checkpoint configuration.

```bash
python run.py --backbone cian --data /path/to/NELL --few 5 \
  --backbone-checkpoint /path/to/cian_best --checkpoint runs/cian_cgnet.pt --variant r

python run.py --backbone recdap --data /path/to/NELL --few 5 \
  --backbone-checkpoint /path/to/recdap/state_dict --checkpoint runs/recdap_cgnet.pt --variant s
```

Training selects the best generator on development MRR and then reports filtered test MRR and Hits@1/5/10. To evaluate saved weights, add `--mode test` or `--mode dev` and supply the same backbone, data, and architecture settings. Run `python run.py --help` for settings such as CIAN's `--max-neighbor` and ReCDAP's `--lstm-hiddendim`, `--lstm-layers`, `--hop`, and `--num-diffusion-iters`.

## Add another backbone

Wrap a loaded backbone with `FrozenEncoder(backbone, encode_fn)`. Implement `encode_fn(backbone, triples)` to return a tensor of shape `[N, 2, d]` in head/tail order. Use that encoder for training positives and tail corruptions, and for support and candidate pairs at inference. Pass the per-relation training tensors as `RelationFeatures` to `CGNet.fit`. Set `top_m=None` for S-CGNet. Set `top_m=M` for R-CGNet's per-relation selection.

This package precomputes one filtered tail corruption per distinct positive training triple. Evaluation ranks the supplied filtered relation candidates with the CGNet prototype inner product. The original backbone scoring pipelines and benchmark replication scripts live in their respective projects. At the default 20 draws, each support pair contributes 60 synthetic rows and one original row.

## Acknowledgments and Citation

The backbone code is adapted from the [original CIAN implementation](https://github.com/cjlyl/FKGC-CIAN) and the [original ReCDAP implementation](https://github.com/hou27/ReCDAP-FKGC). If you use this code in your research, cite the CGNet paper and the corresponding backbone paper.

- Yuling Li, Kui Yu, Xiaoling Huang, and Yuhong Zhang. *Learning Inter-Entity-Interaction for Few-Shot Knowledge Graph Completion*. EMNLP 2022. [ACL Anthology](https://aclanthology.org/2022.emnlp-main.524/).
- Jeongho Kim, Chanyeong Heo, and Jaehee Jung. *ReCDAP: Relation-Based Conditional Diffusion with Attention Pooling for Few-Shot Knowledge Graph Completion*. SIGIR 2025. [DOI](https://doi.org/10.1145/3726302.3730241).

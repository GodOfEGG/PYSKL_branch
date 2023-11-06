# DG-STGCN

## Abstract

Graph convolution networks (GCN) have been widely used in skeleton-based action recognition. We note that existing GCN-based approaches primarily rely on prescribed graphical structures (i.e., a manually defined topology of skeleton joints), which limits their flexibility to capture complicated correlations between joints. To move beyond this limitation, we propose a new framework for skeleton-based action recognition, namely Dynamic Group Spatio-Temporal GCN (DG-STGCN). It consists of two modules, DG-GCN and DG-TCN, respectively, for spatial and temporal modeling. In particular, DG-GCN uses learned affinity matrices to capture dynamic graphical structures instead of relying on a prescribed one, while DG-TCN performs group-wise temporal convolutions with varying receptive fields and incorporates a dynamic joint-skeleton fusion module for adaptive multi-level temporal modeling. On a wide range of benchmarks, including NTURGB+D, Kinetics-Skeleton, BABEL, and Toyota SmartHome, DG-STGCN consistently outperforms state-of-the-art methods, often by a notable margin.

## Citation

```BibTeX
@article{duan2022dg,
  title={DG-STGCN: Dynamic Spatial-Temporal Modeling for Skeleton-based Action Recognition},
  author={Duan, Haodong and Wang, Jiaqi and Chen, Kai and Lin, Dahua},
  journal={arXiv preprint arXiv:2210.05895},
  year={2022}
}
```



## Training & Testing

You can use the following command to train a model.

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# For example: train DG-STGCN on AIST++(SMPL format, Segment Length 60 frames) with 1 GPUs, with validation, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/dgstgcn/aist++_smpl_60/j.py 1 --validate --test-last --test-best
```

You can use the following command to test a model.

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
# For example: test DG-STGCN on AIST++(SMPL format, Segment Length 60 frames) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/dgstgcn/aist++_smpl_60/j.py checkpoints/SOME_CHECKPOINT.pth 1 --eval top_k_accuracy --out result.pkl
```

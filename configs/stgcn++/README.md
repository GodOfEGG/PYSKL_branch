# STGCN++

## Introduction

STGCN++ is a variant of STGCN developed in PYSKL with some modifications in the architecture of the spatial module and the temporal module. The architecture of STGCN++ is described in PYSKL [tech report](https://arxiv.org/abs/2205.09443).

## Citation

```BibTeX
@inproceedings{duan2022pyskl,
  title={Pyskl: Towards good practices for skeleton action recognition},
  author={Duan, Haodong and Wang, Jiaqi and Chen, Kai and Lin, Dahua},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={7351--7354},
  year={2022}
}
```


## Training & Testing

You can use the following command to train a model.

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# For example: train STGCN++ on AIST++(SMPL format, Segment Length 60 frames) with 1 GPUs, with validation, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/stgcn++/stgcn++_aist++_smpl_60/j.py 1 --validate --test-last --test-best
```

You can use the following command to test a model.

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
# For example: test STGCN++ on AIST++(SMPL format, Segment Length 60 frames) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/stgcn++/stgcn++_aist++_smpl_60/j.py checkpoints/SOME_CHECKPOINT.pth 1 --eval top_k_accuracy --out result.pkl
```

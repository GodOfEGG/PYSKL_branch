# STGCN

## Introduction

STGCN is one of the first algorithms that adopt Graph Convolution Neural Networks for skeleton processing. We provide STGCN trained on NTURGB+D with 2D skeletons (HRNet) and 3D skeletons in both the original training setting and the **PYSKL** training setting. We provide checkpoints for four modalities: Joint, Bone, Joint Motion, and Bone Motion. The accuracy of each modality links to the weight file.

## Citation

```BibTeX
@inproceedings{yan2018spatial,
  title={Spatial temporal graph convolutional networks for skeleton-based action recognition},
  author={Yan, Sijie and Xiong, Yuanjun and Lin, Dahua},
  booktitle={Thirty-second AAAI conference on artificial intelligence},
  year={2018}
}
# If you use the STGCN with PYSKL practices in your work
@misc{duan2022pyskl,
    title={PYSKL: a toolbox for skeleton-based video understanding},
    author={PYSKL Contributors},
    howpublished = {\url{https://github.com/kennymckormick/pyskl}},
    year={2022}
}
```


## Training & Testing

You can use the following command to train a model.

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# For example: train STGCN on AIST++(SMPL format, Segment Length 60 frames) with 1 GPUs, with validation, with PYSKL practice, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/stgcn/stgcn_aist++_smpl_60/j.py 1 --validate --test-last --test-best
```

You can use the following command to test a model.

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
# For example: test STGCN on AIST++(SMPL format, Segment Length 60 frames) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/stgcn/stgcn_aist++_smpl_60/j.py checkpoints/SOME_CHECKPOINT.pth 1 --eval top_k_accuracy --out result.pkl
```

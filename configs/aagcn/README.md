# AAGCN

## Abstract

Graph convolutional networks (GCNs), which generalize CNNs to more generic non-Euclidean structures, have achieved remarkable performance for skeleton-based action recognition. However, there still exist several issues in the previous GCN-based models. First, the topology of the graph is set heuristically and fixed over all the model layers and input data. This may not be suitable for the hierarchy of the GCN model and the diversity of the data in action recognition tasks. Second, the second-order information of the skeleton data, i.e., the length and orientation of the bones, is rarely investigated, which is naturally more informative and discriminative for the human action recognition. In this work, we propose a novel multi-stream attention-enhanced adaptive graph convolutional neural network (MS-AAGCN) for skeleton-based action recognition. The graph topology in our model can be either uniformly or individually learned based on the input data in an end-to-end manner. This data-driven approach increases the flexibility of the model for graph construction and brings more generality to adapt to various data samples. Besides, the proposed adaptive graph convolutional layer is further enhanced by a spatial-temporal-channel attention module, which helps the model pay more attention to important joints, frames and features. Moreover, the information of both the joints and bones, together with their motion information, are simultaneously modeled in a multi-stream framework, which shows notable improvement for the recognition accuracy. 

## Citation

```BibTeX
@article{shi2020skeleton,
  title={Skeleton-based action recognition with multi-stream adaptive graph convolutional networks},
  author={Shi, Lei and Zhang, Yifan and Cheng, Jian and Lu, Hanqing},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={9532--9545},
  year={2020},
  publisher={IEEE}
}
```

## Training & Testing

You can use the following command to train a model.

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# For example: train AAGCN on AIST++(SMPL format, Segment Length 60 frames) with 1 GPUs, with validation, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/aagcn/aagcn_aist++_smpl_60/j.py 1 --validate --test-last --test-best
```

You can use the following command to test a model.

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
# For example: test AAGCN on AIST++(SMPL format, Segment Length 60 frames) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/aagcn/aagcn_aist++_smpl_60/j.py checkpoints/SOME_CHECKPOINT.pth 1 --eval top_k_accuracy --out result.pkl
```

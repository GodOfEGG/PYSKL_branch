# CTRGCN

## Abstract

Graph convolutional networks (GCNs) have been widely used and achieved remarkable results in skeleton-based action recognition. In GCNs, graph topology dominates feature aggregation and therefore is the key to extracting representative features. In this work, we propose a novel Channel-wise Topology Refinement Graph Convolution (CTR-GC) to dynamically learn different topologies and effectively aggregate joint features in different channels for skeleton-based action recognition. The proposed CTR-GC models channel-wise topologies through learning a shared topology as a generic prior for all channels and refining it with channel-specific correlations for each channel. Our refinement method introduces few extra parameters and significantly reduces the difficulty of modeling channel-wise topologies. Furthermore, via reformulating graph convolutions into a unified form, we find that CTR-GC relaxes strict constraints of graph convolutions, leading to stronger representation capability. Combining CTR-GC with temporal modeling modules, we develop a powerful graph convolutional network named CTR-GCN which notably outperforms state-of-the-art methods on the NTU RGB+D, NTU RGB+D 120, and NW-UCLA datasets.

## Citation

```BibTeX
@inproceedings{chen2021channel,
  title={Channel-wise topology refinement graph convolution for skeleton-based action recognition},
  author={Chen, Yuxin and Zhang, Ziqi and Yuan, Chunfeng and Li, Bing and Deng, Ying and Hu, Weiming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13359--13368},
  year={2021}
}
```


## Training & Testing

You can use the following command to train a model.

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# For example: train CTRGCN on AIST++(SMPL format, Segment Length 60 frames) with 1 GPUs, with validation, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/ctrgcn/ctrgcn_aist++_smpl_60/j.py 1 --validate --test-last --test-best
```

You can use the following command to test a model.

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
# For example: test CTRGCN on AIST++(SMPL format, Segment Length 60 frames) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/ctrgcn/ctrgcn_aist++_smpl_60/j.py checkpoints/SOME_CHECKPOINT.pth 1 --eval top_k_accuracy --out result.pkl
```

# PYSKL on AIST++ Dataset

This Repo is a copy of [PYSKL](https://github.com/kennymckormick/pyskl/tree/main) with the modification below:

1. In pyskl/datasets/pipelines/pose_related.py, we add SMPL format to class JointToBone.
2. In pyskl/utils/graph.py, we add SMPL format to class Graph.
3. In pyskl/datasets/pipelines/sampling.py, we add the class [ContinuousSampleFrames, ContinuousSample, ContinuousSampleDecode] and [RandomSampleFrames, RandomSample, RandomSampleDecode].


## Installation
```shell
git clone https://github.com/GodOfEGG/pyskl-on-aist-plus.git
cd pyskl
# This command runs well with conda 22.9.0, if you are running an early conda version and got some errors, try to update your conda first
conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .
```


## Data Preparation

1. Download the annotations from [AIST++ website](https://google.github.io/aistplusplus_dataset/factsfigures.html).
2. Use the following commands to segment the data by bar or by time. The configuration files are in INI format. Some example configuration files are 
in ./configs/segmentation
```shell
# segment by bars
python data_segmentation/AIST2PYSKL_segment_by_bar.py configs/segmentation/{config_name}
# segment by time
python data_segmentation/AIST2PYSKL_segment_by_time.py configs/segmentation/{config_name}
```

## Training & Testing

You can use following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.
```shell
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# Testing
bash tools/dist_test.sh {config_name} {checkpoint} {num_gpus} --out {output_file} --eval top_k_accuracy mean_class_accuracy
```
For specific examples, please go to the README for each specific algorithm we supported.

## Citation


```BibTeX

```




model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='smpl', mode='spatial'),
        num_person=1,
        base_channels=64,
        ch_ratio=2,
        num_stages=10),
    cls_head=dict(type='GCNHead', num_classes=10, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = 'data/aist++/aist++_smpl_240.pkl'
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='RandomScale', scale=0.1),
    dict(type='RandomRot', theta=0.3),
    dict(type='GenSkeFeat', dataset='smpl', feats=['j']),
    dict(type='UniformSample', clip_len=200),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='smpl', feats=['j']),
    dict(type='UniformSample', clip_len=200, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='smpl', feats=['j']),
    dict(type='UniformSample', clip_len=200, num_clips=5),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='test'))

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0003, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.0001, by_epoch=False)
total_epochs = 50
checkpoint_config = dict(interval=10)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/stgcn++/stgcn++_aist++_smpl_240/j_aug2-12'

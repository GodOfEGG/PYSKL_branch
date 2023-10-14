model = dict(
    type='RecognizerGCN',
    backbone=dict(
		type='MSG3D',
        graph_cfg=dict(layout='smpl', mode='binary_adj'),
        num_person=1,
        base_channels=120,
        num_gcn_scales=10,
        num_g3d_scales=5,
        tcn_dropout=0),
    cls_head=dict(type='GCNHead', num_classes=10, in_channels=480))

dataset_type = 'PoseDataset'
ann_file = 'data/aist++/aist++_smpl_240.pkl'
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='RandomScale', scale=0.1),
    dict(type='RandomRot', theta=0.3),
    dict(type='GenSkeFeat', dataset='smpl', feats=['j']),
    dict(type='UniformSample', clip_len=210),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='smpl', feats=['j']),
    dict(type='UniformSample', clip_len=210, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='smpl', feats=['j']),
    dict(type='UniformSample', clip_len=210, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='test'))

# optimizer
optimizer = dict(type='SGD', lr=0.04, momentum=0.6, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.0006, by_epoch=True)
total_epochs = 50
checkpoint_config = dict(interval=-1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/msg3d/msg3d_aist++_smpl_240/j_aug10'

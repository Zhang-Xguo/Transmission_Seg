_base_ = ["PTv3_gridnethd_7class_stage_b_las_new_half.py"]

# Continue from the best public seven-class SpUNet checkpoint.  Keep the
# existing block-disjoint Stage B training and validation lists unchanged.
weight = "exp/gridnethd/spunet_7class_public_stage_a/model/model_best.pth"
resume = False
save_path = "exp/gridnethd/spunet_7class_stage_b_las_new_blocks"
seed = 42

# Physical GPUs 4, 5, 6: 16 cropped tiles and two loader workers per GPU.
batch_size = 48
num_worker = 6
empty_cache = False
enable_amp = True
epoch = 30
eval_epoch = 3

optimizer = dict(type="AdamW", lr=0.0002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR", max_lr=0.0002, pct_start=0.1,
    anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0,
)
param_dicts = None

model = dict(
    _delete_=True,
    type="DefaultSegmentor",
    backbone=dict(
        type="SpUNet-v1m1", in_channels=3, num_classes=7,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    criteria=[
        dict(type="PointWeightedCrossEntropyLoss", loss_weight=1.0,
             ignore_index=255, weight=[2.0, 1.3, 3.0, 0.8, 1.0, 1.0, 1.0]),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=0.5,
             ignore_index=255),
    ],
)

# The inherited precise-test hook expects a separate full-scene split and
# would make the training job depend on it.  Validate on the fixed tile set
# every three effective passes and save the best mIoU checkpoint.
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None),
]

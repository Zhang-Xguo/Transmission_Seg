_base_ = ["PTv3_gridnethd_7class_public_stage_a_3gpu_capped.py"]

# Stage A uses exactly the existing public 7-class train_final / val_final
# split.  100 effective passes are grouped into 10 validation/checkpoint
# cycles, matching the earlier PTv3 Stage A schedule.
weight = "checkpoints/spunet/scannet20_rgb3_7class_random_head.pth"
resume = False
save_path = "exp/gridnethd/spunet_7class_public_stage_a"
seed = 42

batch_size = 48           # 16 samples per GPU on physical GPU 4, 5, 6
num_worker = 6            # 2 loader workers per GPU; cap CPU load
empty_cache = False
enable_amp = True
epoch = 100
eval_epoch = 10

# All SpUNet layers are tuned, including the random seven-class classifier.
optimizer = dict(type="AdamW", lr=0.0006, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR", max_lr=0.0006, pct_start=0.04,
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
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=255),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0,
             ignore_index=255),
    ],
)

# The inherited PreciseEvaluator uses a nonexistent pc7/test_final split.
# Keep the same regular validation/checkpoint cycle and run a separate
# full-resolution evaluation of the best checkpoint after training.
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None),
]

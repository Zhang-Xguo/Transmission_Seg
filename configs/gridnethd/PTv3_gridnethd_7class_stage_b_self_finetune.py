_base_ = ["PTv3_gridnethd_7class_public_stage_a_3gpu_capped.py"]

# Stage B continues the completed direct-seven-class Stage A model.  This is
# not an 11->7 head conversion and not a resume of Stage A's optimizer state.
weight = (
    "exp/gridnethd/ptv3_7class_public_stage_a_3gpu_capped/"
    "model/model_best.pth"
)
resume = False
save_path = "exp/gridnethd/ptv3_7class_stage_b_self_finetune"

# Three GPUs (CUDA_VISIBLE_DEVICES=4,5,6); GPU 7 remains free.  Loader workers
# are intentionally capped to avoid the excessive CPU use seen previously.
batch_size = 6
num_worker = 6
empty_cache = True
epoch = 30
eval_epoch = 3

# Conservative adaptation: the classifier/decoder changes faster than PTv3
# blocks, reducing catastrophic forgetting of the public-data initialization.
optimizer = dict(type="AdamW", lr=0.0002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR", max_lr=[0.0002, 0.00002], pct_start=0.1,
    anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.00002)]

model = dict(
    criteria=[
        dict(type="PointWeightedCrossEntropyLoss", loss_weight=1.0,
             ignore_index=255, weight=[2.0, 1.3, 3.0, 0.8, 1.0, 1.0, 1.0]),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=0.5,
             ignore_index=255),
    ]
)

public_root = "data/gridnethd/pc7"
self_root = "data/ours_stage_b_7class_v1"

train_transform = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="RandomColorJitter", brightness=0.2, contrast=0.2,
         saturation=0.2, hue=0.05, p=0.8),
    dict(type="RandomColorDrop", p=0.15, color_augment=0.0),
    dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train",
         return_grid_coord=True),
    dict(type="SphereCrop", point_max=150000, mode="random"),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment",
                               "supervision_weight"), feat_keys=("color",)),
]

val_transform = [
    dict(type="Copy", keys_dict={"segment": "origin_segment"}),
    dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train",
         return_grid_coord=True, return_inverse=True),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment",
                               "origin_segment", "inverse"), feat_keys=("color",)),
]

data = dict(
    train=dict(
        _delete_=True, type="ConcatDataset",
        datasets=[
            # Frozen public replay subset balances the custom set repeated ten
            # times without another full Stage A.
            dict(type="Gridnethd", split="train_stage_b_replay.json", data_root=public_root,
                 transform=train_transform, test_mode=False, ignore_index=255,
                 default_supervision_weight=1.0, loop=1),
            # T7-T8 is excluded as a spatial buffer; repetition remains 1:1.
            dict(type="Gridnethd", split="train_self.json", data_root=self_root,
                 transform=train_transform, test_mode=False, ignore_index=255,
                 default_supervision_weight=1.0, loop=10),
        ], loop=1,
    ),
    val=dict(data_root=self_root, split="val_fixed.json", transform=val_transform),
    test=dict(data_root=self_root, split="val_fixed.json"),
)

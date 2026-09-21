_base_ = ["PTv3_gridnethd_7class_stage_b_self_finetune.py"]
batch_size = 8

# Stage B starts from the public-data Stage A checkpoint through the base file.
# Only half of las_new is added; the other spatial half remains validation.
save_path = "exp/gridnethd/ptv3_7class_stage_b_las_new_half"
epoch = 30
eval_epoch = 3
self_root = "data/ours_stage_b_7class_v1"
public_root = "data/gridnethd/pc7"
train_transform = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]), dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="RandomColorJitter", brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
    dict(type="RandomColorDrop", p=0.15, color_augment=0.0),
    dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="SphereCrop", point_max=150000, mode="random"), dict(type="NormalizeColor"), dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment", "supervision_weight"), feat_keys=("color",)),
]
val_transform = [
    dict(type="Copy", keys_dict={"segment": "origin_segment"}),
    dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True),
    dict(type="NormalizeColor"), dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"), feat_keys=("color",)),
]
data = dict(
    train=dict(
        _delete_=True, type="ConcatDataset",
        datasets=[
            dict(type="Gridnethd", split="train_stage_b_replay.json", data_root=public_root,
                 transform=train_transform, test_mode=False, ignore_index=255,
                 default_supervision_weight=1.0, loop=1),
            dict(type="Gridnethd", split="train_self.json", data_root=self_root,
                 transform=train_transform, test_mode=False, ignore_index=255,
                 default_supervision_weight=1.0, loop=10),
            dict(type="Gridnethd", split="train_las_new_blocks.json", data_root=self_root,
                 transform=train_transform, test_mode=False, ignore_index=255,
                 default_supervision_weight=1.0, loop=10),
        ], loop=1,
    ),
    val=dict(data_root=self_root, split="val_fixed_las_blocks.json", transform=val_transform),
    test=dict(data_root=self_root, split="val_fixed_las_blocks.json"),
)

_base_ = ["PTv3_gridnethd_color.py"]

# Preserve the official GridNet-HD PTv3 recipe (100 epochs, 5 cm voxels,
# 20 m windows, OneCycleLR, CE + Lovasz and test-time augmentation), changing
# only the taxonomy and the initialization checkpoint.
weight = "model_best_PTv3_backbone_only.pth"
resume = False
save_path = "exp/gridnethd/ptv3_7class_public_stage_a"

# Keep 100 effective dataset passes, but split them into ten validation stages
# so model_best.pth tracks convergence of the new seven-class head.
epoch = 100
eval_epoch = 10

model = dict(
    num_classes=7,
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=255),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0,
             ignore_index=255),
    ],
)

data_root = "data/gridnethd/pc7"
names = ["tower", "conductor", "insulator", "vegetation", "building",
         "ground", "other"]
data = dict(
    num_classes=7,
    names=names,
    train=dict(data_root=data_root),
    val=dict(data_root=data_root),
    test=dict(data_root=data_root),
)

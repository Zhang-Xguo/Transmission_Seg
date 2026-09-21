_base_ = ["PTv3_gridnethd_7class_stage_b_eval_fixed.py"]

# ScanNet20 backbone initialization with a fresh 7-class head.  The head is
# deliberately NOT mapped to unrelated ScanNet categories.
weight = "checkpoints/spunet/scannet20_rgb3_7class_random_head.pth"
save_path = "exp/gridnethd/spunet_scannet20_initial_val_fixed"
num_worker = 2
batch_size_test = 1

model = dict(
    _delete_=True,
    type="DefaultSegmentor",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=3,
        num_classes=7,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=255)],
)

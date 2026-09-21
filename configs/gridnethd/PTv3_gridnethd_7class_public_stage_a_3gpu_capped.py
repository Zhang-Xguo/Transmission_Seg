_base_ = ["PTv3_gridnethd_7class_public_stage_a_3gpu.py"]

save_path = "exp/gridnethd/ptv3_7class_public_stage_a_3gpu_capped"
num_worker = 6  # two loader workers per GPU; preprocessing is already prefetched

# Some 20 m chunks contain more than 1.7M raw points.  Bound the post-voxel
# training sample so rare dense chunks cannot exhaust a 40 GB GPU.  Random
# spatial crops vary over the ten dataset loops; validation remains uncropped.
data = dict(
    train=dict(
        transform=[
            dict(
                type="RandomRotate",
                angle=[-1, 1],
                axis="z",
                center=[0, 0, 0],
                p=0.5,
            ),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=150000, mode="random"),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("color",),
            ),
        ]
    )
)

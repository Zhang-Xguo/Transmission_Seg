_base_ = ["PTv3_gridnethd_color_fast.py"]

weight = "exp/gridnethd/ptv3_7class_stage_b_self_finetune/model/model_best.pth"
save_path = "exp/gridnethd/ptv3_7class_stage_b_full_scene_fast"

model = dict(num_classes=7)
data = dict(
    num_classes=7,
    names=["tower", "conductor", "insulator", "vegetation", "building", "ground", "other"],
)

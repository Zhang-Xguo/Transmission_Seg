_base_ = ["PTv3_gridnethd_7class_stage_b_self_finetune.py"]

weight = (
    "exp/gridnethd/ptv3_7class_stage_b_self_finetune/model/model_best.pth"
)
save_path = "exp/gridnethd/ptv3_7class_stage_b_eval_legacy"

data = dict(
    test=dict(
        data_root="data/ours_stage_b_7class_v1",
        split="val_legacy_compare.json",
    )
)

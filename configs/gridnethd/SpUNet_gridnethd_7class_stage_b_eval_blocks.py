_base_ = ["SpUNet_gridnethd_7class_eval_fixed.py"]

weight = "exp/gridnethd/spunet_7class_stage_b_las_new_blocks/model/model_best.pth"
save_path = "exp/gridnethd/spunet_stage_b_best_profiled_val_blocks"
seed = 42

data = dict(test=dict(data_root="data/ours_stage_b_7class_v1",
                      split="val_fixed_las_blocks.json"))

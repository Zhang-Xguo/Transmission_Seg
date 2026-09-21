_base_ = ["PTv3_gridnethd_7class_public_stage_a.py"]

# Memory-safe restart after a rare oversized chunk exhausted a 40 GB GPU at
# batch_size_per_gpu=3.  Three physical GPUs use two samples each; GPU 7 stays
# available for inference/testing.  Learning rates follow linear batch scaling.
save_path = "exp/gridnethd/ptv3_7class_public_stage_a_3gpu"
batch_size = 6
empty_cache = True

optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]

# 完整 Block 推理样例

请将一个完整场景的原始点云和对应真值放入 `input/`：

```text
examples/full_block/
├── input/
│   ├── input.las
│   └── ground_truth.las
└── output/
    └── prediction.las
```

文件约定：

- `input/input.las`：完整 block 原始点云，保留原始坐标、RGB、intensity 等属性；
- `input/ground_truth.las`：与原始点云逐点对应的 7 分类真值；
- `output/prediction.las`：模型生成的完整场景预测结果；
- 7 类标签使用 `0-6`，忽略标签使用 `255`。

完整场景推理命令：

```bash
python scripts/run_stage_b_full_blocks.py \
  --inputs examples/full_block/input/input.las \
  --output-root examples/full_block/output/work \
  --python "$(which python)" \
  --gpu 0 \
  --weight exp/gridnethd/ptv3_7class_stage_b_self_finetune/model/model_best.pth
```

脚本会在 `output/work/` 下生成预处理、推理和合并结果。可将最终完整 LAS
复制或重命名为 `output/prediction.las`。

## 大文件说明

LAS、模型权重和推理输出默认被 `.gitignore` 排除。完整 block 文件通常超过
GitHub 普通文件限制，建议通过仓库的 private Release、Git LFS 或内部文件服务器提供，
并在本说明中补充下载地址和 SHA256。

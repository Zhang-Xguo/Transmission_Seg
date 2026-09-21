# Transmission_Seg

基于 Point Transformer V3（PTv3）的输电通道点云 7 类语义分割训练与推理代码。本仓库整理自实际使用的 Stage A 公共数据预训练、Stage B 输电数据微调、固定验证集评测和完整 LAS 场景推理流程。

正在评估以 SpUNet-v1m1 替代 PTv3；模型、预训练参数转换与初始验证命令见
[`SPUNET_BASELINE.md`](SPUNET_BASELINE.md)。

本仓库只保存研发源码与配置，不包含模型权重、训练数据、LAS 样例、日志、MIPOT/Windows/C++ 工程或历史 BBOX 后处理代码。

## 类别定义

| ID | 类别 | 英文名 |
|---:|---|---|
| 0 | 杆塔 | tower |
| 1 | 导线 | conductor |
| 2 | 绝缘子 | insulator |
| 3 | 植被 | vegetation |
| 4 | 建筑 | building |
| 5 | 地面 | ground |
| 6 | 其他 | other |

训练和预测标签均为 `0-6`，`255` 仅表示需要忽略的 GT 点。

## 目录结构

```text
configs/                  Stage A、Stage B、验证与完整场景推理配置
pointcept/                训练和推理所需的 Pointcept/PTv3 Python 源码
libs/pointops/            Pointops CUDA 扩展源码
tools/train.py            训练入口
tools/test.py             验证和指标计算入口
scripts/                  数据划分、LAS 预处理、整场景推理与结果合并
examples/full_block/      完整 block 输入、真值和输出目录约定
prepare_gridnethd.py      GridNet-HD 数据预处理
requirements.txt          已验证的 Linux CUDA 环境依赖
```

## 环境安装

已验证环境为 Linux、Python 3.8、PyTorch 2.1.0 和 CUDA 12.1。需要 NVIDIA GPU、CUDA 编译工具链和 C++ 编译器。

```bash
git clone https://github.com/Zhang-Xguo/Transmission_Seg.git
cd Transmission_Seg
python -m pip install --no-build-isolation -r requirements.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

## 数据格式

预处理后的每个 tile 是一个目录，核心文件为：

```text
coord.npy                 # float32, [N, 3]
color.npy                 # float32/uint16, [N, 3]
segment.npy               # int32, [N]，标签 0-6，忽略点为 255
supervision_weight.npy    # float32, [N]，仅 Stage B 加权监督使用
```

公开数据默认放在 `data/gridnethd/pc7`，自建数据默认放在 `data/ours_stage_b_7class_v1`。划分 JSON 中保存相对于对应数据根目录的 tile 路径。

## Stage A：公共 GridNet-HD 预训练

配置使用 3 张 GPU，总 batch size 为 6，并限制单次裁剪最多 150000 点：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python tools/train.py \
  --config-file configs/gridnethd/PTv3_gridnethd_7class_public_stage_a_3gpu_capped.py \
  --num-gpus 3
```

Stage A 需要将 PTv3 backbone 初始化权重放到仓库根目录的 `model_best_PTv3_backbone_only.pth`。权重不会提交到 Git。

## Stage B：输电数据微调

Stage B 从 Stage A 的最佳模型初始化，重新建立优化器，并混合公共数据 replay 与自建训练集：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python tools/train.py \
  --config-file configs/gridnethd/PTv3_gridnethd_7class_stage_b_self_finetune.py \
  --num-gpus 3
```

默认读取：

```text
exp/gridnethd/ptv3_7class_public_stage_a_3gpu_capped/model/model_best.pth
data/gridnethd/pc7/train_stage_b_replay.json
data/ours_stage_b_7class_v1/train_self.json
data/ours_stage_b_7class_v1/val_fixed.json
```

路径和输出目录可通过 `--options key=value` 覆盖。

## 固定验证集评测

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  --config-file configs/gridnethd/PTv3_gridnethd_7class_stage_b_eval_fixed.py \
  --num-gpus 1 \
  --options \
    weight=exp/gridnethd/ptv3_7class_stage_b_self_finetune/model/model_best.pth \
    save_path=exp/gridnethd/stage_b_eval_fixed_manual \
    batch_size_test=1 num_worker=2
```

`test.log` 会报告整体 mIoU、mAcc、allAcc，以及各类别 IoU、Recall/Accuracy。预测数组保存在实验输出目录的 `result/` 中。

## 完整 LAS 场景推理

仓库已预留 [`examples/full_block`](examples/full_block/README.md) 目录。将原始点云
命名为 `input/input.las`，将对应真值命名为 `input/ground_truth.las`；推理结果放在
`output/` 中。

完整场景流程分为三步：

1. `prepare_custom_las_ptv3_stream.py` 将 LAS 分块转换为 PTv3 输入；
2. `tools/test.py` 对分块数据推理；
3. `merge_custom_ptv3.py` 按原始点顺序合并预测并写回 LAS。

批量入口：

```bash
python scripts/run_stage_b_full_blocks.py \
  --inputs /path/to/block_0.las /path/to/block_1.las \
  --output-root outputs/full_blocks \
  --python "$(which python)" \
  --gpu 0 \
  --weight exp/gridnethd/ptv3_7class_stage_b_self_finetune/model/model_best.pth
```

也可以使用 `--input-list scenes.txt`，文本中每行填写一个 LAS 路径。具体参数以 `python scripts/run_stage_b_full_blocks.py --help` 为准。输出 LAS 保持原始点顺序，并写入 7 类预测结果和类别显示颜色。

## 数据划分相关脚本

- `build_stage_b_manifest.py`：扫描和记录场景/tile 对应关系；
- `write_stage_b_splits.py`：生成 Stage B 固定训练与验证划分；
- `make_las_new_half_split.py`：按 block 选择一半 `las_new` 加入训练；
- `split_las_new_tiles_by_source_block.py`：保持 block 与 tile 的完整对应关系；
- `export_stage_b_validation_las.py`：导出验证结果用于查看；
- `verify_block_prediction.py`：核对完整场景输出点数、字段和预测范围。

## 模型与数据

模型权重、原始 LAS、预处理数据和实验输出被 `.gitignore` 排除。共享实验时应单独提供：

```text
model_best.pth
数据划分 JSON
必要的样例 LAS 或下载地址
训练配置与对应 Git commit
```

本项目基于 Pointcept/PTv3 开发，引用和许可证信息见源码头部及 `LICENSE`。

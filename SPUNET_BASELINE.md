# SpUNet-v1m1 初始基线

本实验使用 Pointcept 的 `SpUNet-v1m1`（SpConv SparseUNet）替代 PTv3。
源码：`pointcept/models/sparse_unet/spconv_unet_v1m1_base.py`。

预训练权重来自 [torch-pointcloud 的 ScanNet20 SpUNet-v1m1](https://huggingface.co/torch-pointcloud/spunet-v1m1.scannet20.pointcept)，
是 Pointcept 模型的转换版本。原模型有 6 个输入通道（RGB + normal）和 20 类输出；
本项目只有 RGB 输入和 7 类输出。转换脚本保留 RGB 对应的输入卷积权重、映射其余
354 个骨干张量，并用固定种子 `42` 初始化新的 7 类分类头。**这个分类头未经输电
数据训练，因此此处验证指标仅用于确认数据、模型和评测流程可以运行，不代表模型
的真实可用精度，也不能与已经微调过的 PTv3 指标直接比较。**

## 下载和转换

在仓库根目录执行：

```bash
mkdir -p checkpoints/spunet
curl -L --fail -o checkpoints/spunet/scannet20_model.safetensors \
  https://huggingface.co/torch-pointcloud/spunet-v1m1.scannet20.pointcept/resolve/main/model.safetensors

PYTHONPATH=. python scripts/prepare_spunet_pretrained.py \
  --source checkpoints/spunet/scannet20_model.safetensors \
  --output checkpoints/spunet/scannet20_rgb3_7class_random_head.pth
```

权重及转换后的 checkpoint 位于 `checkpoints/`，不会提交到 Git。
本次下载文件 SHA256：`11658b4daecc75d1b5510469b778ddf4cd61d4f95619bc66439c0e59e58c71cf`。

## 固定验证集评测

使用与 PTv3 相同的 `data/ours_stage_b_7class_v1/val_fixed.json`（107 tiles）。
需先把原有 `data/` 目录放到仓库根目录，或建立本机符号链接。然后运行：

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 python tools/test.py \
  --config-file configs/gridnethd/SpUNet_gridnethd_7class_eval_fixed.py \
  --num-gpus 1 \
  --options save_path=exp/gridnethd/spunet_scannet20_initial_val_fixed \
            num_worker=2 batch_size_test=1
```

日志：`exp/gridnethd/spunet_scannet20_initial_val_fixed/test.log`；
逐 tile 预测：`exp/gridnethd/spunet_scannet20_initial_val_fixed/result/`。

如将 107 个 tile 分配到多个 GPU 并行评测，可用下列脚本汇总各输出目录，
得到逐类 IoU、Precision、Recall 和完整混淆矩阵：

```bash
python scripts/evaluate_spunet_val_fixed.py \
  --data-root data/ours_stage_b_7class_v1 \
  --split data/ours_stage_b_7class_v1/val_fixed.json \
  --result-dir exp/gridnethd/spunet_scannet20_initial_val_fixed/result \
               exp/gridnethd/spunet_scannet20_initial_val_fixed_part1/result \
               exp/gridnethd/spunet_scannet20_initial_val_fixed_part2/result \
  --output exp/gridnethd/spunet_scannet20_initial_val_fixed/metrics_107.json
```

## 初始验证结果（未训练 7 类头）

固定验证集全部 107 个 tile，23,184,589 个有效 GT 点。此次将清单分为三份，
分别在 GPU 4、5、6 上执行相同权重的推理，再按逐点混淆矩阵汇总：

| 整体指标 | 数值 |
|---|---:|
| mIoU | 1.67% |
| 宏平均 Precision（无预测类别记 0） | 11.30% |
| 宏平均 Recall | 11.82% |
| 逐点 Accuracy | 5.43% |

| 类别 | IoU | Precision | Recall |
|---|---:|---:|---:|
| tower | 5.39% | 6.19% | 29.44% |
| conductor | 0.003% | 3.87% | 0.003% |
| insulator | 0.95% | 0.97% | 32.09% |
| vegetation | 1.20% | 63.14% | 1.21% |
| building | 0.00% | 无预测点 | 0.00% |
| ground | 4.11% | 4.92% | 20.02% |
| other | 0.00% | 无预测点 | 0.00% |

完整报告（包括混淆矩阵）在本机：
`exp/gridnethd/spunet_scannet20_initial_val_fixed/metrics_107.json`。
结果极低主要是因为 ScanNet20 的室内语义与输电 7 类不对应，且 7 类头没有训练。
不能据此判断微调后的 SpUNet 上限。

## 后续训练

初始验证通过后，应在相同 7 类数据上训练新分类头并微调整个 SpUNet，至少完成
公共 GridNet-HD Stage A，再用固定的 Stage B 数据微调，最后按相同的固定验证集
比较 IoU、Recall、Precision 和推理耗时。未经训练的当前 checkpoint 不应作为
正式推理模型交付。

工程化方面，SpConv 官方提供 Windows 预编译包和纯 C++ `libspconv` 路线，
但其 PyTorch 接口不能直接用 TorchScript/libtorch 部署；迁移后的 Windows 推理、
`libspconv` 集成与最终 `dist.tar` 仍需单独验证。

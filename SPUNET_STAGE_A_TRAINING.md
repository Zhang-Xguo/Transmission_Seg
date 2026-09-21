# SpUNet 公开 GridNet-HD 7 类 Stage A 训练

此次只训练 Stage A，不启动自建数据 Stage B。与原 PTv3 Stage A 一样，使用
`data/gridnethd/pc7/train_final`（13,604 tiles）训练，
`data/gridnethd/pc7/val_final`（2,928 tiles）验证；100 次有效数据遍历，
每 10 次验证并保存 `model_best.pth`。初始权重为 ScanNet20 SpUNet 骨干，
7 类分类头随机初始化，具体转换见 `SPUNET_BASELINE.md`。

GPU 固定为物理卡 4、5、6。总 batch 48（每卡 16），共 6 个 DataLoader worker，
每卡 2 个，并将 OpenMP、MKL、OpenBLAS、NumExpr 线程均限制为 1。
在高密度 tile 压测中，显存峰值约 20–21 GB/卡；更大的总 batch 72 使用约
30–32 GB/卡，但吞吐更低，因此选择 48。

从仓库根目录执行：

```bash
PYTHON="$(which python)" bash scripts/train_spunet_stage_a.sh
```

输出：

```text
exp/gridnethd/spunet_7class_public_stage_a/train.log
exp/gridnethd/spunet_7class_public_stage_a/model/model_best.pth
exp/gridnethd/spunet_7class_public_stage_a/model/model_last.pth
exp/gridnethd/spunet_7class_public_stage_a/TRAIN_STATUS.json
```

`TRAIN_STATUS.json` 仅在训练进程退出后写入，`completed=true` 要求：退出码为 0、
完成 10 次验证且最佳权重存在。该报告可用于确认训练是否真正结束。

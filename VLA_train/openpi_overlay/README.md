# openpi overlay — 自定义文件的 openpi 外备份

openpi (`VLAs/openpi/`) 会被重新下载，因此 serve/train/eval 链路依赖的所有本地
自定义文件在这里保一份（openpi 目录之外）。fresh clone 后一键铺回：

```bash
bash /data3/zhixu/LIBERO-PeRM/VLA_train/openpi_overlay/apply_overlay.sh   # 默认铺到 VLAs/openpi
```

- Snapshot 时间：2026-07-25；基准 openpi commit `937167a`（"commit before merge"，
  上游 `e061c09` Pi05+PyTorch 之后）。
- 内容清单见 `apply_overlay.sh` 头注释；共 25 个文件、1.2M（含全部自定义 config 的
  norm_stats assets）。
- **不含 checkpoints/（498G）**——重装 openpi 前务必把 `VLAs/openpi/checkpoints/`
  整个搬走或保留，删了无法恢复。W 的 ckpt 在 repo 根 `checkpoints/`，不受影响。
- openpi 内文件又改动后，重新 snapshot：在 `VLAs/openpi` 下把 git status 里的
  modified/untracked 关键文件按相对路径 cp 进 `overlay/` 即可（保持目录结构）。

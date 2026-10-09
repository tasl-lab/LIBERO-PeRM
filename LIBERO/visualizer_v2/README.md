# P-Bench Dataset Quality Visualizer V2

用于 P_Bench 训练数据（heuristic skill 生成的 rollouts）的质量审查工具。

## 特性

相对于 V1（`../visualizer/`）的改进：
- **视频而非 GIF** — 从 MP4 裁剪，每个 ~60KB（V1 GIF 平均 9MB）
- **Agentview only** — 裁剪掉 wrist/frontview 干扰
- **Review mode** — 逐个 layout 审查，键盘导航
- **Comment 系统** — 对每个 layout 写评论，localStorage 持久化 + Export/Import JSON
- **总大小 26MB** (V1 2.6GB，缩小 100 倍)

## 文件结构

```
visualizer_v2/
├── README.md           # 本文档
├── template.html       # HTML 模板（含 __MISSIONS_JSON__ 占位符）
├── index.html          # 注入数据后的最终页面（build 生成）
├── videos/             # 裁剪后的 MP4 视频，~26MB
│   ├── 1-1_004/
│   │   ├── Succ_batch_0.mp4
│   │   ├── Succ_batch_1.mp4
│   │   └── Fail_batch_3.mp4
│   └── ...
└── (comments.json)     # 用户 Export 后生成的评论文件
```

构建脚本在上级目录：`../build_visualizer_v2.py`

## 使用步骤

### 1. 首次 build

```bash
cd /home/vla-reasoning/proj/P_Bench/LIBERO/
python build_visualizer_v2.py
```

做两件事：
1. 扫描 `dataset/perm_*/libero_*/Videos/rollout_*.mp4`，用 ffmpeg 裁剪成 agentview only
2. 读取 `dataset/*/mission_info.yaml` + `batch_run/sr_summary.txt`，注入到 `template.html` 生成最终 `index.html`

Missions 按 series → task → index 数字排序。

### 2. 数据更新后重 build

如果重新生成了数据（新的 HDF5/视频），重跑 build：

```bash
# 全量（重裁剪 + 重注入数据）
python build_visualizer_v2.py

# 只重新注入数据（视频没变）
python build_visualizer_v2.py --only-html

# 只生成视频，不改 HTML
python build_visualizer_v2.py --skip-html  # (如果实现)
```

### 3. 启动 Web Server

```bash
cd /home/vla-reasoning/proj/P_Bench/LIBERO/visualizer_v2
python -m http.server 8088
```

浏览器打开 `http://<服务器 IP>:8088/`。

## 页面功能

### Grid Mode (默认)

- 卡片式浏览，每张卡片一个 mission
- 左侧 sidebar 过滤：Series / SR / Workspace / Has Comments / Result
- 搜索框支持 mission_id 或 language
- 随机抽样（20 个）、展开全部、折叠全部
- 有评论的 mission 卡片上有紫色标记

### Review Mode (新)

点击 sidebar 的 `Review Mode` 按钮进入。

- **按数字顺序**遍历所有 mission（1-1_004 → 1-1_006 → ... → 3-15_041_1）
- 左侧：大图视频查看器 + 同 mission 下其他 rollout 缩略图（Fail 在前）
- 右侧：mission 元信息 + comment textarea + Prev/Next 导航

**键盘快捷键**：
- `←` / `→` — 前一个/下一个 mission
- `↑` / `↓` — 同 mission 内切换 rollout
- `C` — 聚焦 comment 输入框
- `Esc` — 退出到 Grid Mode

### Comment 系统

- 每个 mission 可写评论（例如 "pick 时抖动过大"）
- 保存在浏览器 localStorage（下次打开自动加载）
- **"Export Comments"** — 下载 `comments.json`（可备份或分享给同事）
- **"Import Comments"** — 上传 `comments.json` 覆盖 localStorage

## 技术说明

- **视频压缩**：ffmpeg 从 MP4 裁剪左 1/3（agentview），h264/crf=28
- **数据注入**：`template.html` 里的 `const MISSIONS = __MISSIONS_JSON__;` 被 build 脚本替换为实际 JSON
- **无依赖**：页面是 self-contained HTML（一个文件），本地 http.server 即可

## 常见问题

### 视频加载慢/看不了

- 确认 `videos/` 目录存在且有 MP4 文件
- 检查 http.server 是否在跑：`ss -tlnp | grep 8088`

### Comments 丢失

- localStorage 只对同一浏览器/同一 URL 有效
- 切换浏览器前先 "Export Comments" → 新浏览器 "Import Comments"

### 数据更新后 visualizer 不变

- Build 脚本用 ffmpeg `-y` 覆盖旧视频
- 如果 mission 数量变了（新增 layouts），必须重新 build（`python build_visualizer_v2.py`）
- 浏览器刷新时用 Ctrl+Shift+R 强制刷新缓存

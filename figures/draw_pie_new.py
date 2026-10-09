import matplotlib

matplotlib.use("Agg")  # server / no GUI
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    # --- 颜色定义 ---
    COLORS_PREFERENCE = ["#e57373", "#64b5f6", "#81c784", "#d0d0d0"]
    COLORS_DIFFICULTY = ["#a5d6a7", "#90caf9", "#ef9a9a", "#d0d0d0"]
    COLORS_COMBINATION = ["#ef9a9a", "#ba68c8", "#ffb74d", "#90caf9", "#4db6ac", "#a5d6a7"]

    # 新增：Trajectory Length 和 Skills 的颜色（使用相似色调）
    COLORS_TRAJECTORY = ["#c8f7dc", "#a5d6a7", "#81c784", "#66bb6a", "#4caf50", "#43a047", "#388e3c"]
    COLORS_SKILLS = ["#f7c6c6", "#ef9a9a", "#e57373", "#ef5350", "#f44336", "#e53935", "#d32f2f"]

    # --- 数据准备 ---
    plot_data = [
        {
            "labels": ["low-traj", "high-skill", "high-profile"],
            "values": [5, 7, 14],
            "colors": ["#f7c6c6", "#cfe8ff", "#c8f7dc"],
            "title": "Single preference type"
        },
        {
            "labels": ["low-traj", "high-skill", "high-profile", "No preference"],
            "values": [20, 27, 35, 11],
            "colors": COLORS_PREFERENCE,
            "title": "Single preference missions by type"
        },
        {
            "labels": ["Easy", "Medium", "Hard"],
            "values": [13, 7, 6],
            "colors": ["#c8f7dc", "#cfe8ff", "#f7c6c6"],
            "title": "Single preference difficulty"
        },
        {
            "labels": ["Easy", "Medium", "Hard", "No preference"],
            "values": [37, 28, 17, 11],
            "colors": COLORS_DIFFICULTY,
            "title": "Single preference missions by difficulty"
        },
        {
            "labels": ["low + low", "low + high-skill", "low + high-profile",
                       "high-skill + high-skill", "high-skill + high-profile", "high-profile + high-profile"],
            "values": [8, 14, 6, 8, 4, 12],
            "colors": COLORS_COMBINATION,
            "title": "Multi-preference missions by type"
        },
        # 新增：Trajectory Length 分布
        {
            "labels": ["0-200", "200-400", "400-600", "600-800", "800-1000", "1000-1200", "1200-1400"],
            "values": [10, 47, 27, 11, 1, 2, 1],
            "colors": COLORS_TRAJECTORY,
            "title": "Trajectory Length Distribution"
        },
        # 新增：Skills 数量分布
        {
            "labels": ["1 skill", "2 skills", "3 skills", "4 skills", "7 skills", "8 skills"],
            "values": [21, 213, 9, 118, 6, 9],
            "colors": COLORS_SKILLS,
            "title": "Skills Number Distribution"
        }
    ]

    # --- 开始绘图（横向布局）---
    # 设置大图画布：2行6列（上面4个，下面3个居中）
    fig = plt.figure(figsize=(24, 10), dpi=300)
    gs = fig.add_gridspec(2, 6, hspace=0.25, wspace=0.25)

    subtitles = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]

    for i, data in enumerate(plot_data):
        # 确定子图位置
        if i < 4:
            # 前4个图：第一行，每个占1.5列
            col_start = int(i * 1.5)
            col_end = int((i + 1) * 1.5)
            ax = fig.add_subplot(gs[0, col_start:col_end])
        else:
            # 后3个图：第二行，居中对齐（从第0.5列开始）
            offset = i - 4  # 0, 1, 2
            col_start = int(offset * 2)
            col_end = int((offset + 1) * 2)
            ax = fig.add_subplot(gs[1, col_start:col_end])

        values = data["values"]
        total = sum(values)

        # 对于新增的两个图(f和g)，使用更智能的标签处理
        if i >= 5:
            # 将小于3%的合并为"Others"
            threshold = 3.0
            large_indices = [idx for idx, val in enumerate(values) if 100.0 * val / total >= threshold]
            small_indices = [idx for idx, val in enumerate(values) if 100.0 * val / total < threshold]

            if small_indices:
                # 计算Others的总和
                others_sum = sum(values[idx] for idx in small_indices)
                others_labels = [data["labels"][idx] for idx in small_indices]

                # 重新构建数据
                new_values = [values[idx] for idx in large_indices] + [others_sum]
                new_labels = [data["labels"][idx] for idx in large_indices] + ["Others"]
                new_colors = [data["colors"][idx] for idx in large_indices] + ["#d0d0d0"]

                # 绘制饼图，不使用autopct，手动添加标签
                wedges, texts = ax.pie(
                    new_values,
                    colors=new_colors,
                    startangle=90,
                    wedgeprops=dict(edgecolor="white", linewidth=1.5),
                )

                # 手动添加标签和百分比（标签在外，百分比在标签下方）
                for w, label, val in zip(wedges, new_labels, new_values):
                    ang = (w.theta1 + w.theta2) / 2.
                    # 标签位置
                    x = 1.25 * np.cos(np.deg2rad(ang))
                    y = 1.25 * np.sin(np.deg2rad(ang))
                    pct = 100.0 * val / sum(new_values)

                    # 标签文字
                    ax.text(
                        x, y, f"{label}\n{pct:.1f}%",
                        ha="center", va="center",
                        fontsize=11,
                        fontweight="bold"
                    )

                # 添加Others的详细说明（在图下方）
                others_detail = ", ".join([f"{lbl}" for lbl in others_labels])
                ax.text(0, -1.45, f"Others: {others_detail}",
                         ha="center", va="top", fontsize=11, style='italic')
            else:
                # 没有小比例，正常绘制
                wedges, texts = ax.pie(
                    values,
                    colors=data["colors"],
                    startangle=90,
                    wedgeprops=dict(edgecolor="white", linewidth=1.5),
                )

                # 手动添加标签和百分比
                for w, label, val in zip(wedges, data["labels"], values):
                    ang = (w.theta1 + w.theta2) / 2.
                    x = 1.25 * np.cos(np.deg2rad(ang))
                    y = 1.25 * np.sin(np.deg2rad(ang))
                    pct = 100.0 * val / total

                    ax.text(
                        x, y, f"{label}\n{pct:.1f}%",
                        ha="center", va="center",
                        fontsize=11,
                        fontweight="bold"
                    )
        else:
            # 原来的图保持不变
            wedges, _ = ax.pie(
                values,
                colors=data["colors"],
                startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=1.5),
            )

            # 标注文字
            for w, label, val in zip(wedges, data["labels"], values):
                ang = (w.theta1 + w.theta2) / 2.
                x = 1.25 * np.cos(np.deg2rad(ang))
                y = 1.25 * np.sin(np.deg2rad(ang))
                pct = 100.0 * val / total

                if pct < 3.0:
                    x *= 1.5
                    y *= 1.5

                ax.text(
                    x, y, f"{label}\n{pct:.1f}%",
                    ha="center", va="center",
                    fontsize=11,
                    fontweight="bold"
                )

        # 设置子图标题和下标
        ax.set_title(data["title"], fontsize=12, pad=15, fontweight="bold")

        # 在子图左上角添加 (a), (b) 等下标
        ax.text(-1.5, 1.3, subtitles[i], fontsize=16, fontweight="bold", va="top", ha="left")
        ax.set(aspect="equal")

    # 保存
    os.makedirs("./figures", exist_ok=True)
    save_path = "./figures/combined_preference_pies_horizontal.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Combined figure saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
import matplotlib
matplotlib.use("Agg")  # server / no GUI

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
import os

def draw_pie_outer_percent(
    labels,
    values,
    colors,
    title,
    save_path,
    startangle=90,
):

    import numpy as np
    import matplotlib.pyplot as plt
    save_path=os.path.join("./figures",save_path )
    total = sum(values)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=300)

    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=startangle,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )

    ax.set(aspect="equal")
    ax.set_title(title, fontsize=16, pad=18)

    # 外圈文字
    for w, label, val in zip(wedges, labels, values):
        ang = (w.theta1 + w.theta2) / 2
        x = 1.28 * np.cos(np.deg2rad(ang))
        y = 1.28 * np.sin(np.deg2rad(ang))
        pct = 100.0 * val / total

        ax.text(
            x,
            y,
            f"{label}\n{pct:.1f}%",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

def main():
    COLORS_PREFERENCE_DATASET = [
        "#e57373",  # low-traj        (soft red, 干净安全)
        "#64b5f6",  # high-skill      (clear blue)
        "#81c784",  # high-profile    (fresh green)
        "#d0d0d0",  # No preference   (light gray)
    ]
    COLORS_DIFFICULTY_DATASET = [
        "#a5d6a7",  #
        "#90caf9",  #

        "#ef9a9a",  # Hard

        "#d0d0d0",  # No preference
    ]

    # 混合组合的颜色定义 (语义化配色)
    COLORS_COMBINATION = [
        "#ef9a9a",  # L1+L1: 浅红
        "#ba68c8",  # L1+L2: 紫色 (红+蓝)
        "#ffb74d",  # L1+L3: 橙色 (红+绿)
        "#90caf9",  # L2+L2: 浅蓝
        "#4db6ac",  # L2+L3: 青色 (蓝+绿)
        "#a5d6a7",  # L3+L3: 浅绿
    ]

    pie_jobs = [
        # ---------------- Preference abstraction ----------------
        dict (
            labels=["low-traj", "high-skill", "high-profile"],
            # 对应关系：low-traj(红), high-skill(蓝), high-profile(绿)
            colors=["#f7c6c6", "#cfe8ff", "#c8f7dc"],
            values=[5, 7, 14],
            title="Single preference type",
            save_path="pie_preference_abstraction.png",
        ),

        # ---------------- Difficulty (1-26 missions) ----------------
        dict (
            labels=["Easy", "Medium", "Hard"],
            # 对应关系：green, blue, Hard(红)
            colors=["#c8f7dc","#cfe8ff" , "#f7c6c6"],
            values=[13, 7, 6],
            title="Single preference difficulty",
            save_path="pie_difficulty.png",
        ),
        # ---------------- Difficulty (Total 93 missions) ----------------
        dict (
            labels=["Easy", "Medium", "Hard", "No preference"],
            values=[37, 28, 17, 11],
            colors=COLORS_DIFFICULTY_DATASET,
            title="Single preference missions by difficulty",
            save_path="pie_difficulty_dataset.png",
        ),

        # ---------------- Type / Preference Abstraction (Total 93 missions) ----------------
        dict (
            labels=["low-traj", "high-skill", "high-profile", "No preference"],
            values=[20, 27, 35, 11],
            colors=COLORS_PREFERENCE_DATASET,
            title="Single preference missions by type",
            save_path="pie_preference_dataset.png",
        ),

        # ---------------- Preference Composition (L1/L2/L3 Combinations) ----------------
        dict (
            labels=[
                "low + low",
                "low + high-skill",
                "low + high-profile",
                "high-skill + high-skill",
                "high-skill + high-profile",
                "high-profile + high-profile"
            ],
            values=[8, 14, 6, 8, 4, 12],  # 总计 52 个任务
            colors=COLORS_COMBINATION,
            title="Multi-preference missions by type",
            save_path="pie_preference_composition.png",
        ),
    ]

    for job in pie_jobs:
        draw_pie_outer_percent(
            labels=job["labels"],
            values=job["values"],
            colors=job["colors"],
            title=job["title"],
            save_path=job["save_path"],
        )


if __name__ == "__main__":
    main()

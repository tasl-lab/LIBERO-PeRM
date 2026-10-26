import os
import shutil
from pathlib import Path
import re


def extract_layout_id(layout_name):
    """从Layout_1_4或Layout_1_6_0中提取layout id (1_4或1_6_0)"""
    match = re.match(r'Layout_(.+)', layout_name)
    return match.group(1) if match else None


def extract_mission_id(mission_name):
    """从Mission_67或Mission_m14中提取mission id (67或m14)"""
    match = re.match(r'Mission_(.+)', mission_name)
    return match.group(1) if match else None

def parse_mission_string(full_name):
    """
    输入: Mission_m1_L4_8
    输出: mission_id='m1', layout_id='4_8'
    """
    # 匹配 Mission_ 后面跟着的 mission_id，以及 L 后面跟着的 layout_id
    match = re.match(r"Mission_(m\d+)_L(.+)", full_name)
    if match:
        return match.group(1), match.group(2)
    return None, None
def find_matching_video_demo_folder(mission_id, layout_id, video_demo_path):
    """在video_demo目录下找到匹配的folder"""
    # 遍历video_demo目录
    for folder_name in os.listdir(video_demo_path):
        folder_path = os.path.join(video_demo_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 尝试匹配 Mission_{mission_id}_L{layout_id} 格式
        # 例如: Mission_6_L4_3
        pattern = f'Mission_{mission_id}_L{layout_id.replace("_", "_")}'
        if folder_name.startswith(f'Mission_{mission_id}_L'):
            return folder_path

    return None


def copy_pref_files(source_dir, dest_dir, mission_id):
    """复制High_pref和Low_pref文件（如果存在）"""
    high_pref_file = f'High_pref_for_mission_{mission_id}.txt'
    low_pref_file = f'Low_pref_for_mission_{mission_id}.txt'

    copied_files = []

    # 复制High_pref文件
    high_pref_path = os.path.join(source_dir, high_pref_file)
    if os.path.exists(high_pref_path):
        shutil.copy2(high_pref_path, dest_dir)
        copied_files.append(high_pref_file)
        print(f"  ✓ 复制: {high_pref_file}")

    # 复制Low_pref文件
    low_pref_path = os.path.join(source_dir, low_pref_file)
    if os.path.exists(low_pref_path):
        shutil.copy2(low_pref_path, dest_dir)
        copied_files.append(low_pref_file)
        print(f"  ✓ 复制: {low_pref_file}")

    return copied_files


def main_v0():
    # 源路径和目标路径
    source_base_path = '/data3/zhixu/LIBERO-PeRM/LIBERO/dataset'
    dest_base_path = '/data3/zhixu/LIBERO-PeRM/video_demo'

    # 检查路径是否存在
    if not os.path.exists(source_base_path):
        print(f"错误: 源路径不存在 {source_base_path}")
        return

    if not os.path.exists(dest_base_path):
        print(f"错误: 目标路径不存在 {dest_base_path}")
        return

    # 统计信息
    total_missions = 0
    copied_count = 0
    skipped_count = 0

    # 遍历所有Layout文件夹
    for layout_name in sorted(os.listdir(source_base_path)):
        layout_path = os.path.join(source_base_path, layout_name)

        # 只处理Layout_开头的目录
        if not os.path.isdir(layout_path) or not layout_name.startswith('Layout_'):
            continue

        layout_id = extract_layout_id(layout_name)
        if not layout_id:
            continue

        print(f"\n处理 {layout_name} (Layout ID: {layout_id})")

        # 遍历每个Layout下的Mission文件夹
        for mission_name in sorted(os.listdir(layout_path)):
            mission_path = os.path.join(layout_path, mission_name)

            # 只处理Mission_开头的目录
            if not os.path.isdir(mission_path) or not mission_name.startswith('Mission_'):
                continue

            mission_id = extract_mission_id(mission_name)
            if not mission_id:
                continue

            total_missions += 1

            # 在video_demo目录下查找匹配的文件夹
            matching_folder = find_matching_video_demo_folder(mission_id, layout_id, dest_base_path)

            if matching_folder:
                print(f"  找到匹配: {mission_name} -> {os.path.basename(matching_folder)}")

                # 复制pref文件
                copied_files = copy_pref_files(mission_path, matching_folder, mission_id)

                if copied_files:
                    copied_count += 1
                else:
                    print(f"  ⚠ 未找到pref文件")
                    skipped_count += 1
            else:
                print(f"  ✗ 未找到匹配的video_demo文件夹: {mission_name}")
                skipped_count += 1

    # 输出统计信息
    print(f"\n" + "=" * 50)
    print(f"处理完成!")
    print(f"总任务数: {total_missions}")
    print(f"成功复制: {copied_count}")
    print(f"跳过/未找到: {skipped_count}")
    print("=" * 50)


def main():
    # 基础路径配置
    source_base_path = '/data3/zhixu/LIBERO-PeRM/LIBERO/dataset'
    dest_base_path = '/data3/zhixu/LIBERO-PeRM/video_demo'

    # 你提供的 52 个任务目标列表
    # missions_to_process = [
    #     'Mission_m1_L4_8', 'Mission_m2_L5_9', 'Mission_m3_L5_2', 'Mission_m4_L4_2',
    #     'Mission_m5_L5_5', 'Mission_m6_L4_5', 'Mission_m7_L10_5', 'Mission_m8_L10_4',
    #     'Mission_m9_L8_5', 'Mission_m10_L11_3', 'Mission_m11_L1_8', 'Mission_m12_L3_8',
    #     'Mission_m13_L9_2', 'Mission_m14_L1_1', 'Mission_m15_L3_2', 'Mission_m16_L1_3',
    #     'Mission_m17_L2_4', 'Mission_m18_L11_2', 'Mission_m19_L9_9', 'Mission_m20_L5_10',
    #     'Mission_m21_L7_4', 'Mission_m22_L3_6_1', 'Mission_m23_L1_7', 'Mission_m24_L7_2',
    #     'Mission_m25_L3_3', 'Mission_m26_L1_4', 'Mission_m27_L9_8', 'Mission_m28_L6_8',
    #     'Mission_m29_L11_7', 'Mission_m30_L11_5_2', 'Mission_m31_L6_6', 'Mission_m32_L6_4',
    #     'Mission_m33_L1_10', 'Mission_m34_L1_9', 'Mission_m35_L2_5', 'Mission_m36_L11_6',
    #     'Mission_m37_L7_5', 'Mission_m38_L3_7', 'Mission_m39_L5_11', 'Mission_m40_L9_10',
    #     'Mission_m41_L6_3', 'Mission_m42_L9_7', 'Mission_m43_L2_2_1', 'Mission_m44_L8_3',
    #     'Mission_m45_L6_7', 'Mission_m46_L6_5', 'Mission_m47_L4_7', 'Mission_m48_L5_8',
    #     'Mission_m49_L9_4_3', 'Mission_m50_L1_6_1', 'Mission_m51_L2_6', 'Mission_m52_L10_7'
    # ]
    missions_to_process=['Mission_1_L5_1',
    'Mission_2_L4_1',
    'Mission_3_L5_2',
    'Mission_4_L4_2',
    'Mission_5_L5_3',
    'Mission_6_L4_3',
    'Mission_7_L5_4',
    'Mission_8_L4_4',
    'Mission_9_L5_5',
    'Mission_10_L4_5',
    'Mission_11_L9_1',
    'Mission_12_L5_6',
    'Mission_13_L10_1',
    'Mission_14_L2_1',
    'Mission_15_L7_1',
    'Mission_16_L3_1',
    'Mission_17_L9_2',
    'Mission_18_L1_1',
    'Mission_19_L9_3',
    'Mission_20_L1_2',
    'Mission_21_L2_4',
    'Mission_22_L5_7',
    'Mission_23_L3_2',
    'Mission_24_L1_3',
    'Mission_25_L10_2',
    'Mission_26_L11_1',
    'Mission_27_L3_3',
    'Mission_28_L1_4',
    'Mission_29_L3_4',
    'Mission_30_L1_5_0',
    'Mission_31_L9_4_2',
    'Mission_32_L1_6_0',
    'Mission_33_L8_2',
    'Mission_34_L4_6',
    'Mission_35_L9_5_0',
    'Mission_36_L6_1',
    'Mission_37_L9_6',
    'Mission_38_L6_2',
    'Mission_39_L7_2',
    'Mission_40_L1_7',
    'Mission_41_L7_3',
    'Mission_42_L3_5',
    'Mission_43_L8_3',
    'Mission_44_L2_2_0',
    'Mission_45_L7_4',
    'Mission_46_L3_6_1',
    'Mission_47_L6_3',
    'Mission_48_L9_7',
    'Mission_49_L7_5',
    'Mission_50_L3_7',
    'Mission_51_L2_3',
    'Mission_52_L10_6'

]

    # 统计信息
    total_missions = len(missions_to_process)
    copied_count = 0
    skipped_count = 0

    print(f"开始处理，总计任务数: {total_missions}\n")

    for full_mission_name in missions_to_process:
        # 1. 解析任务名提取 ID
        # 使用正则从 "Mission_m1_L4_8" 提取 mission_id="m1" 和 layout_id="4_8"
        match = re.match(r"Mission_(\d+)_L(.+)", full_mission_name)

        if not match:
            print(f"  ✗ 格式解析失败，跳过: {full_mission_name}")
            skipped_count += 1
            continue

        mission_id = match.group(1)  # 如 "m1"
        layout_id = match.group(2)  # 如 "4_8"

        # 2. 构造源目录路径
        # 假设源路径结构为: .../dataset/Layout_4_8/Mission_m1
        source_mission_dir = os.path.join(source_base_path, f"Layout_{layout_id}", f"Mission_{mission_id}")

        # 3. 构造并创建目标目录路径
        # 目标路径结构为: .../video_demo/Mission_m1_L4_8
        target_mission_dir = os.path.join(dest_base_path, f"{full_mission_name}_ood")

        if os.path.exists(source_mission_dir):
            # 创建目标文件夹（如果不存在）
            os.makedirs(target_mission_dir, exist_ok=True)

            print(f"正在同步: {full_mission_name}")

            # 4. 执行文件复制逻辑
            # 调用你原有的 copy_pref_files 函数
            copied_files = copy_pref_files(source_mission_dir, target_mission_dir, mission_id)

            if copied_files:
                copied_count += 1
            else:
                print(f"  ⚠ 源目录存在但未找到 pref 文件: {source_mission_dir}")
                skipped_count += 1
        else:
            print(f"  ✗ 找不到源目录: {source_mission_dir}")
            skipped_count += 1

    # 输出统计结果
    print(f"\n" + "=" * 50)
    print(f"处理完成!")
    print(f"计划任务: {total_missions}")
    print(f"成功复制: {copied_count}")
    print(f"未完成数: {skipped_count}")
    print("=" * 50)

if __name__ == "__main__":
    main()
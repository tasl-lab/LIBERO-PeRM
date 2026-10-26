#!/usr/bin/env python3
"""
根据 libero_perm_eval 映射直接重命名 (无确认)
"""

import os
import re

base_dir = "/data3/zhixu/LIBERO-PeRM/VLAs/openpi/third_party/libero/libero/libero/bddl_files"
eval_dir = os.path.join(base_dir, "libero_perm_eval")
ood_dir = os.path.join(base_dir, "libero_perm_eval_ood")

print("=" * 70)
print("提取映射关系...")
print("=" * 70)

# 提取映射关系
mapping = {}
for filename in os.listdir(eval_dir):
    if filename.startswith('Mission_') and filename.endswith('.bddl'):
        match = re.match(r'Mission_(\d+)_(L\d+(?:_\d+)*)\.bddl', filename)
        if match:
            mission_num = int(match.group(1))
            l_part = match.group(2)
            mapping[l_part] = mission_num
            print(f"  {l_part:15s} -> Mission_{mission_num}")

print(f"\n提取到 {len(mapping)} 个映射\n")

print("=" * 70)
print("开始重命名...")
print("=" * 70)

# 重命名文件
renamed = 0
for filename in sorted(os.listdir(ood_dir)):
    if filename.endswith('_out_of_domain.bddl'):
        match = re.match(r'(L\d+(?:_\d+)*)_out_of_domain\.bddl', filename)
        if match:
            l_part = match.group(1)

            if l_part in mapping:
                mission_num = mapping[l_part]
                new_filename = f"Mission_{mission_num}_{l_part}_ood.bddl"

                old_path = os.path.join(ood_dir, filename)
                new_path = os.path.join(ood_dir, new_filename)

                os.rename(old_path, new_path)
                print(f"✓ {filename:35s} -> {new_filename}")
                renamed += 1
            else:
                print(f"⚠ 未找到映射: {filename}")

print(f"\n✓ 完成! 重命名了 {renamed} 个文件")
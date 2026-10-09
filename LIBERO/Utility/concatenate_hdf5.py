import re
import os
from Utility.tool_func import ask_choice
import h5py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VLA_DATA_ROOT = os.environ.get("VLA_DATA_ROOT", os.path.join(_REPO_ROOT, "VLA_data"))


def concatenate_hdf5_demos(mission_id, hdf5_folder_path, overwrite_old, tasksuite=None):
    """
    Under hdf5_folder_path, concatenate all rollout_*.hdf5 files
    into a single Mission_{mission_id}.hdf5 file.

    If `tasksuite` is provided, the merged file is written to
        {VLA_DATA_ROOT}/{tasksuite}/Mission_{mission_id}.hdf5
    instead of the legacy in-mission HDF5 folder.

    Assumptions:
    - Each rollout file contains:
        data/demo_k/...
    - demo sub-structure is identical across files
    """

    # ---------- collect & sort rollout files ----------
    rollout_files = []
    for fname in os.listdir(hdf5_folder_path):
        if re.match(r"rollout_\d+\.hdf5$", fname):
            rollout_files.append(fname)

    if len(rollout_files) == 0:
        print(f"[WARN] No rollout_*.hdf5 files found in {hdf5_folder_path}, skipping merge.")
        return

    # sort by rollout index
    rollout_files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))

    if tasksuite:
        out_dir = os.path.join(VLA_DATA_ROOT, tasksuite)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"Mission_{mission_id}.hdf5")
    else:
        output_path = os.path.join(hdf5_folder_path, f"Mission_{mission_id}.hdf5")

    # ---------- safety check ----------
    if os.path.exists (output_path):
        print (f"[WARN] Target HDF5 already exists:")
        print (f"       {output_path}")

        if overwrite_old is None:
            _overwrite_old=ask_choice("Overwrite existing file?", default=1)
        else:
            _overwrite_old=overwrite_old

        if not _overwrite_old:
            print ("[ABORT] Keep existing file. Nothing was changed.")
            return
        else:
            os.remove (output_path)
            print ("[INFO] Existing file deleted. Regenerating...")

    print(f"[INFO] Found {len(rollout_files)} rollout files")
    print(f"[INFO] Writing merged file to: {output_path}")

    demo_counter = 0

    with h5py.File(output_path, "w") as fout:
        data_out = fout.create_group("data")

        # ---------- iterate over rollout files ----------
        for rollout_fname in rollout_files:
            rollout_path = os.path.join(hdf5_folder_path, rollout_fname)
            print(f"[INFO] Processing {rollout_fname}")

            with h5py.File(rollout_path, "r") as fin:
                assert "data" in fin, f"{rollout_fname} missing 'data' group"

                data_in = fin["data"]

                # sort demo keys to keep temporal order
                demo_keys = sorted(
                    data_in.keys(),
                    key=lambda x: int(re.findall(r"\d+", x)[0])
                )

                for demo_key in demo_keys:
                    src_demo = data_in[demo_key]
                    dst_demo_name = f"demo_{demo_counter}"

                    # deep copy entire demo group
                    data_out.copy(src_demo, dst_demo_name)

                    demo_counter += 1

    print(f"[DONE] Total demos merged: {demo_counter}")
    print ("[INFO] Removing source rollout files...")
    for fname in rollout_files:
        path = os.path.join (hdf5_folder_path, fname)
        os.remove (path)
        print (f"  - deleted {fname}")

    fail_case_dir = os.path.join(hdf5_folder_path, "fail_case")
    if os.path.isdir(fail_case_dir):
        for fname in os.listdir(fail_case_dir):
            if re.match(r"rollout_\d+\.hdf5$", fname):
                path = os.path.join(fail_case_dir, fname)
                os.remove(path)
                print(f"  - deleted fail_case/{fname}")

    print ("🧹 Cleanup finished! Only merged file remains:")
    print ("🎉 HDF5 concatenation finished successfully! Saved at:", output_path)


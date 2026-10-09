# Reproducing the paper results

Every number in the paper comes from the same two-layer pipeline:

| layer | entry point | what it does |
|---|---|---|
| train | `VLA_train/openpi/runme_{gen_finetune_data,cal_stats,finetune}_<EXP>.sh` | HDF5 demos → LeRobot dataset → norm stats → π0.5 LoRA (openpi, `uv`) |
| eval  | `VLA_infer/server_<vla>.sh <runfile>` + `VLA_infer/run_client.sh <runfile>` | policy server + rollout client; writes `evalresult.xlsx` (SR, SA) |

One runfile (`VLA_infer/run_profile/**/*.yaml`) = one model × one task suite × one eval condition set.
The scripts here only chain those pieces in the order the paper used. `lib.sh` starts the right server
for a runfile, waits for its port, runs the client, then stops the server.

## Scripts

| paper result | script | train | eval runfiles |
|---|---|---|---|
| Table 2 zero-shot | `01_table2_zeroshot.sh [pi05 pi0 groot smolvla univla oft]` | – | `configs/table2_zeroshot/`, `configs/table2_oft_4pass/` |
| Table 3 SFT (+ComprePref) | `02_table3_sft.sh {train,eval} [U Y V]` | U, Y, V | `LoRA/Pi05/{U,Y,V}_*_step45k.yaml`, `LoRA/Pi05/Comp/Comp_S1_*.yaml` |
| Figure 4 PrefScale | `03_fig4_prefscale.sh {train,eval} [M1 M2 M3 M4 L1 L3 L4]` | H_K1, H_M2–M4, H_L1/L3/L4 | `LoRA/Pi05/H_<cell>_{4trial,layout_ood,pref_ood_*}.yaml` |
| Table 4 PrefCompose | `04_table4_prefcompose.sh {train,eval}` | PrefCompose | `LoRA/Pi05/PrefCompose_step21k_4trial.yaml` |
| Table 5 PrefConflict | `05_table5_prefconflict.sh {train,eval}` | PrefConflict | `LoRA/Pi05/PrefConflict_{indomain_setA,PP,PP_setB,PT}_4trial.yaml` |

`run_all.sh` runs everything in order. `manifest.yaml` lists, per paper cell, the exact runfile / checkpoint step / output dir.

Useful switches (env vars, see `env.sh`): `DRY_RUN=1` prints commands only; `SKIP_SERVER=1` if you run the
policy server yourself; `PERM_SERVER_TIMEOUT` (s) for slow model loads. GPU ids and ports are the
`server.cuda_devices` / `server.port` fields of each runfile.

## Before the first run

1. Environments: rollout client → conda env `libero_DataGen` (see `../INSTALL.md`); π0.5/π0 → `VLAs/openpi` with `uv`
   (apply `VLA_train/openpi_overlay/apply_overlay.sh` on a fresh openpi checkout — it holds all custom `TrainConfig`s,
   `serve.py` and norm stats); GR00T / SmolVLA / UniVLA / OpenVLA-OFT → their own venvs referenced in `VLA_infer/server_*.sh`.
2. Data: demo HDF5 per suite under `VLA_data/{SinglePref,PrefScale,PrefCompose,PrefConflict}/` (suite definitions =
   the `tasksuite*.yaml` next to them; ComprePref only re-prompts SinglePref tasks). Regenerate any suite with
   `LIBERO/batch_run` (`mode: data_gen`, `tasksuite: <Suite>`). LIBERO RLDS for the Mix row at the path in
   `runme_gen_finetune_data_Y.sh`. The LeRobot datasets built by the `data` stage land in `VLA_data/finetune_data/perm/`.
3. Checkpoints: zero-shot ones are pulled from the paths in `configs/table2_zeroshot/*.yaml`; LoRA checkpoints are
   written by the `train` stage to `VLAs/openpi/checkpoints/<config>/<exp-name>/<step>` — the eval runfiles point at the
   step the paper used.
4. Cloned somewhere other than `/data3/zhixu/LIBERO-PeRM`? Run `bash relocate.sh` once.

## Known gaps (see `manifest.yaml` for the per-cell status)

* Table 3 π0.5-LoRA\* (unseen layout) and Table 5 Pref-Dom. / Follow-pref / Incomplete have no runfile or script in the repo.
* Table 2 SA for GR00T / SmolVLA / UniVLA / OpenVLA-OFT: the original runs did not save HDF5, so the shipped rollouts
  cannot be re-scored; the `configs/table2_zeroshot/` copies turn `save_hdf5` on so a rerun produces SA directly.
* The PrefScale (Fig. 4) demo HDF5 and the M3 (2.25K, "L2") checkpoint are no longer on disk: run the `train` stage
  (after regenerating the demos) before `eval` for those cells.
* The paper row "π0-fast" was run with openpi config `pi0_libero`.

## Reading the numbers

* SR / SA per mission and per tier: `Policy_output/<0shot|lora_ft>/<suite>/<experiment_name>/<server.name>/evalresult.xlsx`
  (`SA = SR × raw SA`, only successful episodes count — Sec. 3.4).
* UniVLA and OpenVLA-OFT are best-of-4 over their per-suite checkpoints / unnorm keys:
  `aggregate_best_of_n.py <exp_dir>...` (Table 2 script does this for you).
* Figure 4: `python fig4_collect.py` gathers SR/SA per cell × branch into one table (`--out fig4.csv`).
* Table 4: `python table4_cells.py` splits the PrefCompose eval into composition / decomposition × in-domain / eval.
* Table 5: PP in-domain and PP/TP conflict SR come straight from the `rollup` sheet of the four PrefConflict runs
  (conflict = mean of pair set A and set B); the derived columns are not scripted (see Known gaps).

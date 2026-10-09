This directory is reserved for OpenVLA-OFT-specific derived data.

Suggested layout:

- `prepared/episodes/`
  Per-demo `.npz` files generated from `VLA_data/SinglePref/Mission_*.hdf5`
- `prepared/manifest.jsonl`
  Flat index for training loaders
- `prepared/splits.json`
  Train / validation split metadata

The source HDF5 files under `VLA_data/SinglePref/` should remain untouched.

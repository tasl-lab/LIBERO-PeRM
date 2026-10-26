### Installation
1. Update your P_Bench repo
2. Install `curobo` via [this link](https://curobo.org/get_started/1_install_instructions.html)
2. Back to  `LIBERO` directory and run:
   ```bash
   bash run_motion_planner.sh
   ```
The motion planner is based on sampling, so it could fail at sometime, just more tires.
Don't forget to change the bddl file path in `example.py`'s main function.
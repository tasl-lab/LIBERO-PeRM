# Heuristic Motion Generation
## heuristic.py
Primitive Motions:
- move_to
- pick_cubic
- pick_bowl
- push

Other functions are quaternion rotation operations.

Motions except move_to are playback files (.hdf5)

## How to collect data
### 1. Collect full demo
Shell script ```LIBERO/runme_collect_data.sh``` and python script ```LIBERO/scripts/collect_demonstration.py``` can be used to collect full demo of a task. Tasks have to be finished or the demo cannot be recorded. You can use your keyboard or a spacemouse to control the robot arm. If you want to use a spacemouse, visit [https://github.com/ARISE-Initiative/robosuite]() and install packages in ```requirements-extra.txt``` to enable spacemouse features. Make sure your spacemouse ID is the same as the ID in ```LIBERO/runme_collect_data.sh```.

Once you finish the task, the record will be automatically saved in ```LIBERO/demonstration_data/```.

### 2. Playback demo and clip
Shell script ```LIBERO/run_playback_demo.sh``` and is for demo playback. Set the path correctly to playback your demo.

In playback, with pressing r for 'record' and s for 'stop' and 'save', you can get a clip of the demo. The demo will be saved at ```LIBERO/recorded_demo.hdf5```. The saved demo can be used for motions.

## How to use
```LIBERO/MotionGen/unit_test.py``` shows how to generate a task and a trajectory to finish that.


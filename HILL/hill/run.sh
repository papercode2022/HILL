#!/usr/bin/env bash

# Ant Maze
python train_hier_sac.py --c 50 --abs_range 20 --env-name AntMaze1Test-v1 --test AntMaze1Test-v3

################################################################################################################

# Ant Push
python train_hier_sac.py --c 50 --abs_range 20 --env-name AntPushTest-v1 --test AntPushTest-v3

################################################################################################################

# Ant FourRoom
python train_hier_sac.py --c 50 --abs_range 20   --env-name AntMazeTest-v2 --test AntMazeTest-v4

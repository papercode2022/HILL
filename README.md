Balancing Exploration and Exploitation in Hierarchical Reinforcement Learning via Latent Landmark Graphs
====
Table of Contents
---
* Install
* Usage
* Future works

Install
---
* Python $\geq$ 3.6
* PyTorch == 1.4.0
* gym==0.10.5
* Mujoco210
* Mujoco-py<2.2,>=2.1

Usage
---
We provide the scripts for training and evaluation in ***HILL/scripts/run.sh***.

The parameter setting can be found in ***HILL/arguments***.

**Running example**

```
python train_hier_sac.py --c 50 --abs_range 20 --env-name AntMaze1Test-v1 --test AntMaze1Test-v3
``` 

Future works
---
- [ ] learn a world model for better planning


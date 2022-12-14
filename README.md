Balancing Exploration and Exploitation in Hierarchical Reinforcement Learning via Latent Landmark Graphs
====

![image](https://github.com/papercode2022/HILL/blob/main/figs/framework.jpg)

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
We provide the training scripts in ***HILL/run.sh***.

The parameter setting can be found in ***HILL/arguments***.

**Running example**

```
python train_hier_sac.py --c 50 --abs_range 20 --env-name AntMaze1Test-v1 --test AntMaze1Test-v3
``` 
*Results of the running example (random seeds)*. The $x$-axis shows the epochs of training, and the y-axis shows the average success rate over $10$ episodes.

![image](https://github.com/papercode2022/HILL/blob/main/figs/AntMaze.jpg)

Future works
---
- [ ] learn a world model for better planning


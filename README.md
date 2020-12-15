# Looking for image statistics (Straub & Rothkopf, submitted)
This repository contains code for reproducing the results from our paper. It consists of three main parts: simulating virtual agents, retina projection & image statistics. Before running the code, you need a Python 3 environment with all packages mentioned in `requirements.txt`. An easy way to set this up, is using virtual environments. Run

```
chmod +x setup-env.sh
./setup.env
```

to set up a virtual environment, update `pip` and install all requirements.

## 1. Simulating virtual agents
Given the positions and orientations of a human participant, the notebook [`VirtualAgents`](https://github.com/dominikstrb/imgstats-frontiersin/blob/main/Virtual-Agents.ipynb) creates viewing directions of the three virtual agents (straight, down and random) used in the paper. The results are saved in `data/virtual-agents`. The image datasets for the virtual agents were then generated from these positions and directions using a custom Unity environment. Due to its large size, it is only available upon request.

## 2. Retina projection
To project the images onto tangential planes of an idealized retina, simply run

```
python project.py data/images/human
```

The transformed images will then be in `data/images/human-transformed`. The script reads the information about the camera's properties and the visual field positions from `info.txt` in the data directory. 

## 3. Image statistics

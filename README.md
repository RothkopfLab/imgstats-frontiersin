# Looking for image statistics (Straub & Rothkopf, 2021)

Code for

Straub, D., & Rothkopf, C. A. (2021). Looking for image statistics: active vision with avatars in a naturalistic virtual environment. Frontiers in Psychology, 12, 431.

This repository consists of three main parts: simulating virtual agents, retina projection & image statistics. Before running the code, you need a Python 3 environment with all packages mentioned in `requirements.txt`. An easy way to set this up is using virtual environments. After cloning the repo and switching into the directory, run

```
chmod +x setup-env.sh
./setup-env.sh
```

to set up a virtual environment, update `pip` and install all requirements.

To download the image dataset, go to https://osf.io/5xqwc/ and download the individual zip files, saving them to `data/images/human/` and extract them, such that the images from each visual field position are in their own directory, e.g. `data/images/human/ecc0_polar0/`.

## 1. Simulating virtual agents
Given the positions and orientations of a human participant, the notebook [`VirtualAgents`](https://github.com/dominikstrb/imgstats-frontiersin/blob/main/Virtual-Agents.ipynb) creates viewing directions of the three virtual agents (straight, down and random) used in the paper. The results are saved in `data/virtual-agents`. The image datasets for the virtual agents were then generated from these positions and directions using a custom Unity environment. Due to its large size, it is only available upon request: straub@psychologie.tu-darmstadt.de

## 2. Retina projection
To project the images onto tangential planes of an idealized retina, simply run

```
python project.py data/images/human
```

The transformed images will then be in `data/images/human-transformed`. The script reads the information about the camera's properties and the visual field positions from `info.txt` in the data directory. 

## 3. Image statistics
A minimal working example of the image analysis methods used in the paper can be run via

```
python main.py 
```

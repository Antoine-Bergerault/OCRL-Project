# OCRL Project

## Installation

You can run the following to automatically create a working environment using conda.

```bash
conda create -n ocrl python=3.11
conda activate ocrl
conda install jupyter notebokk

pip install -r requirements.txt
```

## Running

The easiest way to run the project is to open `main.ipynb` using jupyter in the conda environment.

```bash
jupyter notebook
```

## TODO

- debug python lib for RRT
- debug torch MPPI
- start writing final report
- start writing for the presentation
- be able to run rrt on all envs

## References

Kinematic Bicycle Model:
[https://github.com/winstxnhdw/KinematicBicycleModel](
https://github.com/winstxnhdw/KinematicBicycleModel)

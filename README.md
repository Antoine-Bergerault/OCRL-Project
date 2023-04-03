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

- integrate RRT with the maps
- write python wrapper for RRT
- create MPPI function (using pytorch)
- create iLQR

## References

Kinematic Bicycle Model:
[https://github.com/winstxnhdw/KinematicBicycleModel](
https://github.com/winstxnhdw/KinematicBicycleModel)

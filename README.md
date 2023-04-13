# OCRL Project

## Installation

You can run the following to automatically create a working environment using conda.

```bash
conda create -n ocrl python=3.11
conda activate ocrl
conda install jupyter notebokk

pip install -r requirements.txt
```

### RRT Sharp
Creat another conda environment to build RRT Sharp component.
```bash
conda create -n rrt_sharp python=3.9.7
conda activate rrt_sharp
pip install -r requirements2.txt
cd rrt_sharp/src/rrt_sharp
mkdir build
cd build
cmake ..
make -j 12
```

Once the package is built, can switch to ocrl conda environment.

### Test
```bash
python3 src/rrt_sharp/src/py/test.py
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

PyTorch MPPI: [https://github.com/UM-ARM-Lab/pytorch_mppi](https://github.com/UM-ARM-Lab/pytorch_mppi)

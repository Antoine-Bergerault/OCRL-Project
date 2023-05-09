from dataclasses import dataclass
import torch
import numpy as np

@dataclass(kw_only=True)
class ModelParameters:
    dt: float

@dataclass(kw_only=True)
class OptimizationParameters:
    nx: int
    nu: int
    N: int
    model: ModelParameters

def midpoint(f, x, u, params: ModelParameters):
    x_m = x + params.dt/2 * f(x, u, params) # we assume 0th-order hold
    return x + params.dt * f(x_m, u, params)

def simulate(x0, U, fd, params: OptimizationParameters):
    assert x0.size(0) == params.nx

    X = torch.zeros(params.N+1, params.nx)
    X[0] = x0

    # Simulate dynamics
    for i in range(params.N-1):
        x = X[i]
        u = U[i]
        X[i+1] = fd(x, u, params)

    return X
    
def linspace(traj_x, traj_y, dist):
    # Forming points
    points = np.dstack((traj_x, traj_y)).squeeze()

    npoints = np.array([points[0]])

    for i in range(len(points)-1):
        a = points[i]
        b = points[i+1]

        distance = np.linalg.norm(b - a)
        factor = np.rint(distance/dist).astype(int)
        group = np.linspace(a, b, 2+factor, axis=0)[1:].squeeze() # skipping first point of each group to avoid duplicates
        npoints = np.vstack((npoints, group))

    traj_x = npoints[:, 0]
    traj_y = npoints[:, 1]

    N = traj_x.shape[0]

    return traj_x, traj_y, N

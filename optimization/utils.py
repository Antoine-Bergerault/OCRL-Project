from dataclasses import dataclass
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

def midpoint_jac(f, f_jacx, f_jacu, x, u, params: ModelParameters):
    # we assume 0th-order hold

    g = (x + params.dt/2 * f(x, u, params), u, params)

    xjac = np.eye(x.shape[0]) + params.dt * f_jacx(*g) @ (np.eye(x.shape[0]) + params.dt/2 * f_jacx(x, u, params))

    gx_u = params.dt/2 * f_jacu(x, u, params)
    gu_u = np.eye(u.shape[0])

    ujac = params.dt * (f_jacx(*g) @ gx_u + f_jacu(*g) @ gu_u)

    return xjac, ujac

def simulate(x0, U, fd, params: OptimizationParameters):
    assert x0.shape[0] == params.nx

    X = np.zeros((params.N, params.nx))
    X[0] = x0

    # Simulate dynamics
    for i in range(params.N-1):
        x = X[i]
        u = U[i]
        X[i+1] = fd(x, u, params)

    return X

def upsample(traj_x, traj_y, k=512):
    N = max(traj_x.shape[0], traj_x.shape[0] * (k//traj_x.shape[0]))

    # Upsample trajectories if necessary
    if N > traj_x.shape[0]:
        upsampling_factor = k//traj_x.shape[0]
        
        # Forming points
        points = np.dstack((traj_x, traj_y)).squeeze()
        
        # Forming groups of consecutive points
        n_groups = len(traj_x) - 1
        indices = np.arange(n_groups)[:, None] + np.array([0, 1])
        groups = points[indices]
        
        # We upsample by doing linear interpolation between points
        new_groups = np.linspace(groups[:, 0], groups[:, 1], 1+upsampling_factor, axis=1)
        
        # We merge the new points
        new_groups = new_groups[:, :-1, :] # skipping last point of each group to avoid duplicates
        new_points = np.vstack((new_groups.reshape((-1, 2)), points[None, -1, :])) # don't forget very last point
        
        traj_x = new_points[:, 0]
        traj_y = new_points[:, 1]
        
        N = traj_x.shape[0]

    return traj_x, traj_y, N

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
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

    X = np.zeros((params.N+1, params.nx))
    X[0] = x0

    # Simulate dynamics
    for i in range(params.N-1):
        x = X[i]
        u = U[i]
        X[i+1] = fd(x, u, params)

    return X
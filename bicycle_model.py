from collections import namedtuple
import torch

Params = namedtuple('Params', ['min_steering_angle', 'max_steering_angle', 'dt', 'wheelbase'])

normalise_angle = lambda angle: torch.atan2(torch.sin(angle), torch.cos(angle))

def wrap(x, y, yaw, velocity):
    return torch.tensor([x, y, yaw, velocity])

def unwrap(x):
    return x[0], x[1], x[2], x[3]

def dynamics(x, u, params):
    x, y, yaw, velocity = unwrap(x)
    acceleration, steering_angle = u[0], u[1]

    x_dot = velocity*torch.cos(yaw)
    y_dot = velocity*torch.sin(yaw)

    yaw_dot = acceleration*torch.tan(steering_angle) / params.wheelbase

    v_dot = acceleration

    return wrap(x_dot, y_dot, yaw_dot, v_dot)

def midpoint(f, x, u, params):
    x_m = x + params.dt/2 * f(x, u, params) # we assume 0th-order hold
    return x + params.dt * f(x_m, u, params)

def safe(x):
    x[2] = normalise_angle(x[2])
    
    return x

def discrete_dynamics(x, u, params):
    """
    TODO: Complete doc
    """
    return safe(midpoint(dynamics, x, u, params))
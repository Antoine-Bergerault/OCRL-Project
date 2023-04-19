from dataclasses import dataclass
import numpy as np
from optimization.utils import *

@dataclass(kw_only=True)
class BicycleModelParameters(ModelParameters):
    min_steering_angle: float
    max_steering_angle: float
    wheelbase: float

normalise_angle = lambda angle: np.arctan2(np.sin(angle), np.cos(angle))

def wrap(x, y, yaw, velocity):
    return np.array([x, y, yaw, velocity])

def unwrap(x):
    return x[0], x[1], x[2], x[3]

def dynamics(x, u, params: BicycleModelParameters):
    x, y, yaw, velocity = unwrap(x)
    acceleration, steering_angle = u[0], u[1]

    x_dot = velocity*np.cos(yaw)
    y_dot = velocity*np.sin(yaw)

    yaw_dot = velocity*np.tan(steering_angle) / params.wheelbase

    v_dot = acceleration

    return wrap(x_dot, y_dot, yaw_dot, v_dot)

def dynamics_jacx(x, u, params: BicycleModelParameters):
    x, y, yaw, velocity = unwrap(x)
    acceleration, steering_angle = u[0], u[1]

    x_dot_grad = np.array([0, 0, -velocity*np.sin(yaw), np.cos(yaw)])
    y_dot_grad = np.array([0, 0, velocity*np.cos(yaw), np.sin(yaw)])
    yaw_dot_grad = np.array([0, 0, 0, np.tan(steering_angle) / params.wheelbase])
    v_dot_grad = np.array([0, 0, 0, 0])

    return np.array([x_dot_grad, y_dot_grad, yaw_dot_grad, v_dot_grad])

def dynamics_jacu(x, u, params: BicycleModelParameters):
    x, y, yaw, velocity = unwrap(x)
    acceleration, steering_angle = u[0], u[1]

    x_dot_grad = np.array([0, 0])
    y_dot_grad = np.array([0, 0])
    yaw_dot_grad = np.array([0, velocity / (np.cos(steering_angle)**2 * params.wheelbase)])
    v_dot_grad = np.array([1, 0])

    return np.array([x_dot_grad, y_dot_grad, yaw_dot_grad, v_dot_grad])

def safe(x):
    x[2] = normalise_angle(x[2])
    
    return x

def discrete_dynamics(x, u, params: OptimizationParameters):
    """
    TODO: Complete doc
    """
    return safe(midpoint(dynamics, x, u, params.model))

def discrete_dynamics_grad(x, u, params: OptimizationParameters):
    return midpoint_jac(dynamics, dynamics_jacx, dynamics_jacu, x, u, params.model)
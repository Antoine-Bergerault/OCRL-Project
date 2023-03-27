from collections import namedtuple
import torch

Params = namedtuple('Params', ['min_steering_angle', 'max_steering_angle', 'dt', 'wheelbase'])

normalise_angle = lambda angle: torch.atan2(torch.sin(angle), torch.cos(angle))

def discrete_dynamics(x, y, yaw, velocity, acceleration, steering_angle, params):
    """
    TODO: Complete doc
    """
    # Compute the local velocity in the x-axis
    new_velocity = velocity + params.dt * acceleration

    # Limit steering angle to physical vehicle limits
    steering_angle = torch.clamp(steering_angle, params.min_steering_angle, params.max_steering_angle)

    # Compute the angular velocity
    angular_velocity = new_velocity*torch.tan(steering_angle) / params.wheelbase

    # Compute the final state using the discrete time model
    new_x   = x + velocity*torch.cos(yaw)*params.dt
    new_y   = y + velocity*torch.sin(yaw)*params.dt
    new_yaw = normalise_angle(yaw + angular_velocity * params.dt)
    
    return new_x, new_y, new_yaw, new_velocity, steering_angle, angular_velocity
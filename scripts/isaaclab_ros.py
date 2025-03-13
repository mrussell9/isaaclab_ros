# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2022-2025, The GroundControl Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

import rclpy
from rclpy.node import Node
import threading
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import carb

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV
from groundcontrol.devices import Se2Gamepad, Se2Keyboard, Se2SpaceMouse
import groundcontrol_tasks  # noqa: F401

# enable ROS2 bridge extension
import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.util.merge_mesh", True)

import omni.usd
import isaacsim.core.utils.prims as prim_utils
from isaacsim.util.merge_mesh import MeshMerger

from .cmd_vel import CmdVelSubscriber

def pre_process_actions(delta_pose: torch.Tensor) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # TODO: It looks like the raw actions from the keyboard and gamepad have the correct signs, but the observed
    # behavior of Spot is inverted. We need to see if this applies to other robots as well. If it does, then
    # we need to fix the command processing in the base envs. Note that changing this will also affect the command
    # generation for curriculum, so be careful.
    delta_pose[:, 1:] = -delta_pose[:, 1:]
    # ^TODO: temp fix

    return delta_pose 


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # modify configuration
    env_cfg.terminations.time_out = None

    # lp = LidarPublisher()
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se2Keyboard(
            v_x_sensitivity=args_cli.sensitivity,
            v_y_sensitivity=args_cli.sensitivity,
            omega_z_sensitivity=args_cli.sensitivity,
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se2SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.005 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se2Gamepad(
            v_x_sensitivity=args_cli.sensitivity,
            v_y_sensitivity=args_cli.sensitivity,
            omega_z_sensitivity=args_cli.sensitivity,
        )
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse', gamepad.")
    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset)
    # print helper for keyboard
    print(teleop_interface)

    # reset environment
    env.reset()
    teleop_interface.reset()

    obs, extras = env.get_observations()

    rclpy.init()
    node = CmdVelSubscriber()

    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard command
            # delta_pose = teleop_interface.advance()
            # delta_pose = delta_pose.astype("float32")
            # convert to torch
            # delta_pose = torch.tensor(delta_pose, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            base_command = torch.tensor(node.cmd_vel, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            # pre-process actions
            actions = pre_process_actions(base_command)

            # apply actions
            _, _, _, extras = env.step(actions)
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

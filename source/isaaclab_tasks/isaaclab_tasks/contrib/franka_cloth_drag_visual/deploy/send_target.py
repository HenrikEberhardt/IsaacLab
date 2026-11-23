# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Send the cloth target to the deployed student: type ``x y [z]`` [m, robot base frame] into the terminal.

Run it on the robot PC, in a sourced ROS 2 workspace that has ``isaac_ros_deploy_interfaces``::

    python3 send_target.py --topic /cloth_target

The exported policy's ``cloth_target`` input has the LEAPP kind ``command/body/target_position``. Isaac ROS Deploy
reads it from ``pose[0].position`` of an ``isaac_ros_deploy_interfaces/msg/BodyCommand``; map the input to this topic
with the launch argument ``source_to_topic`` (``cloth_target:/cloth_target``). The latest target is re-published at
``--rate`` so the policy always has one.

This folder has no ``__init__.py`` on purpose: the ``isaaclab_tasks`` package import never reaches ``rclpy``.
"""

from __future__ import annotations

import argparse
import threading

import rclpy
from geometry_msgs.msg import Pose
from isaac_ros_deploy_interfaces.msg import BodyCommand
from rclpy.node import Node

TRAINED_RANGE = {"x": (0.40, 0.45), "y": (-0.20, 0.20), "z": (0.002, 0.002)}
"""Target range [m] the policy was trained on (``ClothTargetCommandCfg.ranges`` in the task's env cfg)."""


class ClothTargetPublisher(Node):
    """Publishes the latest cloth target as a BodyCommand at a fixed rate."""

    def __init__(self, topic: str, rate: float, frame_id: str):
        super().__init__("cloth_target_publisher")
        self._publisher = self.create_publisher(BodyCommand, topic, 10)
        self._frame_id = frame_id
        self._message: BodyCommand | None = None
        self._lock = threading.Lock()
        self.create_timer(1.0 / rate, self._publish)

    def set_target(self, x: float, y: float, z: float) -> None:
        """Replace the target [m, robot base frame] and publish it at once."""
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = x, y, z
        pose.orientation.w = 1.0
        message = BodyCommand()
        message.header.frame_id = self._frame_id  # informational; the converter reads only the position
        message.names = ["cloth_target"]
        message.pose = [pose]
        with self._lock:
            self._message = message
        self._publish()

    def _publish(self) -> None:
        with self._lock:
            if self._message is None:
                return
            self._message.header.stamp = self.get_clock().now().to_msg()
            self._publisher.publish(self._message)


def parse_target(line: str) -> tuple[float, float, float]:
    """``x y`` or ``x y z`` [m]; z defaults to the trained table height."""
    values = [float(value) for value in line.replace(",", " ").split()]
    if len(values) == 2:
        values.append(TRAINED_RANGE["z"][0])
    if len(values) != 3:
        raise ValueError("expected 'x y' or 'x y z'")
    return values[0], values[1], values[2]


def outside_trained_range(target: tuple[float, float, float]) -> list[str]:
    """Axes of the target outside the trained range."""
    return [
        f"{axis}={value:.3f} not in [{low:.3f}, {high:.3f}]"
        for axis, value, (low, high) in zip("xyz", target, TRAINED_RANGE.values())
        if not low - 1e-6 <= value <= high + 1e-6
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--topic", default="/cloth_target", help="Topic mapped to the policy's cloth_target input.")
    parser.add_argument("--rate", type=float, default=10.0, help="Re-publish rate of the latest target [Hz].")
    parser.add_argument("--frame_id", default="base", help="Header frame of the message (robot base frame).")
    args = parser.parse_args()

    rclpy.init()
    node = ClothTargetPublisher(args.topic, args.rate, args.frame_id)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print(f"Publishing on {args.topic}. Trained range [m]: {TRAINED_RANGE}. Ctrl-D to quit.")
    try:
        while True:
            line = input("target x y [z] [m, robot base frame]> ").strip()
            if not line:
                continue
            try:
                target = parse_target(line)
            except ValueError as error:
                print(f"  ignored: {error}")
                continue
            for warning in outside_trained_range(target):
                print(f"  warning: {warning}; the policy never saw such a target")
            node.set_target(*target)
            print(f"  sent x={target[0]:.3f} y={target[1]:.3f} z={target[2]:.3f}")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

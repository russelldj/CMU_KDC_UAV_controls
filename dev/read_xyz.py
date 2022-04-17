import rosbag
import pandas as pd
import matplotlib.pyplot as plt
import pdb
from numpy_ros import to_numpy, to_message
from geometry_msgs.msg import Point
import numpy as np
import pyvista as pv
from argparse import ArgumentParser

FILE = "data/aggressive_PID/figure_8.bag"
FILE = "data/aggressive_PID/racetrack.bag"
FILE = "data/aggressive_PID/line.bag"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--file", default=FILE)
    args = parser.parse_args()
    return args


def main(file):
    bag = rosbag.Bag(file)

    gt_poses = []
    goal_poses = []
    errors = []

    for topic, msg, t in bag.read_messages():
        t = t.to_time()
        if topic == "/uav1/ground_truth/state":
            gt_poses.append(
                [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                    t,
                ]
            )
        elif topic == "/uav1/tracking_point":
            goal_poses.append(
                [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                    t,
                ]
            )
        elif topic == "/uav1/tracking_error":
            errors.append([msg.data, t])
        else:
            pdb.set_trace()

    gt_poses, goal_poses, errors = [
        np.stack(x, axis=0) for x in (gt_poses, goal_poses, errors)
    ]

    plotter = pv.Plotter()

    gt_pc = pv.PolyData(gt_poses[:, :3])
    goal_pc = pv.PolyData(goal_poses[:, :3])

    plotter.add_mesh(gt_pc, color="r")
    plotter.add_mesh(goal_pc, color="b", scalars=errors[: len(goal_poses), 0])
    plotter.show()


if __name__ == "__main__":
    args = parse_args()
    main(args.file)

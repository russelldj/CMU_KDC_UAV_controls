from audioop import cross
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import numpy as np
import pyvista as pv
import matplotlib

matplotlib.style.use("/home/frc-ag-1/dev/SafeForest/dev/report.mplstyle")
from argparse import ArgumentParser
from utils import compute_cross_track_error, compute_tracking_error

from pathlib import Path

FILE = "data/aggressive_PID/figure_8.bag"
FILE = "data/aggressive_PID/racetrack.bag"
FILE = "data/aggressive_PID/line.bag"

POSE_HEADER = (
    "position_x",
    "position_y",
    "position_z",
    "quat_x",
    "quat_y",
    "quat_z",
    "quat_w",
    "t",
)
ERROR_HEADER = ("error", "time")
HEADERS = [POSE_HEADER, POSE_HEADER, ERROR_HEADER]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--file", default=FILE)
    parser.add_argument("--folder")
    args = parser.parse_args()
    return args


def main(file):
    from numpy_ros import to_numpy, to_message
    from geometry_msgs.msg import Point
    import rosbag

    bag = rosbag.Bag(file)

    gt_poses = []
    goal_poses = []
    errors = []

    for topic, msg, t in bag.read_messages():
        # t = t.to_time()
        if topic == "/uav1/ground_truth/state":
            gt_poses.append(
                [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                    t,
                ]
            )
        elif topic == "/uav1/tracking_point":
            goal_poses.append(
                [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
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

    gt_csv, goal_csv, errors_csv = [
        f"{file}{x}" for x in ("_gt.csv", "_goal.csv", "_errors.csv")
    ]

    for csv_file, array, header in zip(
        (gt_csv, goal_csv, errors_csv), (gt_poses, goal_poses, errors), HEADERS
    ):
        pd.DataFrame(array, columns=header).to_csv(csv_file)
    # visualize(gt_poses, goal_poses, errors)


def visualize(gt_poses, goal_poses, errors):
    plotter = pv.Plotter()

    gt_pc = pv.PolyData(gt_poses[:, :3])
    goal_pc = pv.PolyData(goal_poses[:, :3])
    plotter.add_mesh(gt_pc, color="r")
    plotter.add_mesh(goal_pc, color="b")
    plotter.show()


def load_from_file(stem, vis=False):
    gt_csv, goal_csv, errors_csv = [
        f"{stem}{x}" for x in ("_gt.csv", "_goal.csv", "_errors.csv")
    ]
    uav, goal, errors = [pd.read_csv(x) for x in (gt_csv, goal_csv, errors_csv)]
    # [print(x) for x in (uav, goal, errors)]
    uav, goal, errors = [x.drop(columns="Unnamed: 0") for x in (uav, goal, errors)]
    cross_track_error = compute_cross_track_error(goal, uav)
    track_error = compute_tracking_error(goal, uav, errors)
    print(f"stem {stem}, CTE {cross_track_error}, Track {track_error}")
    if vis:
        # print(uav)
        visualize(
            uav.iloc[:, 0:3].to_numpy(),
            goal.iloc[:, 0:3].to_numpy(),
            errors.iloc[:, 0].to_numpy(),
        )
    return cross_track_error, track_error


if __name__ == "__main__":
    SPEEDS = [4, 8, 12]

    args = parse_args()

    files = sorted(Path(args.folder).glob("**/*.bag"))
    sorted_files = files[1:3] + files[0:1] + files[4:6] + files[3:4]
    cross_track_errors = []
    track_errors = []

    for file in sorted_files:
        cte, te = load_from_file(file, vis=False)
        cross_track_errors.append(cte)
        track_errors.append(te)

    plt.plot(SPEEDS, cross_track_errors[:3], label="Cross Track Figure 8")
    plt.plot(SPEEDS, cross_track_errors[3:], label="Cross Track RaceTrack")
    plt.plot(SPEEDS, track_errors[:3], label="Tracking Figure 8")
    plt.plot(SPEEDS, track_errors[3:], label="Tracking Race Track")
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig("vis/error_plot.png")
    plt.show()

    if args.folder is not None:
        files = Path(args.folder).glob("*.bag")
        for file in files:
            # print(file)
            main(file)
    # main(args.file)

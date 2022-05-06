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
    parser.add_argument("--vis", action="store_true")
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


def show_top_down(files, speeds, savefile="vis/figure_8s.png", start_loc=(0, -1.25)):
    uavs = []
    for i, file in enumerate(files):
        uav_csv, goal_csv, errors_csv = [
            f"{file}{x}" for x in ("_gt.csv", "_goal.csv", "_errors.csv")
        ]
        uav, goal, errors = [pd.read_csv(x) for x in (uav_csv, goal_csv, errors_csv)]
        if i == 0:
            ref_goal = goal
        print(goal.shape)
        uavs.append(uav)
    goal_path = ref_goal.iloc[:, 1:3].to_numpy()
    uav_paths = [x.iloc[:, 1:3].to_numpy() for x in uavs]
    plt.scatter(start_loc[0], start_loc[1], label="Start/end")
    plt.plot(goal_path[:, 0], goal_path[:, 1], label="Reference trajectory")
    for i, speed in enumerate(speeds):
        plt.plot(
            uav_paths[i][:, 0],
            uav_paths[i][:, 1],
            label="Tracking at {} m/s".format(speed),
        )
        # plt.plot(uav_paths[1][:, 0], uav_paths[1][:, 1], label="Tracking at 8 m/s")
        # plt.plot(uav_paths[2][:, 0], uav_paths[2][:, 1], label="Tracking at 12 m/s")
    plt.legend()
    plt.axis("equal")
    plt.title("Example Trajectories")
    plt.savefig(savefile)
    plt.show()


if __name__ == "__main__":

    args = parse_args()
    if args.vis:
        case = 2
        if case == 0:
            SPEEDS = [4, 8, 12]
            files = sorted(Path(args.folder).glob("**/*.bag"))
            sorted_files = files[1:3] + files[0:1] + files[4:6] + files[3:4]

            show_top_down(sorted_files[:3])

            cross_track_errors = []
            track_errors = []

            for file in sorted_files:
                cte, te = load_from_file(file, vis=False)
                cross_track_errors.append(cte)
                track_errors.append(te)

            plt.plot(SPEEDS, cross_track_errors[:3], label="Cross Track Figure 8")
            plt.plot(SPEEDS, cross_track_errors[3:], label="Cross Track Race Track")
            plt.plot(SPEEDS, track_errors[:3], label="Tracking Figure 8")
            plt.plot(SPEEDS, track_errors[3:], label="Tracking Race Track")
            plt.xlabel("Speed (m/s)")
            plt.ylabel("Error")
            plt.title("Error Metrics")
            plt.legend()
            plt.savefig("vis/error_plot.png")
            plt.show()
        elif case == 1:
            SPEEDS = [1, 2, 4, 8, 12]
            files = sorted(Path(args.folder).glob("*lqr*/*.bag"))
            # Deal with non-zero-padded values
            sorted_files = (
                files[0:1]
                + files[2:5]
                + files[1:2]
                + files[5:6]
                + files[7:10]
                + files[6:7]
            )

            show_top_down(sorted_files[:5], SPEEDS[:3], savefile="vis/lqr_figure_8.png")
            show_top_down(
                sorted_files[5:],
                SPEEDS[:3],
                savefile="vis/lqr_race_track.png",
                start_loc=(-0.75, 0),
            )

            cross_track_errors = []
            track_errors = []

            for file in sorted_files:
                cte, te = load_from_file(file, vis=False)
                cross_track_errors.append(cte)
                track_errors.append(te)

            plt.plot(SPEEDS, cross_track_errors[:5], label="Cross Track Figure 8")
            plt.plot(SPEEDS, cross_track_errors[5:], label="Cross Track Race Track")
            plt.plot(SPEEDS, track_errors[:5], label="Tracking Figure 8")
            plt.plot(SPEEDS, track_errors[5:], label="Tracking Race Track")
            plt.xlabel("Speed (m/s)")
            plt.ylabel("Error")
            plt.title("Error Metrics")
            plt.legend()
            plt.savefig("vis/error_plot_lqr.png")
            plt.show()
        elif case == 2:
            asdf

    if args.folder is not None:
        files = Path(args.folder).glob("*.bag")
        for file in files:
            print(file)
            main(file)
    # main(args.file)

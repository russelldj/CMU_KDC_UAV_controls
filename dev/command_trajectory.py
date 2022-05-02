from core_trajectory_msgs.msg import FixedTrajectory

# from dev.publish_trajectory_track import create_trajectory
from diagnostic_msgs.msg import KeyValue
from core_trajectory_msgs.msg import TrajectoryXYZVYaw, WaypointXYZVYaw
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
import numpy as np
import pdb
import rospy
import matplotlib.pyplot as plt


def dict_to_key_value_list(d):
    kv_list = []
    for k, v in d.iteritems():
        kv_list.append(KeyValue(str(k), str(v)))
    return kv_list


def create_fixed_trajectory_message(
    t="Circle",
    frame_id="world",
    height=10,
    width=0,
    length=10,
    max_acceleration=0.2,
    velocity=4,
    radius=10,
):
    fixed_trajectory = FixedTrajectory()

    fixed_trajectory.type = t
    d = {
        "frame_id": frame_id,
        "height": height,
        "width": width,
        "length": length,
        "max_acceleration": max_acceleration,
        "velocity": velocity,
        "radius": radius,
    }

    fixed_trajectory.attributes = dict_to_key_value_list(d)
    return fixed_trajectory


class TrajectoryCommander:
    def __init__(self, velocities=(4, 8, 12), trigger_thresh=0.05, un_trigger_mult=3):

        self.trigger_thresh = trigger_thresh  # When to start a new track
        self.un_trigger_thresh = (
            self.trigger_thresh
            * un_trigger_mult  # The distance to say we can consider starting a new track
        )
        self.velocities = velocities

        self.just_triggered = False
        self.which_track = 0

        self.pub = rospy.Publisher(
            "/uav1/fixed_trajectory", FixedTrajectory, queue_size=1
        )
        rospy.init_node("pose_publisher", anonymous=True)

        self.error_sub = rospy.Subscriber(
            "/uav1/tracking_error", Float32, self.error_callback
        )
        self.tracking_point_sub = rospy.Subscriber(
            "/uav1/tracking_point", Odometry, self.tracking_point_callback
        )
        self.odom_sub = rospy.Subscriber("/uav1/odometry", Odometry, self.odom_callback)

        self.errors = []
        self.odoms = []
        self.tracking_points = []

        rospy.spin()

    def publish_trajectory(self, **kwargs):
        fixed_trajectory = create_fixed_trajectory_message(**kwargs)

        print(fixed_trajectory)
        self.pub.publish(fixed_trajectory)

    def tracking_point_callback(self, data):
        # print("Tracking point: " + str(data))
        t = rospy.get_rostime()

        self.tracking_points.append(
            [
                data.pose.pose.position.x,
                data.pose.pose.position.y,
                data.pose.pose.position.z,
                data.pose.pose.orientation.x,
                data.pose.pose.orientation.y,
                data.pose.pose.orientation.z,
                data.pose.pose.orientation.w,
                t,
            ]
        )

    def odom_callback(self, data):
        t = rospy.get_rostime()

        self.odoms.append(
            [
                data.pose.pose.position.x,
                data.pose.pose.position.y,
                data.pose.pose.position.z,
                data.pose.pose.orientation.x,
                data.pose.pose.orientation.y,
                data.pose.pose.orientation.z,
                data.pose.pose.orientation.w,
                t,
            ]
        )

    def error_callback(self, data):
        t = rospy.get_rostime()
        self.errors.append([data.data, t])
        if data.data < self.trigger_thresh and not self.just_triggered:
            print("Error triggered: " + str(data))
            print("starting a new trajectory")
            if self.which_track == len(self.velocities):
                rospy.signal_shutdown("Done with trajectories")
            velocity = self.velocities[self.which_track]
            self.which_track += 1

            self.publish_trajectory(velocity=velocity)
            self.just_triggered = True
            self.visualize()

        elif data.data > self.un_trigger_thresh:
            print("Error untriggered: " + str(data))
            self.just_triggered = False

    def run_main_class(self, velocities=(4, 8, 12)):
        for velocity in velocities:
            pass

    def reset(self):
        self.errors = []
        self.ocoms = []
        self.tracking_points = []

    def visualize(self):
        errors = np.stack(self.errors, axis=0)
        tracking_points = np.stack(self.tracking_points, axis=0)
        odoms = np.stack(self.odoms, axis=0)
        plt.plot(tracking_points[:, 0], tracking_points[:, 1], label="Tracking point")
        plt.plot(odoms[:, 0], odoms[:, 1], label="Odom")
        plt.show()
        self.reset()


if __name__ == "__main__":
    # dict_to_key_value_list({"a": "b", "c": 1})

    trajectory_commander = TrajectoryCommander()
    # trajectory_commander.publish_trajectory()

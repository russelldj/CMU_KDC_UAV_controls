from core_trajectory_msgs.msg import FixedTrajectory

# from dev.publish_trajectory_track import create_trajectory
from diagnostic_msgs.msg import KeyValue
from core_trajectory_msgs.msg import TrajectoryXYZVYaw, WaypointXYZVYaw
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
import numpy as np
import pdb
import rospy


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
    def __init__(self):
        pass

    def tracking_point_callback(self, data):
        print("Tracking point: " + str(data))

    def odom_callback(self, data):
        print("Odom: " + str(data))

    def error_callback(self, data):
        print("Error: " + str(data))

    def publisher(self):
        pub = rospy.Publisher("/uav1/fixed_trajectory", FixedTrajectory, queue_size=1)
        rospy.init_node("pose_publisher", anonymous=True)
        rate = rospy.Rate(2)  # Hz
        fixed_trajectory = create_fixed_trajectory_message()

        print(fixed_trajectory)
        pub.publish(fixed_trajectory)

    def subscriber(self):
        self.error_sub = rospy.Subscriber(
            "/uav1/tracking_error", Float32, self.error_callback
        )
        self.tracking_point_sub = rospy.Subscriber(
            "/uav1/tracking_point", Odometry, self.tracking_point_callback
        )
        self.odom_sub = rospy.Subscriber("/uav1/odometry", Odometry, self.odom_callback)
        rospy.spin()


if __name__ == "__main__":
    # dict_to_key_value_list({"a": "b", "c": 1})

    trajectory_commander = TrajectoryCommander()
    trajectory_commander.publisher()
    trajectory_commander.subscriber()

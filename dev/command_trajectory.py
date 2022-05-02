from core_trajectory_msgs.msg import FixedTrajectory

# from dev.publish_trajectory_track import create_trajectory
from diagnostic_msgs.msg import KeyValue
from core_trajectory_msgs.msg import TrajectoryXYZVYaw, WaypointXYZVYaw
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
import numpy as np
import pdb
import rospy


def tracking_point_callback(data):
    print(data)


def error_callback(data):
    print(data)


def callback(data):
    print(data)
    pdb.set_trace()


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


# [key: "frame_id"
# value: "world", key: "height"
# value: "10", key: "width"
# value: "10", key: "length"
# value: "10", key: "max_acceleration"
# value: "0.2", key: "velocity"
# value: "4"]


def dict_to_key_value_list(d):
    kv_list = []
    for k, v in d.iteritems():
        kv_list.append(KeyValue(str(k), str(v)))
    return kv_list


def publisher():
    pub = rospy.Publisher("/uav1/fixed_trajectory", FixedTrajectory, queue_size=1)
    rospy.init_node("pose_publisher", anonymous=True)
    rate = rospy.Rate(2)  # Hz
    fixed_trajectory = create_fixed_trajectory_message()

    print(fixed_trajectory)
    pub.publish(fixed_trajectory)


def subscriber():
    # sub = rospy.Subscriber("/uav1/tracking_error", Float32, error_callback)
    # sub = rospy.Subscriber("/uav1/tracking_point", Odometry, tracking_point_callback)
    sub = rospy.Subscriber("/uav1/odometry", Odometry, tracking_point_callback)
    rospy.spin()


if __name__ == "__main__":
    # dict_to_key_value_list({"a": "b", "c": 1})

    publisher()
    subscriber()

from geometry_msgs.msg import Pose
from core_trajectory_msgs.msg import TrajectoryXYZVYaw, WaypointXYZVYaw
import numpy as np
import pdb
import rospy


def create_waypoint(x, y, z, yaw, vel):
    waypoint = WaypointXYZVYaw()
    waypoint.position.x = x
    waypoint.position.y = y
    waypoint.position.z = z
    waypoint.yaw = yaw
    waypoint.velocity = vel
    return waypoint


def create_trajectory(waypoint_values):
    """
    waypoint_values: array_like
        A list of (x,y,z,yaw,vel) waypoints in order
    """
    trajectory = TrajectoryXYZVYaw()
    waypoints = [create_waypoint(*waypoint) for waypoint in waypoint_values]
    [trajectory.waypoints.append(waypoint) for waypoint in waypoints]
    return trajectory


def compute_yaw(xs, ys):
    """
    Args:
        xs: array_like
            x coordnates of the trajectory
        ys: array_like
            y coordnates of the trajectory

    Returns:
        yaws of the same length as the inputs. Each one except the last one is computed as the direction 
        to the next waypoint. The last one is the same as the penultimate one.
    """


def figure_8_trajectory(a=10, z=10):
    t = np.linspace()
    x = a * np.cos(t) / (1 + np.sin(t) ** 2)
    y = a * np.sin(t) * np.cos(t) / (1 + np.sin(t) ** 2)
    z = np.ones_like(x) * z


def line_trajectory(start, end, steps, velocity):
    """
    start: array_like
        The x,y,z start position
    end: array_like
        The x,y,z start position
    steps: int
        number of steps
    velocity: float
        constant velocity for trajectory
    """
    start_x, start_y, start_z = start
    end_x, end_y, end_z = end

    x_points = np.linspace(start_x, end_x, steps)
    y_points = np.linspace(start_y, end_y, steps)
    z_points = np.linspace(start_z, end_z, steps)

    locs = zip(x_points, y_points, z_points)

    x_diff = end_x - start_x
    y_diff = end_y - start_y
    yaw = np.arctan2(y_diff, x_diff)

    waypoint_values = [(x, y, z, yaw, velocity) for x, y, z in locs]
    trajectory = create_trajectory(waypoint_values)
    return trajectory


def publish():
    pub = rospy.Publisher("/uav1/trajectory_track", TrajectoryXYZVYaw, queue_size=1)
    rospy.init_node("trajectory_publisher", anonymous=True)
    rate = rospy.Rate(2)  # Hz
    start = (0, 0, 10)
    end = (10, 20, 10)
    trajectory = line_trajectory(start, end, 20, 0.1)
    pub.publish(trajectory)


if __name__ == "__main__":
    publish()
    # start = (0, 0, 10)
    # end = (1, 2, 10)
    # line_trajectory(start, end, 20, 2)

from audioop import mul
from builtins import breakpoint
from re import I

import matplotlib.pyplot as plt
import numpy as np
import slycot
from control import lqr
from matplotlib.colors import LogNorm
import pyvista as pv

from dynamic_simulation import DynamicsSimulator
import tqdm
from vis import add_origin_cube

G = 9.81


def multiply_quaternions(q_1, q_2, unitize: bool = True):
    """ Return the quaternion product q_1 * q_2 

    Args:
        q_1: (4,) quaternion (w, x, y, z)
        q_2: (4,) quaternion (w, x, y, z)
        unitize: ensure result has unit norm
    
    Returns:
        (w, x, y, z)
    """
    assert q_1.shape == (4,)
    assert q_2.shape == (4,)
    q_out_0 = q_1[0] * q_2[0] - np.dot(q_1[1:], q_2[1:])
    q_out_v = q_1[0] * q_2[1:] + q_2[0] * q_1[1:] + np.cross(q_1[1:], q_2[1:])
    q_out = np.concatenate(([q_out_0], q_out_v), axis=0)
    if unitize:
        q_norm = np.linalg.norm(q_out)
        if q_norm == 0:
            breakpoint()
        q_out = q_out / q_norm
    return q_out


def invert_quaternion(q):
    """(w, x, y, z)"""
    q_conj = np.concatenate(([q[0]], -q[1:]))
    return q_conj


class LQRController:
    def __init__(
        self,
        R,
        Q,
        x_init=np.array([0, 0, 0]),
        v_init=np.array([0, 0, 0]),
        q_init=np.array([1, 0, 0, 0]),
        epsilon=1e-7,
        timestep=1e-2,
        clamp_threshold=2,
    ):
        self.R = R
        self.Q = Q
        self.K = np.zeros((4, 10))

        # State variables
        self.x_spatial = x_init
        self.v_spatial = v_init
        self.q_spatial = q_init

        # Control variables
        self.angular_vel_body = np.ones(3) * epsilon
        self.c = epsilon

        # Reference variables
        self.x_spatial_ref = None
        self.v_spatial_ref = None
        self.q_spatial_ref = None

        # Forward Euler
        self.dt = timestep

        # Hackery
        self.clamp_threshold = clamp_threshold

        # Plotting
        self.state_history = []

    def _set_u(self, u):
        self.angular_vel_body = u[:3]
        self.c = u[3]

    def _get_u(self):
        u = np.concatenate((self.angular_vel_body, [self.c]))
        return u

    def _get_state(self):
        state = np.concatenate((self.x_spatial, self.q_spatial, self.v_spatial))
        return state

    def _set_state(self, state):
        self.x_spatial = state[0:3]
        self.q_spatial = state[3:7]
        self.v_spatial = state[7:9]

    def compute_error(
        self, x_spatial_des, q_spatial_des, v_spatial_des, subtraction_error=False
    ):
        x_error = self.x_spatial - x_spatial_des
        v_error = self.v_spatial - v_spatial_des
        # Error formulation taken from
        # https://github.com/llanesc/lqr-tracking/blob/270f2f5164a668bfb77e19f5191595f1d3913a16/src/lqr_quaternion.cpp#L225
        q_spatial_inv = invert_quaternion(self.q_spatial)
        print(f"About to compute error with {self.q_spatial}, {q_spatial_des}")
        q_error = multiply_quaternions(q_spatial_inv, q_spatial_des, unitize=False)
        # As done in the paper, the w term is set to zero
        q_error[0] = 0
        error = np.concatenate((x_error, q_error, v_error), axis=0)
        # print(f"Error {error}")
        return error

    # Equation (15)
    def compute_control_signal(self, feedforward=False, clamp=False, use_ref=False):
        A = self.compute_A()
        B = self.compute_B()
        # print(A)
        # print(B)
        # print(self.Q)
        # print(self.R)
        if False:
            fig, axs = plt.subplots(1, 2)
            cb1 = axs[0].matshow(A, norm=LogNorm(vmin=1e-12, vmax=10))
            cb2 = axs[1].matshow(B, norm=LogNorm(vmin=1e-12, vmax=10))
            plt.colorbar(cb1, ax=axs[0])
            plt.colorbar(cb2, ax=axs[1])
            plt.show()

        try:
            self.K, _, _ = lqr(A, B, self.Q, self.R)
            # print("Computed a new K")
        except slycot.exceptions.SlycotArithmeticError:
            pass
            # print("Using the old K")

        error = self.compute_error(
            self.x_spatial_ref, self.q_spatial_ref, self.v_spatial_ref
        )

        if feedforward:
            u_ref = self._get_u()
            u = u_ref - self.K @ error
        elif use_ref:
            u_ref = np.array([0, 0, 0, G])
            u = u_ref - self.K @ error
        else:
            u = -self.K @ error

        if clamp:
            u[:3] = np.clip(u[:3], -self.clamp_threshold, self.clamp_threshold)
        # print(u)

        self._set_u(u)

        # x = self._get_state()

        x_dot = self.compute_dynamics(u)
        # x_dot = A @ x + B @ u
        return x_dot

    def compute_dynamics(self, u):
        p_dot = self.v_spatial

        q_w, q_x, q_y, q_z = self.q_spatial
        w_x, w_y, w_z = u[:3]
        c = u[3]

        v_dot = np.array(
            [
                2 * (q_w * q_y + q_x * q_z) * c,
                2 * (q_y * q_z - q_w * q_x) * c,
                -G + (1 - 2 * q_x ** 2 - 2 * q_y ** 2) * c,
            ]
        )

        q_dot = 0.5 * np.array(
            [
                -w_x * q_x - w_y * q_y - w_z * q_z,
                w_x * q_w + w_z * q_y - w_y * q_z,
                w_y * q_w - w_z * q_x + w_x * q_z,
                w_z * q_w + w_y * q_x - w_x * q_y,
            ]
        )
        x_dot = np.concatenate((p_dot, q_dot, v_dot))
        return x_dot

    def q_partial_correction(self, epsilon=1e-12):
        q_norm = np.linalg.norm(self.q_spatial) + epsilon
        qq_t = np.expand_dims(self.q_spatial, axis=1) @ np.expand_dims(
            self.q_spatial, axis=0
        )
        eye_4 = np.eye(4)
        q_partial_correction = (eye_4 - q_norm ** 2 * qq_t) / q_norm
        return q_partial_correction

    # Equation 20
    def d_dq_q_dot(self):
        # TODO This very well might not be right
        x, y, z = self.angular_vel_body
        dqdot_dq = np.array(
            [[0, -x, -y, -z], [x, 0, z, -y], [y, -z, 0, x], [z, y, -x, 0]]
        )
        q_partial_correction = self.q_partial_correction()
        dqdot_dq = 0.5 * dqdot_dq @ q_partial_correction
        return dqdot_dq

    # Equation
    def d_dq_dot_v(self):
        x, y, z, w = self.q_spatial
        d_dq_dot_v = np.array([[y, z, w, x], [-x, -w, z, y], [w, -x, -y, z]])

        q_partial_correction = self.q_partial_correction()
        d_dq_dot_v = 2 * self.c * d_dq_dot_v @ q_partial_correction
        return d_dq_dot_v

    # Equation (16)
    def compute_A(self):
        A = np.zeros((10, 10))
        # TODO are these correct?
        A[0, 7] = 1
        A[1, 8] = 1
        A[2, 9] = 1

        A[3:7, 3:7] = self.d_dq_q_dot()
        A[7:10, 3:7] = self.d_dq_dot_v()
        return A

    def d_dw_dot_q(self):
        x, y, z, w = self.q_spatial
        d_dw_dot_q = 0.5 * np.array([[-x, -y, -z], [w, -z, -y], [z, w, x], [-y, x, w]])
        return d_dw_dot_q

    def d_dc_dot_v(self):
        x, y, z, w = self.q_spatial
        d_dc_dot_v = np.array(
            [w * y + x * z, y * z - w * x, w ** 2 - x ** 2 - y ** 2 + z ** 2]
        )
        return d_dc_dot_v

    # Equation (17)
    def compute_B(self):
        B = np.zeros((10, 4))
        B[3:7, 0:3] = self.d_dw_dot_q()
        B[7:10, 3] = self.d_dc_dot_v()
        return B

    def set_ref(self, x_spatial_ref, q_spatial_ref, v_spatial_ref):
        self.x_spatial_ref = x_spatial_ref
        self.v_spatial_ref = v_spatial_ref
        self.q_spatial_ref = q_spatial_ref

    def update_state(self, state_dot, linear_quat_integration=True):
        x_dot = state_dot[:3]
        q_dot = state_dot[3:7]
        v_dot = state_dot[7:]
        self.x_spatial = self.x_spatial + self.dt * x_dot
        self.v_spatial = self.v_spatial + self.dt * v_dot

        if linear_quat_integration:
            print(f"q was {self.q_spatial}")
            print(f"q_dot: {q_dot}")
            self.q_spatial = self.q_spatial + self.dt * q_dot
            print(f"q was updated to {self.q_spatial}")
        else:
            print("about to do quat intetration")
            self.q_spatial = self.q_spatial + self.dt / 2 * multiply_quaternions(
                self.q_spatial, q_dot
            )
        self.q_spatial = self.q_spatial / np.linalg.norm(self.q_spatial)

    def simulate(
        self,
        duration=2,
        plot_3d=False,
        feedforward_ref=False,
        plot_just_position=False,
        clamp=False,
        linear_quat_integration=False,
        use_ref_thrust=True,
    ):
        if (
            self.x_spatial_ref is None
            or self.q_spatial_ref is None
            or self.v_spatial_ref is None
        ):
            raise ValueError("No reference set")

        times = np.arange(0, duration, self.dt)

        for _ in times:
            x_dot = self.compute_control_signal(
                feedforward=feedforward_ref, clamp=clamp, use_ref=use_ref_thrust
            )
            self.update_state(x_dot, linear_quat_integration=linear_quat_integration)
            self.state_history.append(self._get_state())

        all_states = np.stack(self.state_history, axis=0)

        if plot_3d:
            traj = pv.PolyData(all_states[:, :3])
            goal = pv.Sphere(center=self.x_spatial_ref)
            plotter = pv.Plotter()
            plotter.add_mesh(goal)
            plotter.add_mesh(traj, scalars=times)
            add_origin_cube(plotter)
            plotter.show()
        else:
            names = (
                "x",
                "y",
                "z",
                "q_x",
                "q_y",
                "q_z",
                "q_w",
                "x_dot",
                "y_dot",
                "z_dot",
            )
            breakpoint()
            if plot_just_position:
                names = names[:3]

            for i, name in enumerate(names):
                plt.plot(all_states[:, i], label=name)
            plt.legend()
            plt.show()
        # print(f"State at time {t}: {self._get_state()}")


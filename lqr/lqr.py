from audioop import mul

import matplotlib.pyplot as plt
import numpy as np
import slycot
from control import lqr
from matplotlib.colors import LogNorm

from dynamic_simulation import DynamicsSimulator
import tqdm


def multiply_quaternions(q_1, q_2, unitize: bool = True):
    """ Return the quaternion product q_1 * q_2 

    Args:
        q_1: (4,) quaternion (i, j, k, 0)
        q_2: (4,) quaternion (i, j, k, 0)
        unitize: ensure result has unit norm
    
    Returns:
        (i, j, k, 0)
    """
    assert q_1.shape == (4,)
    assert q_2.shape == (4,)
    q_out_0 = q_1[3] * q_2[3] - np.dot(q_1[:3], q_2[:3])
    q_out_v = q_1[3] * q_2[:3] + q_2[3] * q_1[:3] + np.cross(q_1[:3], q_2[:3])
    q_out = np.concatenate((q_out_v, [q_out_0]), axis=0)
    if unitize:
        q_out = q_out / np.linalg.norm(q_out)
    return q_out


def invert_quaternion(q):
    q_conj = np.concatenate((-q[:3], [q[3]]))
    return q_conj


class LQRController:
    def __init__(
        self,
        R,
        Q,
        x_init=np.array([0, 0, 0]),
        v_init=np.array([0, 0, 0]),
        q_init=np.array([0, 0, 0, 1]),
        epsilon=1e-7,
        timestep=1e-3,
    ):
        self.R = R
        self.Q = Q

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

    def compute_error(self, x_spatial_des, q_spatial_des, v_spatial_des):
        x_error = self.x_spatial - x_spatial_des
        v_error = self.v_spatial - v_spatial_des

        q_spatial_inv = invert_quaternion(self.q_spatial)
        q_error = multiply_quaternions(q_spatial_inv, q_spatial_des)
        # As done in the paper, the w term is set to zero
        q_error[3] = 0
        error = np.concatenate((x_error, q_error, v_error), axis=0)
        # print(f"Error {error}")
        return error

    # Equation (15)
    def compute_control_signal(self):
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
            print("Using the old K")

        error = self.compute_error(
            self.x_spatial_ref, self.q_spatial_ref, self.v_spatial_ref
        )

        old_u = self._get_u()

        u = old_u + self.K @ error
        print(u)
        self._set_u(u)

        x = self._get_state()

        x_dot = A @ x + B @ u
        return x_dot

    def q_partial_correction(self):
        q_norm = np.linalg.norm(self.q_spatial)
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

    def update_state(self, state_dot):
        x_dot = state_dot[:3]
        q_dot = state_dot[3:7]
        v_dot = state_dot[7:]
        self.x_spatial = self.x_spatial + self.dt * x_dot
        self.v_spatial = self.v_spatial + self.dt * v_dot

        self.q_spatial = self.q_spatial + self.dt / 2 * multiply_quaternions(
            self.q_spatial, q_dot
        )
        self.q_spatial = self.q_spatial / np.linalg.norm(self.q_spatial)

    def simulate(self, duration=5):
        if (
            self.x_spatial_ref is None
            or self.q_spatial_ref is None
            or self.v_spatial_ref is None
        ):
            raise ValueError("No reference set")

        times = np.arange(0, duration, self.dt)

        for t in times:
            x_dot = self.compute_control_signal()
            self.update_state(x_dot)
            self.state_history.append(self._get_state())

        all_states = np.stack(self.state_history, axis=0)
        names = ("x", "y", "z", "q_x", "q_y", "q_z", "q_w", "x_dot", "y_dot", "z_dot")
        for i, name in enumerate(names[:3]):
            plt.plot(all_states[:, i], label=name)
        plt.legend()
        plt.show()
        # print(f"State at time {t}: {self._get_state()}")


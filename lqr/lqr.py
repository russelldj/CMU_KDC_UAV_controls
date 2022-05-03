from control import lqr
import numpy as np

from dynamic_simulation import DynamicsSimulator


class LQRController:
    def __init__(self, R, Q):
        self.R = R
        self.Q = Q

        # State variables
        self.x_spatial = None
        self.v_spatial = None
        self.q_spatial = None

        # Control variables
        self.angular_vel_body = None
        self.c = None

    def _set_u(self, u):
        self.body_normalized_thrust = u[3]
        self.angular_vel_body = u[:3]

    def _get_u(self):
        u = np.concatenate((self.angular_vel_body, [self.body_normalized_thrust]))
        return u

    def _get_state(self):
        state = np.concatenate((self.x_spatial, self.q_spatial, self.v_spatial))
        return state

    def _set_state(self, state):
        self.x_spatial = state[0:3]
        self.q_spatial = state[3:7]
        self.v_spatial = state[7:9]

    # Equation (15)
    def compute_control_signal(self):
        A = self.compute_A()
        B = self.compute_B()
        K, _, _ = lqr(A, B, self.Q, self.R)

        breakpoint()

    def q_partial_correction(self):
        q_norm = np.linalg.norm(self.q_spatial)
        qq_t = np.expand_dim(self.q_spatial, axis=1) @ np.expand(self.q_spatial, axis=0)
        eye_4 = np.eye(4)

        q_partial_correction = (eye_4 - q_norm ** 2 * qq_t) / q_norm
        return q_partial_correction

    def d_dq_q_dot(self):
        x, y, z, w = self.angular_vel_body
        dqdot_dq = np.array(
            [[0, -x, -y, -z], [x, 0, z, -y], [y, -z, 0, x], [z, y, -x, 0]]
        )
        q_partial_correction = self.q_partial_correction()
        dqdot_dq = 0.5 * dqdot_dq * q_partial_correction
        return dqdot_dq

    # Equation
    def d_dq_dot_v(self):
        x, y, z, w = self.q_spatial
        d_dq_dot_v = np.array([[y, z, w, x], [-x, -w, z, y], [w, -x, -y, z]])

        q_partial_correction = self.q_partial_correction()
        d_dq_dot_v = 2 * self.c * d_dq_dot_v * q_partial_correction
        return d_dq_dot_v

    # Equation (16)
    def compute_A(self):
        A = np.zeros((9, 9))
        A[0, 7] = 1
        A[1, 8] = 1
        A[2, 9] = 1
        A[3:7, 3:7] = self.d_dq_q_dot()
        A[7:9, 3:7] = self.d_dq_dot_v

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
        B = np.zeros((9, 3))
        B[3:6, 0:3] = self.d_dw_dot_q()
        B[7:9, 0:3] = self.d_dc_dot_v()

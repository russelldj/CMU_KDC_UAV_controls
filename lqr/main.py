import numpy

from lqr import LQRController
import numpy as np

if __name__ == "__main__":
    Q = np.diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]) * 0.0001
    R = np.diag([500, 500, 500, 7.5])
    X_INIT = np.array([0, 0, 0])
    X_REF = np.array([0, 0, 10])
    Q_REF = np.array([1, 0, 0, 0])
    V_REF = np.array([0, 0, 0])

    controller = LQRController(R, Q, x_init=X_INIT)
    controller.set_ref(X_REF, Q_REF, V_REF)
    controller.simulate(
        feedforward_ref=False,
        linear_quat_integration=False,
        plot_just_position=False,
        use_ref_thrust=True,
    )

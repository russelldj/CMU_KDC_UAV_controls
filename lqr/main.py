import numpy

from lqr import LQRController
import numpy as np

if __name__ == "__main__":
    Q = np.diag([100, 100, 100, 1, 1, 1, 1, 10, 10, 10])
    R = np.diag([1, 5, 5, 0.1])

    controller = LQRController(R, Q)
    controller.set_ref(
        np.array([0, 0, 10]), np.array([0, 0, 0, 1]), np.array([0, 0, 0])
    )
    controller.simulate()

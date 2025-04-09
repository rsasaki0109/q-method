import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot

def estimate_quaternion(w, v, P):
    """
    Estimate the quaternion that best rotates vectors w into vectors v
    using Davenport's Q-method. Weights for each vector pair are specified
    by P.

    Parameters
    ----------
    w : numpy.ndarray
        3 x N array of 'source' vectors.
    v : numpy.ndarray
        3 x N array of 'destination' vectors.
    P : numpy.ndarray
        1 x N array of weights for each vector pair. Should be non-negative.

    Returns
    -------
    q : numpy.ndarray
        A quaternion [x, y, z, w], suitable for scipy.spatial.transform.Rotation.
    """
    if w.shape != v.shape:
        raise ValueError("w and v must have the same shape (3 x N).")
    if w.shape[0] != 3:
        raise ValueError("w and v must be 3 x N arrays.")
    if len(P) != w.shape[1]:
        raise ValueError("Length of P must match the number of columns of w and v.")

    # Normalize weights so that sum of P is 1
    P = np.asarray(P, dtype=float)
    total_weight = np.sum(P)
    if total_weight == 0:
        raise ValueError("All weights in P are zero; cannot normalize.")
    a = P / total_weight

    B = np.zeros((3, 3))
    sigma = 0.0
    z = np.zeros(3)

    # Construct the B matrix, cross-sum vector z, and scalar sigma
    for i in range(w.shape[1]):
        B += a[i] * np.outer(w[:, i], v[:, i])
        sigma += a[i] * np.dot(w[:, i], v[:, i])
        z += a[i] * np.cross(w[:, i], v[:, i])

    # S is the symmetric part of B
    S = B + B.T

    # Build K
    A_11 = S - sigma * np.eye(3)
    A_12 = z.reshape(3, 1)
    A_21 = z.reshape(1, 3)
    A_22 = np.array([[sigma]])

    K_top = np.hstack([A_11, A_12])
    K_bot = np.hstack([A_21, A_22])
    K = np.vstack([K_top, K_bot])

    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eig(K)

    # Identify the eigenvector with the largest eigenvalue
    max_idx = np.argmax(eigenvalues)
    q_ = eigenvectors[:, max_idx]

    # q_ is [x, y, z, w] in the K-based approach
    # But it might have an arbitrary sign, so pick a convention
    if q_[3] < 0:
        q_ = -q_

    # Return in [x, y, z, w] order to match scipy's default
    return q_

def plot_vectors(ax, origin, vectors, colors, labels, linestyle="solid"):
    """
    Helper to plot one or more vectors from a given origin in a 3D Axes object.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D Axes on which to plot the vectors.
    origin : array-like
        The (x, y, z) coordinates of the common origin for all vectors.
    vectors : numpy.ndarray
        3 x N array representing N vectors in 3D space.
    colors : list of str
        A list of color specifiers for each vector.
    labels : list of str
        A list of labels for each vector (used by legend).
    linestyle : str, optional
        The style of the plotted line, by default "solid".
    """
    for i in range(vectors.shape[1]):
        ax.quiver(origin[0], origin[1], origin[2],
                  vectors[0, i], vectors[1, i], vectors[2, i],
                  color=colors[i], linestyle=linestyle,
                  label=labels[i] if (i < len(labels)) else "")

def test_estimate_quaternion():
    """
    Example test:
    1) Generate a random rotation (in this case using Euler angles: z=50°, y=0°, x=0°).
    2) Rotate a known vector w by that rotation to get v.
    3) Estimate the quaternion that rotates w -> v using estimate_quaternion.
    4) Compare v_est to the true v.
    """
    # Define a ground-truth rotation
    r0 = Rot.from_euler('zyx', [50, 0, 0], degrees=True)
    R = r0.as_matrix()

    # Original and rotated vectors
    w = np.array([[1.0], [0.0], [0.0]])
    v = R @ w

    # Weights
    P = np.ones(1) * 0.01

    # Estimate the quaternion
    q = estimate_quaternion(w, v, P)

    # Construct rotation from the estimated quaternion
    # Rotation.from_quat expects [x, y, z, w]
    r_est = Rot.from_quat(q)

    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original w (green)
    plot_vectors(ax, [0, 0, 0], w, ['g'], ['w'])
    # Plot the "true" rotated vector v (blue)
    plot_vectors(ax, [0, 0, 0], v, ['b'], ['v'])

    # Plot the estimated rotation applied to w (red, dashed)
    v_est = r_est.apply(w.T).T
    plot_vectors(ax, [0, 0, 0], v_est, ['r'], ['v_est'], linestyle='dashed')

    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__ == "__main__":
    test_estimate_quaternion()
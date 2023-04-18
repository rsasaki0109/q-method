import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot

def estimate_quaternion(w, v, P):
    B = np.zeros((3, 3))
    sig = 0
    z = np.zeros(3)
    a = P / sum(P)

    w_2 = np.zeros(3)
    v_2 = np.zeros(3)

    for i in range(0, w.shape[1]):
        w_2 = w[:, i]
        v_2 = v[:, i]
        B += a[i] * np.tensordot(w_2, v_2, axes=0)
        sig += a[i] * np.dot(w_2, v_2)
        z += a[i] * np.cross(w_2, v_2)

    S = B + B.T
    A_11 = S - sig*np.eye(3)
    A_12 = np.c_[z]
    A_21 = z
    A_22 = sig

    A_1 = np.hstack([A_11, A_12])
    A_2 = np.hstack([A_21, A_22])

    K = np.append(A_1, np.array([A_2]), axis=0)
    eig, V = np.linalg.eig(K)
    ind = np.argsort(eig)
    V = V[:, ind] / np.tile(np.sqrt(np.sum(np.power(V, 2), axis=1)) , (4, 1))
    q_ = V[:, 3]

    return np.array([-q_[0], -q_[1], -q_[2], q_[3]])

def plot_vectors(ax, origin, vectors, colors, labels, linestyle = "solid"):
    for i in range(vectors.shape[1]):
        ax.quiver(origin[0], origin[1], origin[2],
                  vectors[0, i], vectors[1, i], vectors[2, i],
                  color=colors[i], linestyle = linestyle, label=labels[i] if i == 0 else "")

def test_estimate_quaternion():
    r0 = Rot.from_euler('zyx', [50, 0, 0], degrees=True)
    R = r0.as_matrix()

    w = np.array([[1.0], [0.0], [0.0]])
    v = np.dot(R, w)
    P = np.ones(1) * 0.01

    q = estimate_quaternion(w, v, P)
    r1 = Rot.from_quat([q[1], q[2], q[3], q[0]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_vectors(ax, [0, 0, 0], w, ['g'], ['w'])
    plot_vectors(ax, [0, 0, 0], v, ['b'], ['v'])

    v_est = np.dot(r1.as_matrix(), w)
    plot_vectors(ax, [0, 0, 0], v_est, ['r'], ['v_est'], 'dashed')

    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

test_estimate_quaternion()
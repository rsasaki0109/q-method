import numpy as np
from math import sin, cos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot

def qmethod(w, v, P):
        # q-method
        # estimate q to minimize Wahba's loss function
        # input
        # ref vector v[3×n]
        # obs vector w[3×n]
        # output
        #   quatunion  q[4×1]
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

def qmethod_test():
        r0 = Rot.from_euler('zyx', [50, 0, 0], degrees=True)
        print("--r0--")
        print(r0.as_quat())
        print(r0.as_euler('zyx', degrees=True))
        print(r0.as_matrix())
        
        R = r0.as_matrix()

        w = np.array([[1.0], [0.0], [0.0]])
        print("w")
        print(w)
        
        v = np.dot(R, w)
        print("v")
        print(v)
        
        P = np.ones(1)*0.01

        q = qmethod(w, v, P)
        print("--q--")
        print(q)
        
        r1 = Rot.from_quat([q[1], q[2], q[3], q[0]])
        print("--r1--")
        print(r1.as_quat())
        print(r1.as_euler('zyx', degrees=True))

        r2 = Rot.from_quat([q[0], q[1], q[2], q[3]])
        print("--r2--")
        print(r2.as_quat())
        print(r2.as_euler('zyx', degrees=True))

        # Visualization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot w
        for i in range(w.shape[1]):
                ax.quiver(0, 0, 0, w[0, i], w[1, i], w[2, i], color='b', label='w' if i == 0 else "")

        # Plot v
        for i in range(v.shape[1]):
                ax.quiver(0, 0, 0, v[0, i], v[1, i], v[2, i], color='r', label='v' if i == 0 else "")

        # Calculate v_est
        r1_matrix = r1.as_matrix()
        v_est = np.dot(r1_matrix, w)

        # Plot v_est
        for i in range(v_est.shape[1]):
                ax.quiver(0, 0, 0, v_est[0, i], v_est[1, i], v_est[2, i], color='g', label='v_est' if i == 0 else "")

        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

qmethod_test()    

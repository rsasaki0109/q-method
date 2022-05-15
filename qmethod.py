import numpy as np
from math import sin, cos

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
        R = np.array([
                      [cos(50 * np.pi / 180), -sin(50 * np.pi / 180), 0],
                      [sin(50 * np.pi / 180), cos(50 * np.pi / 180), 0],
                      [0, 0, 1]
                #       [cos(50 * np.pi / 180), -sin(50 * np.pi / 180), 0],
                #       [sin(50 * np.pi / 180), cos(50 * np.pi / 180), 0],
                #       [0, 0, 1]
                        ])       
        print("--R--")
        print(R)
        #w = np.ones((3, 30))
        w = np.ones((3, 1))
        print("w")
        print(w)
        v = np.dot(R, w)
        print("v")
        print(v)
        #P = np.ones(30)*0.01
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

qmethod_test()    

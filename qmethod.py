import numpy as np
from math import sin, cos

def qmethod(w, v, P):
        # q-method
        # estimate q to minimize Wahba's loss function
        # input
        #   基準ベクトルv[3×n]
        #   観測ベクトルw[3×n]
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
        R = np.array([
                      [cos(50), -sin(50), 0],
                      [sin(50), cos(50), 0],
                      [0, 0, 1]
                        ])        
        w = np.ones((3, 30))
        v = np.dot(R, w)
        P = np.ones(30)*0.01

        q = qmethod(w, v, P)
        print(q)

qmethod_test()    

## AA203: Final project on Insulin-Glucose dynamics and control
## File is on LQR tracking and control

import numpy as np

class LQR(x, xf, Q, R, A, B, dt = .01):
    def __init__(self, x, xf, Q, R, A, B, dt):
        self.x = x
        self.xf = xf
        self.Q = Q
        self.R = R
        self.A = A
        self.B = B
        self.dt = dt

    def find_K(self):
        delta = 1 #arbitrary to get into the loop
        K = None
        P = np.zeros((self.n,self.n)) #initializing to 0 #!!!! ISSUE
        P_old = P
        iters = 0
        while np.any(delta >= 1e-4):
            K = -np.linalg.solve((self.R+self.B.T@P_old@self.B),self.B.T@P_old@self.A)
            P_new = self.Q+self.A.T@P_old@(self.A+self.B@K)
            delta = np.abs(P_old-P_new)
            P_old = P_new
            iters += 1

        K = np.squeeze(K)
        print("The value of K: ", str(np.around(K, 2)))
        return K

    def x_error(self, x = None, xf = None):
        if not x:
            x = self.x
        if not xf:
            xf = self.xf

        return x-xf

## AA203: Final project on Insulin-Glucose dynamics and control
## File is on LQR tracking and control

import numpy as np
from util import *

class LQR():
    def __init__(self, x, xf, Q, R, A, B, dt = .01):
        self.x = x
        self.xStar = xf
        self.Q = Q
        self.R = R
        self.A = A
        self.B = B
        self.dt = dt
        self.m = np.shape(self.Q)[0]
        self.n = np.shape(self.Q)[1]
        self.K, self.P = self.find_K_P()

    def find_K_P(self): #infinite time horizon
        delta = 1 #arbitrary to get into the loop
        K = None
        P = np.zeros((self.m,self.n)) #initializing to 0 #!!!! ISSUE
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
        return K, P_old

        # Need to alter: So instead of a while loop, we do a 'for a dT within a time range' loop. And, every new time bracket we calculate a new K and P
        # So, instead of threshold, we would calculate a K for each time-step, making it adaptive.

def simulate_withControl(dT, t, dynamics, ekf, lqr):
    #simulation loop
    u = calculate_control(K, x_error(dynamics.x[:,0], lqr.xStar))
    for i in range(1,len(t)):
        x,measurement = dynamics.update_state(u,t[i],dT)
        u = calculate_control(K, x_error(x, lqr.xStar))
        mu = ekf.update_estimate(u,measurement,dT)
        if i%10 == 0:
            A,B = recalc_Dynamics(mu[3:,i-1])
            lqr.A = A
            lqr.B = B
            K, P = lqr.find_K_P()

def recalc_Dynamics(params):
    params = np.squeeze(params)
    A = np.array([[0., -params[0], params[1]],\
                  [params[2], -params[3], 0.],\
                  [0., 0., -params[4]]])
    B = np.array([[0.],[params[5]],[0.]])
    return A, B

def x_error(x, xStar):
    return x-xStar

def calculate_control(K, x_error):
    return np.squeeze(K@x_error)

## MAIN FROM NOW ON ##

def main():
    #time info
    dT = 0.1
    tf = 10
    t = np.linspace(0,tf,int(tf/dT)+1)
    N = len(t)

    #true state info
    x0 = np.array([[200./60],\
                   [300.],\
                   [2000.]]) #G (dl/sec),I(pmole/L),D(mg)
    params = np.array([[0.01/60],\
                       [0.611/60],\
                       [0.665/60],\
                       [0.508/60],\
                       [0.042/60],\
                       [1./0.0025]]) #k2,k0,k1,k6,ka,km(1/L) (units converted to seconds)
    q = 0.05*x0
    r = np.array([[0.0667],\
                  [6]]) #2% of x0 for G and I
    Q = np.diag(np.squeeze(q))
    R = np.diag(np.squeeze(r))
    dynamics = Dynamics(x0,params,f_x,Q,R,N)

    #estimate of state info
    mu0 = np.array([[4.],\
                    [350],\
                    [500.],\
                    [0.015/60],\
                    [0.650/60],\
                    [0.7/60],\
                    [0.450/60],\
                    [0.05/60],\
                    [500]])
    S0 = np.eye(9)
    q_hat = 0.5*np.vstack((x0,params)) 
    r_hat = 0.5*r 
    Q_hat = np.diag(np.squeeze(q_hat))
    R_hat = np.diag(np.squeeze(r_hat))
    alpha = 0.01
    adaptive = True
    ekf = EKF(mu0,S0,f_x_and_params,Q_hat,R_hat,adaptive,alpha,N)
   
    #simulate
    print('Using Adaptive EKF = ' + str(adaptive))
    simulate(dT,t,dynamics,ekf)

    #Run EKF for a long time to figure out what x_star is!
    #Run LQR
    #plot results

    #repeat using adaptive versus non-adaptive and compare preformances when disturbances are there (change adaptive = True to False)

if __name__ == '__main__':
	main()

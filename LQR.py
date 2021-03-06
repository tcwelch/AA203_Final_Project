## AA203: Final project on Insulin-Glucose dynamics and control
## File is on LQR tracking and control

import matplotlib
import numpy as np
from util import *
import matplotlib.pyplot as plt

class LQR():
    def __init__(self, xf, Q, R, A, B, dt = .01):
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
        while np.any(delta >= 1e-4) and iters <1e5: #max iters constraint
            K = -np.linalg.solve((self.R+self.B.T@P_old@self.B), self.B.T@P_old@self.A) # ERROR 
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
    u = calculate_control(lqr.K, x_error(dynamics.x[:,0], lqr.xStar))
    u_array = [u]
    for i in range(1,len(t)):
        x,measurement = dynamics.update_state(u,t[i],dT)
        mu = ekf.update_estimate(u,measurement,dT)
        A,B = recalc_Dynamics(mu[3:],dT)
        lqr.A = A
        lqr.B = B
        lqr.K, lqr.P = lqr.find_K_P()
        u = calculate_control(lqr.K, x_error(x, lqr.xStar))
        u_array.append(u)
    return np.array(u_array)

def recalc_Dynamics(params, dt):
    params = np.squeeze(params)
    A = np.array([[0., -params[0], params[1]],\
                  [params[2], -params[3], 0.],\
                  [0., 0., -params[4]]])
    B = np.array([[0.],[params[5]],[0.]])
    A = np.eye(A.shape[0], A.shape[1]) + dt*A
    B = dt*B
    return A, B

def x_error(x, xStar):
    returnThis = x-xStar
    returnThis[1] = 0
    returnThis[2] = 0
    return returnThis

def calculate_control(K, x_error):
    return np.maximum(np.array([0.]),np.array([K@x_error]))

## MAIN FROM NOW ON ##

def main():
    #time info
    dT = 0.1
    tf = 30
    t = np.linspace(0,tf,int(tf/dT)+1)
    N = len(t)

    #true state info
    x0 = np.array([[60],\
                   [300.],\
                   [1500.]]) #G (dl/sec),I(pmole/L),D(mg)
    params = np.array([[0.01/60],\
                       [0.611/60],\
                       [0.665/60],\
                       [0.508/60],\
                       [4.2/60],\
                       [1./0.0025]]) #k2,k0,k1,k6,ka,km(1/L) (units converted to seconds)
    q = 0.05*x0
    r = np.array([[0.0667],\
                  [6]]) #2% of x0 for G and I
    Q = np.diag(np.squeeze(q))
    Q[2] = 0
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
                    [5/60],\
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

    # Run for standard case! 
    Q = np.diag([1e3,1e-3,1e-3])
    R = np.array([1e-6]) # we want to conserve insulin
    A, B = recalc_Dynamics(params, dT)
    lqr = LQR(np.array([100, 0, 0]), Q, R, A, B, dT)
    control = simulate_withControl(dT, t, dynamics, ekf, lqr)
    
    plt.figure()
    plt.plot(t, control)
    plt.title('Controller')
    plt.xlabel('Time (s)')
    plt.ylabel('Controller (pmol/s)')

    plot(t,dynamics,ekf)
    plt.pause(.01)
    plt.show()

    #repeat using adaptive versus non-adaptive and compare preformances when disturbances are there (change adaptive = True to False)

if __name__ == '__main__':
	main()

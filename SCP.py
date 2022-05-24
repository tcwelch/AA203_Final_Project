## AA203: Final project on Insulin-Glucose dynamics and control
## File is on LQR tracking and control

import matplotlib
import numpy as np
from util2 import *
import matplotlib.pyplot as plt
import cvxpy as cvx

class SCP():
    def __init__(self, xf, P, Q, R, A, B, dt,N):
        self.xStar = xf
        self.P = P
        self.Q = Q
        self.R = R
        self.A = A
        self.B = B
        self.dt = dt
        self.N = N
        self.m = np.shape(self.Q)[0]
        self.n = np.shape(self.Q)[1]
        self.p = np.shape(self.R)[0]

    def calculate_control(self, x0):
        #Setting up MPC
        x_cvx = cvx.Variable((self.N + 1, self.n))
        u_cvx = cvx.Variable((self.N, self.p))
        cost = []
        constraints = []
        constraints.append(x_cvx[0,:] == x0)
        for i in range(self.N):
            cost.append(cvx.quad_form(x_cvx[i,:].T,self.Q) + cvx.quad_form(u_cvx[i,:].T,self.R))
            constraints.append(x_cvx[i+1,:].T == self.A@(x_cvx[i,:].T) + self.B@(u_cvx[i,:].T))
            constraints.append(u_cvx[i,:] >= 0.)
        cost.append(cvx.quad_form(x_cvx[-1,:],self.P))
        cost = cvx.sum(cost)
        prob = cvx.Problem(cvx.Minimize(cost), constraints)
        prob.solve()
        x = x_cvx.value
        u = u_cvx.value
        return u[0]

def simulate_withControl(dT, t, dynamics, ekf, scp):
    #simulation loop
    u = scp.calculate_control(x_error(dynamics.x[:,0], scp.xStar))
    u_array = [u]
    for i in range(1,len(t)):
        x,measurement = dynamics.update_state(u,t[i],dT)
        mu = ekf.update_estimate(u,measurement,dT)
        A,B = recalc_Dynamics(mu[3:],dT)
        scp.A = A
        scp.B = B
        u = scp.calculate_control(x_error(x, scp.xStar))
        u_array.append(u)
    return np.array(u_array)

def recalc_Dynamics(params, dt):
    params = np.squeeze(params)
    A = np.array([[0., -params[0], params[1]],\
                  [0., -params[2], 0.],\
                  [0., 0., -params[3]]])
    B = np.array([[0.],[params[4]],[0.]])
    A = np.eye(A.shape[0], A.shape[1]) + dt*A
    B = dt*B
    return A, B

def x_error(x, xStar):
    returnThis = x-xStar
    # returnThis[1] = 0
    # returnThis[2] = 0
    return returnThis


## MAIN FROM NOW ON ##

def main():
    #time info
    dT = 0.1
    tf = 65
    t = np.linspace(0,tf,int(tf/dT)+1)
    N = len(t)

    #true state info
    x0 = np.array([[60],\
                   [300.],\
                   [1500.]]) #G (dl/sec),I(pmole/L),D(mg)
    params = np.array([[0.01/60],\
                       [0.611/60],\
                       [0.],\
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
                    [0.450/60],\
                    [5/60],\
                    [500]])
    S0 = np.eye(8)
    params_diabetic = np.array([[0.01/60],\
                                [0.611/60],\
                                [0.508/60],\
                                [4.2/60],\
                                [1./0.0025]])
    q_hat = 0.5*np.vstack((x0,params_diabetic)) 
    r_hat = 0.5*r 
    Q_hat = np.diag(np.squeeze(q_hat))
    R_hat = np.diag(np.squeeze(r_hat))
    alpha = 0.01
    adaptive = True
    ekf_diabetic = EKF_Diabetic(mu0,S0,f_x_and_params_diabetic,Q_hat,R_hat,adaptive,alpha,N)
   
    print('Using Adaptive EKF = ' + str(adaptive))
    # Run for standard case! 
    P = np.diag([1e3,1e-3,1e-3])
    Q = P
    R = np.array([[1e-2]]) # we want to conserve insulin
    A, B = recalc_Dynamics(params_diabetic, dT)
    N = 25
    scp = SCP(np.array([100, 0, 0]), P,Q, R, A, B, dT,N)
    control = simulate_withControl(dT, t, dynamics, ekf_diabetic, scp)
    
    plt.figure()
    plt.plot(t, control)
    plt.title('Controller')
    plt.xlabel('Time (s)')
    plt.ylabel('Controller (pmol/s)')

    plot_diabetic(t,dynamics,ekf_diabetic)
    plt.show()

    #repeat using adaptive versus non-adaptive and compare preformances when disturbances are there (change adaptive = True to False)

if __name__ == '__main__':
	main()

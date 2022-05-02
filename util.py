import numpy as np
import matplotlib.pyplot as plt

def main():
    #time info
    dT = 0.1
    tf = 15
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
                       [400]]) #k2,k0,k1,k6,ka,km(1/L) (units converted to seconds)
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
    q_hat = 0.05*np.vstack((x0,params)) #np.array([])
    r_hat = r #np.array([])
    Q_hat = np.diag(np.squeeze(q_hat))
    R_hat = np.diag(np.squeeze(r_hat))
    window_size = 10
    adaptive = False
    ekf = EKF(mu0,S0,f_x_and_params,Q_hat,R_hat,adaptive,window_size,N)
    #simulation loop
    for i in range(1,len(t)):
        u = np.array([0.])
        x,measurement = dynamics.update_state(u,dT)
        mu = ekf.update_estimate(u,measurement,dT)

    #plot of state and estimate of state dynamics
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t,dynamics.x[0,:dynamics.itr],label='true state')
    plt.plot(t,ekf.mu[0,:ekf.itr],label='ekf estimate')
    plt.ylabel('Glucose Concentration in Blood []')
    plt.title('States and Estimates vs. Time')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(t,dynamics.x[1,:dynamics.itr])
    plt.plot(t,ekf.mu[1,:ekf.itr])
    plt.ylabel('Insulin Concentration in Blood []')

    plt.subplot(3,1,3)
    plt.plot(t,dynamics.x[2,:dynamics.itr])
    plt.plot(t,ekf.mu[2,:ekf.itr])
    plt.ylabel('Glucose Concentration in Stomach []')
    plt.xlabel('Time (seconds)')   

    #plot of parameter dynamics
    plt.figure()
    plt.subplot(6,1,1)
    plt.plot(t,dynamics.params[0]*np.ones((len(t),1)),label='actual value')
    plt.plot(t,ekf.mu[3,:ekf.itr],label='ekf estimate')
    plt.ylabel('k2 []')
    plt.title('Parameters and Estimates vs. Time')

    plt.subplot(6,1,2)
    plt.plot(t,dynamics.params[1]*np.ones((len(t),1)))
    plt.plot(t,ekf.mu[4,:ekf.itr])
    plt.ylabel('k0 []')

    plt.subplot(6,1,3)
    plt.plot(t,dynamics.params[2]*np.ones((len(t),1)))
    plt.plot(t,ekf.mu[5,:ekf.itr])
    plt.ylabel('k1 []')

    plt.subplot(6,1,4)
    plt.plot(t,dynamics.params[3]*np.ones((len(t),1)))
    plt.plot(t,ekf.mu[6,:ekf.itr])
    plt.ylabel('k6 []')

    plt.subplot(6,1,5)
    plt.plot(t,dynamics.params[4]*np.ones((len(t),1)))
    plt.plot(t,ekf.mu[7,:ekf.itr])
    plt.ylabel('ka []')

    plt.subplot(6,1,6)
    plt.plot(t,dynamics.params[5]*np.ones((len(t),1)))
    plt.plot(t,ekf.mu[8,:ekf.itr])
    plt.ylabel('km []')
    plt.xlabel('Time (seconds)')
    
    plt.show()

def f_x(x,u,params):
    params = np.squeeze(params)
    A = np.array([[0., -params[0], params[1]],\
                  [params[2], -params[3], 0.],\
                  [0., 0., -params[4]]])
    B = np.array([[0.],[params[5]],[0.]])
    return A@x + B@u

def f_x_and_params(x,u,params):
    x_short_dot = f_x(x,u,params)
    x_long_dot = np.zeros((9,1))
    x_long_dot[:3] = np.reshape(x_short_dot,(np.shape(x_long_dot[:3])))
    return x_long_dot

class Dynamics():
    def __init__(self,x0,params,f,Q,R,N):
        dim1 = np.shape(Q)
        self.n = dim1[0]
        dim2 = np.shape(R)
        self.m = dim2[0]

        self.x = np.zeros((self.n,N)) #np array
        self.x[:,0] = np.squeeze(x0)
        self.params = params
        self.measurement = np.zeros((self.m,N))
        self.measurement[:,0] = np.squeeze(np.array([x0[0],x0[1]]))
        self.Q = Q
        self.R = R
        self.f = f
        self.itr = 1

    def update_state(self,u,dT):
        self.make_space()
        i = self.itr
        self.x[:,i] = np.squeeze(self.x[:,i-1] + dT*self.f(self.x[:,i-1],u,self.params) \
            + np.random.multivariate_normal(np.zeros((self.n,)),self.Q))
        self.measurement[:,i] = np.squeeze(np.array([self.x[0,i],self.x[1,i]])\
            + np.random.multivariate_normal(np.zeros((self.m,)),self.R))
        self.itr += 1  
        return self.x[:,i],self.measurement[:,i]

    def make_space(self):
        dim = np.shape(self.x)
        if dim[1] == self.itr:
            self.x = np.reshape(self.x,(dim[0],dim[1]*2))
            self.x = np.reshape(self.x,(dim[0],dim[1]*2))


class EKF():
    def __init__(self,mu0,S0,f,Q,R,adaptive,window_size,N):
        dim1 = np.shape(Q)
        self.n = dim1[0]
        dim2 = np.shape(R)
        self.m = dim2[0]

        self.mu = np.zeros((self.n,N))
        self.mu[:,0] = np.squeeze(mu0)
        self.mu_pred = np.zeros((self.n,N)) #<-- need to initialize?
        self.S = S0
        self.f = f
        self.C = np.zeros((self.m,self.n))
        self.C[0,0] = 1.0
        self.C[1,1] = 1.0
        self.Q = Q
        self.R = R
        self.adaptive = adaptive
        self.W = window_size
        self.d = np.zeros((self.m,N))
        self.itr = 1

    def update_estimate(self,u,measurement,dT):
        i = self.itr
        self.make_space()
        #linearize A
        A = np.array([[0.,-self.mu[3,i-1],self.mu[4,i-1],-self.mu[1,i-1],self.mu[2,i-1],0.,0.,0.,0.],\
            [self.mu[5,i-1],-self.mu[6,i-1],0.,0.,0.,self.mu[0,i-1],-self.mu[1,i-1],0.,u[0]],\
            [0.,0.,-self.mu[7,i-1],0.,0.,0.,0.,-self.mu[2,i-1],0.],\
            [0.,0.,0.,0.,0.,0.,0.,0.,0.],\
            [0.,0.,0.,0.,0.,0.,0.,0.,0.],\
            [0.,0.,0.,0.,0.,0.,0.,0.,0.],\
            [0.,0.,0.,0.,0.,0.,0.,0.,0.],\
            [0.,0.,0.,0.,0.,0.,0.,0.,0.],\
            [0.,0.,0.,0.,0.,0.,0.,0.,0.]]) #<--dimensions on x given estimating param from u??
        #discretize A
        A = np.eye(np.shape(A)[0]) + dT*A
        #predict
        self.mu_pred[:,i] = np.squeeze(self.f(self.mu[:3,i-1],u,self.mu[3:,i-1]))
        S_predict = A@self.S@A.T + self.Q

        #linearize C with mu_pred (already done because C is constant)
        #calculating Kalman Gain
        K = S_predict@self.C.T@np.linalg.inv(self.C@S_predict@self.C.T+self.R)
        #calculated predicted measurement
        pred_measurement = self.C@self.mu_pred[:,i]
        #update
        self.mu[:,i] = np.squeeze(self.mu_pred[:,i] \
                       + K@(measurement - pred_measurement))
        self.S = (np.eye(np.shape(A)[0]) + K@self.C)@S_predict

        #update Q if adaptive
        if self.adaptive:
            self.d[:,i] = np.squeeze(measurement - pred_measurement)
            if i < self.W:
                D = (1/self.W)*np.sum(self.d[:,:i]@self.d[:,:i].T)
                delta_x = self.mu[:,:i] - self.mu_pred[:,:i]
                L = np.divide(delta_x,self.d[:,:i])
                self.Q = L@D@L.T
            else:
                D = (1/self.W)*np.sum(self.d[:,(i-self.W):]@self.d[:,(i-self.W):].T)
                delta_x = self.mu[:,(i-self.W):] - self.mu_pred[:,(i-self.W):]
                L = np.divide(delta_x,self.d[:,(i-self.W):])
                self.Q = L@D@L.T
        self.itr += 1
        return self.mu[:,i]

    def make_space(self):
        dim = np.shape(self.mu)
        if dim[1] == self.itr:
            self.mu = np.reshape(self.mu,(dim[0],dim[1]*2))
            self.mu = np.reshape(self.mu,(dim[0],dim[1]*2))

if __name__ == '__main__':
	main()
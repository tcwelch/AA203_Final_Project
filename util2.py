import numpy as np
import matplotlib.pyplot as plt

def main():
    #time info
    dT = 0.1
    tf = 100
    t = np.linspace(0,tf,int(tf/dT)+1)
    N = len(t)

    #true state info
    x0 = np.array([[200./60],\
                   [300.],\
                   [2000.]]) #G (dl/sec),I(pmole/L),D(mg)
    params = np.array([[0.],\
                       [0.611/60],\
                       [0.665/60],\
                       [0.],\
                       [4.2/60],\
                       [1./0.0025]]) #k0,k1,ka(div by 100),km(1/L) (units converted to seconds)
    q = 0.05*x0
    #q[2] = 0 # TOGGLE on if you don't want noise in sugar intake 
    r = np.array([[0.0667],\
                  [6]]) #2% of x0 for G and I
    Q = np.diag(np.squeeze(q))
    Q[2] = 0 #TOGGLE on if you don't want noise in sugar intake
    R = np.diag(np.squeeze(r))
    dynamics = Dynamics(x0,params,f_x,Q,R,N)

    # #adding disturbance (step from food intake)
    # step_idx = 2
    # step_t0 = 8
    # step_height = 1000.
    # step = lambda t: step_height if t - step_t0 == 0 else 0.
    # dynamics.create_disturbance(step,step_idx)

    # # #adding disturbance (sinusoidal variation in I from heartbeat)
    # sin_idx = 1
    # sin_t0 = 5
    # sin_amp = 5.
    # sin_freq = 1
    # sin_I = lambda t: sin_amp*np.sin((sin_freq*2*np.pi)*(t-sin_t0)) if t >= sin_t0 else 0.
    # dynamics.create_disturbance(sin_I,sin_idx)

    #estimate of state info
    mu0 = np.array([[4.],\
                    [350],\
                    [500.],\
                    [0.650/60],\
                    [0.7/60],\
                    [5/60],\
                    [500]])
    S0 = np.eye(7)
    params_diabetic = np.array([[0.611/60],\
                                [0.665/60],\
                                [0.042/60],\
                                [1./0.0025]])
    q_hat = 0.5*np.vstack((x0,params_diabetic)) 
    r_hat = 0.5*r 
    Q_hat = np.diag(np.squeeze(q_hat))
    R_hat = np.diag(np.squeeze(r_hat))
    alpha = 0.01
    adaptive = True
    ekf_diabetic = EKF_Diabetic(mu0,S0,f_x_and_params_diabetic,Q_hat,R_hat,adaptive,alpha,N)
   
    #simulate
    print('Using Adaptive EKF = ' + str(adaptive))
    simulate(dT,t,dynamics,ekf_diabetic)
    #calculating error
    x_true = np.vstack((dynamics.x[:,:dynamics.itr],params_diabetic*np.ones((1,dynamics.itr))))
    x_exp = ekf_diabetic.mu
    error = prmse_diabetic(x_true,x_exp)
    #plotting
    plot_diabetic(t,dynamics,ekf_diabetic)
    print("steady state values: ", dynamics.measurement[:,-1])
    plt.show()

## MAIN STOPS HERE ##

def simulate(dT,t,dynamics,ekf):
    #simulation loop
    for i in range(1,len(t)):
        u = np.array([0.])
        x,measurement = dynamics.update_state(u,t[i],dT)
        mu = ekf.update_estimate(u,measurement,dT)

def prmse_diabetic(x_true,x_exp):
    N = np.shape(x_true)[1]
    error =  np.divide(np.sqrt(1/N)*np.linalg.norm(x_true - x_exp,ord=2,axis=1)*100*N,np.sum(x_true,axis=1))
    print('---Percent RMSE (%)---')
    print('prmse G: ' + str(round(error[0],2)) + '%')
    print('prmse I: ' + str(round(error[1],2)) + '%')
    print('prmse D: ' + str(round(error[2],2)) + '%')
    print('prmse k0: ' + str(round(error[3],2)) + '%')
    print('prmse k1: ' + str(round(error[4],2)) + '%')
    print('prmse ka: ' + str(round(error[5],2)) + '%')
    print('prmse km: ' + str(round(error[6],2)) + '%')
    print('average prmse:' + str(round(np.sum(error)/7,2)) + '%')
    return error
    
def plot_diabetic(t,dynamics,ekf):
    #plot of state and estimate of state dynamics
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t,dynamics.x[0,:dynamics.itr],label='true state')
    plt.plot(t,ekf.mu[0,:ekf.itr],label='ekf estimate')
    plt.ylabel('Glucose Concentration in Blood [dl / sec]',fontsize = 9)
    plt.title('States and Estimates vs. Time')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(t,dynamics.x[1,:dynamics.itr])
    plt.plot(t,ekf.mu[1,:ekf.itr])
    plt.ylabel('Insulin Concentration in Blood [pmol/liter]',fontsize = 9)

    plt.subplot(3,1,3)
    plt.plot(t,dynamics.x[2,:dynamics.itr])
    plt.plot(t,ekf.mu[2,:ekf.itr])
    plt.ylabel('Glucose Concentration in Stomach [mg]',fontsize = 9)
    plt.xlabel('Time (seconds)')   

    #plot of parameter dynamics
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(t,dynamics.params[1]*np.ones((len(t),1)))
    plt.plot(t,ekf.mu[3,:ekf.itr])
    plt.ylabel('k0 [1 / dl sec]',fontsize = 8)

    plt.subplot(4,1,2)
    plt.plot(t,dynamics.params[2]*np.ones((len(t),1)))
    plt.plot(t,ekf.mu[4,:ekf.itr])
    plt.ylabel('k1 [pmol dl / sec mg l]',fontsize = 8)

    plt.subplot(4,1,3)
    plt.plot(t,dynamics.params[4]*np.ones((len(t),1)))
    plt.plot(t,ekf.mu[5,:ekf.itr])
    plt.ylabel('ka [1 / sec]',fontsize = 8)

    plt.subplot(4,1,4)
    plt.plot(t,dynamics.params[5]*np.ones((len(t),1)))
    plt.plot(t,ekf.mu[6,:ekf.itr])
    plt.ylabel('km [1 / l]',fontsize = 8)
    plt.xlabel('Time (seconds)')

def f_x(x,u,params):
    params = np.squeeze(params)
    A = np.array([[0., -params[0], params[1]],\
                  [params[2], -params[3], 0.],\
                  [0., 0., -params[4]]])
    B = np.array([[0.],[params[5]],[0.]])
    return A@x + B@u

def f_x_and_params_diabetic(x,u,params):
    extended_params = np.zeros((6,1))
    extended_params[1] = params[0]
    extended_params[2] = params[1]
    extended_params[4] = params[2]
    extended_params[5] = params[3]
    x_short_dot = f_x(x,u,extended_params)
    x_long_dot = np.zeros((7,1))
    x_long_dot[:3] = np.reshape(x_short_dot,(np.shape(x_long_dot[:3])))
    return x_long_dot

class Dynamics():
    def __init__(self,x0,params,f,Q,R,N):
        dim1 = np.shape(Q)
        self.n = dim1[0]
        dim2 = np.shape(R)
        self.m = dim2[0]
        self.x = np.zeros((self.n,N)) 
        self.x[:,0] = np.squeeze(x0)
        self.params = params
        self.measurement = np.zeros((self.m,N))
        self.measurement[:,0] = np.squeeze(np.array([x0[0],x0[1]]))
        self.Q = Q
        self.R = R
        self.f = f
        self.itr = 1
        self.disturbances = []

    def update_state(self,u,t,dT):
        self.make_space()
        i = self.itr
        #updating state
        x = np.squeeze(self.x[:,i-1] + dT*self.f(self.x[:,i-1],u,self.params) \
            + np.random.multivariate_normal(np.zeros((self.n,)),self.Q))
        self.x[:,i] = np.maximum(x,np.zeros(self.n,))
        #simulating measurement
        measurement = np.squeeze(np.array([self.x[0,i],self.x[1,i]])\
            + np.random.multivariate_normal(np.zeros((self.m,)),self.R))
        self.measurement[:,i] = np.maximum(measurement,np.zeros(self.m,))
        self.add_disturbances(t)
        self.itr += 1  
        return self.x[:,i],self.measurement[:,i]

    def make_space(self):
        dim = np.shape(self.x)
        if dim[1] == self.itr:
            self.x = np.reshape(self.x,(dim[0],dim[1]*2))
            self.x = np.reshape(self.x,(dim[0],dim[1]*2))

    def create_disturbance(self,dist,dist_idx):
        self.disturbances.append((dist,dist_idx))

    def add_disturbances(self,t):
        for dist in self.disturbances:
            d,idx = dist
            self.x[idx,self.itr] = self.x[idx,self.itr] + d(t)

class EKF_Diabetic():
    def __init__(self,mu0,S0,f,Q,R,adaptive,alpha,N):
        dim1 = np.shape(Q)
        self.n = dim1[0]
        dim2 = np.shape(R)
        self.m = dim2[0]
        self.mu = np.zeros((self.n,N))
        self.mu[:,0] = np.squeeze(mu0)
        self.mu_pred = np.zeros((self.n,N)) 
        self.S = S0
        self.f = f
        self.C = np.zeros((self.m,self.n))
        self.C[0,0] = 1.0
        self.C[1,1] = 1.0
        self.Q = Q
        self.R = R
        self.adaptive = adaptive
        self.alpha = alpha
        self.d = np.zeros((self.m,N))
        self.itr = 1

    def update_estimate(self,u,measurement,dT):
        i = self.itr
        self.make_space()
        #linearize A
        A = np.array([[0.,0.,self.mu[3,i-1],self.mu[2,i-1],0.,0.,0.],\
            [self.mu[4,i-1],0.,0.,0.,self.mu[0,i-1],0.,u[0]],\
            [0.,0.,-self.mu[5,i-1],0.,0.,-self.mu[2,i-1],0.],\
            [0.,0.,0.,0.,0.,0.,0.],\
            [0.,0.,0.,0.,0.,0.,0.],\
            [0.,0.,0.,0.,0.,0.,0.],\
            [0.,0.,0.,0.,0.,0.,0.]]) 
        #discretize A
        A = np.eye(np.shape(A)[0]) + dT*A
        #predict
        mu_pred = self.mu[:,i-1] + dT*np.squeeze(self.f(self.mu[:3,i-1],u,self.mu[3:,i-1]))
        #saturating 
        self.mu_pred[:,i] = np.maximum(mu_pred,np.zeros(self.n,))
        S_predict = A@self.S@A.T + self.Q

        #linearize C with mu_pred (already done because C is constant)
        #calculating Kalman Gain
        K = S_predict@self.C.T@np.linalg.inv(self.C@S_predict@self.C.T+self.R)
        #calculated predicted measurement
        pred_measurement = self.C@self.mu_pred[:,i]
        pred_measurement = np.maximum(pred_measurement,np.zeros((self.m,)))
        #update
        d = measurement - pred_measurement
        mu = np.squeeze(self.mu_pred[:,i] + K@d)
        #saturating
        self.mu[:,i] = np.maximum(mu,np.zeros(self.n,))
        self.S = (np.eye(np.shape(A)[0]) + K@self.C)@S_predict

        #update Q if adaptive
        if self.adaptive:
            d = np.reshape(d,(self.m,1))
            self.Q = self.alpha*self.Q + (1-self.alpha)*K@d@d.T@K.T
        self.itr += 1
        return self.mu[:,i]

    def make_space(self):
        dim = np.shape(self.mu)
        if dim[1] == self.itr:
            self.mu = np.reshape(self.mu,(dim[0],dim[1]*2))
            self.mu = np.reshape(self.mu,(dim[0],dim[1]*2))

if __name__ == '__main__':
	main()
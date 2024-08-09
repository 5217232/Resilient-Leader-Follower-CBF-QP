import numpy as np
from random import randint
from scipy.integrate import odeint,solve_ivp 



class Agent:
    def __init__(self,location, color, palpha, ax, F):
        self.location = location.reshape(4,-1)
        self.locations = [[],[]]
        self.Uxs = []
        self.Uys = []
        self.color = color
        self.LED = None
        self.palpha = palpha
        self.body = ax.scatter([],[],c=color,alpha=palpha,s=50)
        self.obs_h = np.ones((1,2))
        self.obs_alpha =  2.0*np.ones((1,2))#
        self.value= randint(0,100)
        self.original = self.value
        self.connections = []
        self.F = F
        self.x = self.location.reshape(1,-1)[0][0:2]
        self.v = self.location.reshape(1,-1)[0][2:]
        self.previous = self.x
        self.prevprev = None
        self.values = []
        self.history = []

    def f(self):
        return np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
    
    def g(self):
        return np.array([ [0,0],[0,0],[1, 0],[0, 1] ])
    
    def step(self,U): #Just holonomic X,T acceleration
        self.U = U.reshape(2,1)
        self.prevprev = self.previous
        self.previous = self.x
        self.location = self.location + (self.f() @ self.location + self.g() @ self.U )*0.01
        self.x = (self.location.reshape(1,-1)[0][0:2]+ self.previous+self.prevprev)/3
        self.v = self.location.reshape(1,-1)[0][2:]
        self.render_plot()
        temp = np.array([self.U[0][0],self.U[1][0]])
        self.Us = np.append(self.Us,temp)
        return self.location

    def step2(self, U):
        self.U = U.reshape(2,-1)
        def model(t, y):
            dydt = self.f() @ y.reshape(4,-1) + self.g()@ self.U
            return dydt.reshape(-1,4)[0]
        steps = solve_ivp(model, [0,0.025], self.location.reshape(-1,4)[0])
        self.prevprev = self.previous
        self.previous = self.x
        what = np.array([steps.y[0][-1], steps.y[1][-1],steps.y[2][-1],steps.y[3][-1]])
        self.location = what.reshape(4,-1)
        # self.x = (self.location.reshape(1,-1)[0][0:2]+ self.previous+self.prevprev)/3
        self.x = self.location.reshape(1,-1)[0][0:2]
        self.v = what[2:]
        self.render_plot()
        self.Uxs = np.append(self.Uxs,U[0])
        self.Uys= np.append(self.Uys,U[0])
        return self.location

    def step3(self, loc):
        self.location = loc.reshape(-1,1)
        self.x = loc[0:2]
        self.v = loc[2:]
        self.render_plot()
    
    def rk4_step(self, U):
        self.U = U.reshape(2,-1)
        def dydt(x):
            return self.f() @ x + self.g() @ self.U

        f1 = dydt(self.location)
        f2 = dydt(self.location+0.5*0.01*f1)
        f3 = dydt(self.location+0.5*0.01*f2)
        f4 = dydt(self.location+0.01*f3)
        self.location = self.location + (0.01/6)*(f1+2*f2+2*f3+f4)
        self.x = self.location.reshape(1,-1)[0][0:2]
        self.v = self.location.reshape(1,-1)[0][2:]
        self.render_plot()
    
    def set_color(self):
        if self.value < 256:
            self.LED = (self.value/255, 0,0)
            return self.LED
        elif self.value < 511:
            self.LED = (0.1, (self.value-256)/255,0.0)
            return self.LED
        else:
            self.LED = (0.9, 0.5, (self.value-511)/255)
            return self.LED
    
    def render_plot(self):
        # scatter plot update
        self.locations[0] = np.append(self.locations[0], self.x[0])
        self.locations[1] = np.append(self.locations[1], self.x[1])
        # self.body.plot(self.locations[0][-2:], self.locations[1][-2:], self.LED)
        self.body.set_offsets(self.x)
        #animate(x)

    def agent_barrier(self,agent,d_min):
        h =  np.linalg.norm(self.x - agent.x)**2 - d_min**2 
        dh_dxi = 2*( self.x - agent.x)
        dh_dxj = -2*( self.x - agent.x)
        ddh_xi = 2*np.ones((1,2))
        dxi = dh_dxi.reshape(1,-1)[0]
        dxj = dh_dxj.reshape(1,-1)[0]
        return h, dxi, dxj, ddh_xi

    def connect(self, agent):
        self.connections.append(agent)

    def neighbors(self):
        return self.connections

    def propagate(self):
        if self.value!=self.original:
            for neigh in self.neighbors():
                neigh.receive(self.value)
        return self.value

    def receive(self, value):
        self.values.append(value)
    
    def w_msr(self):
        small_list=[];big_list=[];comb_list=[]
        if len(self.values)>=2*self.F+1:
            for aa in self.values:
                if aa<self.value:
                    small_list.append(aa)
                elif aa>self.value:
                    big_list.append(aa)
                else:
                    comb_list.append(aa)

            if len(small_list) <=self.F:
                small_list = []
            else:
                small_list = sorted(small_list)
                small_list = small_list[self.F:]

            if len(big_list) <=self.F:
                big_list = []
            else:
                big_list = sorted(big_list)
                big_list = big_list[:len(big_list)-self.F]

            comb_list = small_list+ comb_list + big_list
            total_list =len(comb_list)
            weight = 1/(total_list+1)
            weights = [weight for i in range(total_list)]
            avg = weight*self.value + sum([comb_list[i]*weights[i] for i in range(total_list)])

            self.value = avg
        self.history.append(self.value)
        self.values = []
        self.connections =[]


class Leaders(Agent):
    def __init__(self, value, location, color, palpha, ax, F):
        super().__init__(location, color, palpha, ax, F)
        self.value=value
        self.history = []

    def propagate(self):
        for neigh in self.neighbors():
            neigh.receive(self.value)
        return self.value
    def receive(self, value):
        pass
    def w_msr(self):
        self.history.append(self.value)
        self.values = []
        self.connections =[]
    
class Malicious(Leaders):
    def __init__(self, range, location, color, palpha, ax, F):
        self.range = range
        value = randint(range[0], range[1])
        super().__init__(value, location, color, palpha, ax, F)
    def propagate(self):
        self.value = randint(self.range[0], self.range[1])
        for neigh in self.neighbors():
            neigh.receive(self.value)
        return self.value
    


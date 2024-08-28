import numpy as np
from random import randint
from scipy.integrate import solve_ivp 
from statistics import median
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class Agent:
    def __init__(self,location, color, palpha, ax, F,shape="o"):
        self.location = location.reshape(2,-1)
        self.locations = [[],[]]
        self.Us = []
        self.color = color
        self.palpha = palpha
        if type(self)==Leaders:
            shape = "s"
        self.body = ax.scatter([],[],c=color,alpha=palpha,s=40, marker=shape)
        self.obs_h = np.ones((1,2))
        self.obs_alpha =  2.0*np.ones((1,2))#
        self.value= randint(0,200)
        self.original = self.value
        self.connections = []
        self.F = F
        self.values = []
        self.history = []
        self.set_color()

    def f(self):
        return np.array([0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [1, 0],[0, 1] ])
    
    def step(self,U, dt=0.02): #Just holonomic X,T acceleration
        self.U = U.reshape(2,1)
        self.location = self.location + (self.g() @ self.U )*dt
        self.render_plot()
        temp = np.array([self.U[0][0],self.U[1][0]])
        self.Us = np.append(self.Us,temp)
        return self.location

    def step2(self, U):
        self.U = U.reshape(2,-1)
        def model(t, y):
            dydt = self.g()@ self.U
            return dydt.reshape(-1,2)[0]
        steps = solve_ivp(model, [0,0.05], self.location.reshape(-1,2)[0])
        what = np.array([steps.y[0][-1], steps.y[1][-1]])
        self.location = what.reshape(2,-1)
        self.render_plot()
        temp = np.array([self.U[0][0],self.U[1][0]])
        self.Us = np.append(self.Us,temp)
        return self.location

    def step3(self, location):
        self.body.set_offsets(location)
    
    def render_plot(self):
        x = np.array([self.location[0][0],self.location[1][0]])
        # scatter plot update
        self.locations[0] = np.append(self.locations[0], x[0])
        self.locations[1] = np.append(self.locations[1], x[1])
        self.body.set_offsets([x[0],x[1]])
        # plt.plot(self.locations[0], self.locations[1])
        #animate(x)
    def set_color(self):
        # Get the colormap
        cmap = plt.get_cmap('viridis')
        # Map the value to the colormap
        rgba = cmap(self.value/1000)
        self.LED = [rgba[0], rgba[1], rgba[2]]
        return self.LED
    

    def agent_barrier(self,agent,d_min):
        h =  np.linalg.norm(self.location - agent.location)**2 - d_min**2 
        dh_dxi = 2*( self.location - agent.location[0:2]).T
        dh_dxj = -2*( self.location - agent.location[0:2] ).T
        return h, dh_dxi, dh_dxj

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
    def __init__(self, value, location, color, palpha, ax, F,marker="o"):
        super().__init__(location, color, palpha, ax, F,marker)
        self.value=value
        self.set_color()

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
    def __init__(self, range, location, color, palpha, ax, F,marker="o"):
        self.range = range
        value = randint(range[0], range[1])
        super().__init__(value, location, color, palpha, ax, F,marker)
    def propagate(self):
        self.value = randint(self.range[0], self.range[1])
        for neigh in self.neighbors():
            neigh.receive(self.value)
        return self.value
    




class Vector_Leaders(Leaders):
    def __init__(self, value, location, color, palpha, ax, F, dim):
        super().__init__(value, location, color, palpha, ax, F)
        self.dim = dim
        self.value=value
        self.history = [[] for i in range(self.dim)]

    def propagate(self):
        for neigh in self.neighbors():
            neigh.receive(self.value)
        return self.value
    def receive(self, value):
        pass
    def w_msr(self):
        for i in range(self.dim):
            self.history[i].append(self.value[i])
        self.values = []
        self.connections =[]
    def update_state(self,value):
        self.value = value


class Vector_Followers(Agent):
    def __init__(self, location, color, palpha, ax, F,dim):
        super().__init__(location, color, palpha, ax, F)
        self.dim= dim
        self.history = [[] for i in range(self.dim)]
        self.value = [randint(-500,500)/500 for i in range(self.dim)]
        self.values = [[] for i in range(self.dim)]
    def propagate(self):
        if len(self.value)!=0:
            for neigh in self.neighbors():
                neigh.receive(self.value)
        return self.value
    
    def receive(self, value):
        for i in range(self.dim):
            self.values[i].append(value[i])
    def median(self):
        self.value = []
        if len(self.values[0])>=2*self.F+1:
            for i in range(self.dim):
                med = median(self.values[i])
                self.value.append(med)
                self.history[i].append(med)
        else:
            for i in range(self.dim):
                self.history[i].append(0)
        self.connections =[]
        self.values = [[] for i in range(self.dim)]

    def w_msr(self):
        for i in range(self.dim):
            small_list=[];big_list=[];comb_list=[]
            if len(self.values[0])>=2*self.F+1:
                for aa in self.values[i]:
                    if aa<self.value[i]:
                        small_list.append(aa)
                    elif aa>self.value[i]:
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
                avg = weight*self.value[i] + sum([comb_list[i]*weights[i] for i in range(total_list)])

                self.value[i]= avg

        self.values = [[] for i in range(self.dim)]
        self.connections =[]

    

    

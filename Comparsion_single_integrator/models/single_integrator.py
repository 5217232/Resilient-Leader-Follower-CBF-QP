import numpy as np
from random import randint
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
        self.value= randint(0,1000)
        self.original = self.value
        self.connections = []
        self.F = F
        self.values = []
        self.history = [self.value]
        self.set_color()

    #Returns its \mathbf A matrix
    def f(self):
        return np.array([0,0]).reshape(-1,1)
    
    #Returns its \mathbf B matrix
    def g(self):
        return np.array([ [1, 0],[0, 1] ])
    
    #Updates the agent's state
    def step(self,U, dt=0.02):
        self.U = U.reshape(2,1)
        self.location = self.location + (self.g() @ self.U )*dt
        self.render_plot()
        temp = np.array([self.U[0][0],self.U[1][0]])
        self.Us = np.append(self.Us,temp)
        return self.location
    
    #Updates the agent's plot 
    def render_plot(self):
        x = np.array([self.location[0][0],self.location[1][0]])
        self.locations[0] = np.append(self.locations[0], x[0])
        self.locations[1] = np.append(self.locations[1], x[1])
        self.body.set_offsets([x[0],x[1]])

    #Maps the agent's consensus value to the LED color
    def set_color(self):
        cmap = plt.get_cmap('viridis')
        rgba = cmap(self.value/1000)
        self.LED = [rgba[0], rgba[1], rgba[2]]
        return self.LED
    
    #Computes the h and first derivative of the distance function between the agent and a given agent
    def agent_barrier(self,agent,d_min):
        h =  np.linalg.norm(self.location - agent.location)**2 - d_min**2 
        dh_dxi = 2*( self.location - agent.location[0:2]).T
        dh_dxj = -2*( self.location - agent.location[0:2] ).T
        return h, dh_dxi, dh_dxj

    #Forms an edge with the given agent.
    def connect(self, agent):
            self.connections.append(agent)
    
    #Returns its neighbor set
    def neighbors(self):
        return self.connections

    #Shares its consensus value with its neighbors
    def propagate(self):
        if self.value!=self.original:
            for neigh in self.neighbors():
                neigh.receive(self.value)
        return self.value

    #Receives neighbors' consensus values
    def receive(self, value):
        self.values.append(value)
    
    #Performs W-MSR
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
        self.history = [self.value]

    #Shares its consenus value with its neighbors
    def propagate(self):
        for neigh in self.neighbors():
            neigh.receive(self.value)
        return self.value
    
    def receive(self, value):
        pass
    
    #Throws away the stored values from its neighbors and resets its neighbor set
    def w_msr(self):
        self.history.append(self.value)
        self.values = []
        self.connections =[]
    
class Malicious(Leaders):
    def __init__(self, range, location, color, palpha, ax, F,marker="o"):
        self.range = range
        value = randint(range[0], range[1])
        super().__init__(value, location, color, palpha, ax, F,marker)

    #Randomly generates its consensus value and shares it with all of it neighbors.
    def propagate(self):
        self.value = randint(self.range[0], self.range[1])
        for neigh in self.neighbors():
            neigh.receive(self.value)
        return self.value
    
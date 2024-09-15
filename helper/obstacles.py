import numpy as np
import matplotlib.pyplot as plt

#Creates a circle object
class circle:
    def __init__(self,x,y,radius,ax,id):
        self.location = np.array([x,y]).reshape(-1,1)
        self.radius = radius
        self.id = id
        self.type = 'circle'
        self.x = self.location.reshape(1,-1)[0]
        self.render(ax)
    def render(self,ax):
        circ = plt.Circle((self.location[0],self.location[1]),self.radius,linewidth = 1, edgecolor='k',facecolor='k')
        ax.add_patch(circ)


import numpy as np
import cvxpy as cp
import eigenvalue as eig
import bootstrap_percolation as dp
from random import randint
from models.single_integrator import *
from models.obstacles import *
import matplotlib.pyplot as plt

'''
This code implements Baseline 2 in Environment 1. In Baseline 2, it uses a CBF-QP controller where CBF encodes 
the maintenance of r-robustness. Note it is an implementation of the work at https://ieeexplore.ieee.org/document/10354416.
'''

def unsmoothened_adjacency(dif, A, robots):
    n= len(robots)
    for i in range(n):
        for j in range(i+1, n):
            norm = np.linalg.norm(robots[i]-robots[j])
            if norm <= dif:
                A[i,j] =1
                A[j,i] =1


plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-5,5),ylim=(-5,7)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")


# Sim Parameters 
y_offset = -1.5
F = 1
broadcast_value = randint(0,1000)
inter_agent_collision =0.3
obtacle=0.7
dif =3
num_steps = 1000
leaders = 4
alpha = 2
inter_alpha = 1.5
robustness = 4
obs_alpha = 1.5

#Initialize robots
robots=[]
robots.append( Leaders(broadcast_value, np.array([-0.8,y_offset]),'b',1.0, ax,F))
robots.append( Leaders(broadcast_value, np.array([0,y_offset]),'b',1.0, ax, F))
robots.append( Leaders(broadcast_value, np.array([-1.1,y_offset - 1.2]),'b',1.0, ax, F))
robots.append( Leaders(broadcast_value, np.array([1,y_offset - 1.4]),'b',1.0, ax, F))
robots.append( Agent(np.array([-0.2,y_offset - 1.4]),'g',1.0, ax, F))
robots.append( Agent(np.array([-1.2,y_offset - 2.3]),'g',1.0 , ax, F))
robots.append( Agent(np.array([0.8,y_offset - 1.1]),'g',1.0 , ax, F))
robots.append( Malicious([0,1000],np.array([1.4,y_offset - 1.7]),'r',1.0 , ax, F))
robots.append( Agent(np.array([1.0,y_offset - 2.1]),'g',1.0 , ax, F))
robots.append( Agent(np.array([-0.8,y_offset - 1.6]),'g',1.0 , ax, F))
robots.append( Agent(np.array([0.2,y_offset - 0.5]),'g',1.0 , ax, F))
robots.append( Agent(np.array([-0.4,y_offset - 1.7]),'g',1.0 , ax, F))
robots.append( Agent(np.array([0.5,y_offset - 0.6]),'g',1.0 , ax, F))
robots.append( Agent(np.array([0.1,y_offset - 1.5]),'g',1.0 , ax, F))
robots.append( Agent(np.array([-0.5,y_offset - 1.1]),'g',1.0 , ax, F))

n= len(robots)
inter_agent = int(n*(n-1)/2)
########################## Make Obatacles ###############################
obstacles = []
index = 0
x1 = -1.2#-1.0
x2 = 1.2 #1.0
radius = 0.6
y_s = 0
obstacles.append(circle(-1.8, 1.7,radius,ax,0))
obstacles.append(circle(0, 4.0,radius,ax,0))
obstacles.append(circle(-0.5, 0.5,radius,ax,0))
obstacles.append(circle(1.5, 1.2,radius,ax,0))
obstacles.append(circle(-1.2, 2.3,radius,ax,0))
obstacles.append(circle(-0.5, 3.3,radius,ax,0))
obstacles.append( circle( 1.75,4,radius,ax,0 ) )


num_obstacles = len(obstacles)
########################################################################

# ##################################### 1: CBF Controller###################################
u1 = cp.Variable((2*n,1))
u1_ref = cp.Parameter((2*n,1),value = np.zeros((2*n,1)) )
num_constraints1  = 1 + inter_agent + num_obstacles*n
A1 = cp.Parameter((num_constraints1,2*n),value=np.zeros((num_constraints1,2*n)))
b1 = cp.Parameter((num_constraints1,1),value=np.zeros((num_constraints1,1)))
const1 = [A1 @ u1 >= b1]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref  ) )
cbf_controller = cp.Problem( objective1, const1 )
###################################################################################################

#Setting the goal
goal = []
goal.append(np.array([0,100]).reshape(2,-1))
goal.append(np.array([0,100]).reshape(2,-1))
goal.append(np.array([0,100]).reshape(2,-1))
goal.append(np.array([0,100]).reshape(2,-1))

counter =0
while True:
    robots_location = np.array([aa.location for aa in robots])

    #Compute the actual robustness
    A = np.zeros((n, n))
    unsmoothened_adjacency(dif, A, robots_location)
    delta = np.count_nonzero(A)
    dp.strongly_r_robust(A,leaders, delta)

    #Get the nominal control input \mathbf u_{nom}
    for i in range(n):
        vector = goal[i % leaders] - robots[i].location 
        vector = vector/np.linalg.norm(vector)
        u1_ref.value[2*i] = vector[0][0]
        u1_ref.value[2*i+1] = vector[1][0]
    #Perform W-MSR
    if counter/20 % 1==0:
        for i in range(n):
            for j in range(i+1,n):
                if A[i,j] ==1:
                    robots[i].connect(robots[j])
                    robots[j].connect(robots[i])
        #Agents share their values with neighbors
        for aa in robots:
            aa.propagate()
        # The followers perform W-MSR
        for aa in robots:
            aa.w_msr()
        # All the agents update their LED colors
        for aa in robots:
            aa.set_color()

    #Get the algebraic connectivity and the derivatives w.r.t. positions
    algebraic, beta = eig.mu_m(robots_location,dif)
    print("t:",counter*0.02," and edges:", delta)
    algebraic = np.real(algebraic)

    #Initialize the constraint of QP
    A1.value[0:,] = beta[:]
    b1.value[0] = -alpha*(algebraic-4-0.01) 

    #Inter-agent collision avoidance
    inter_count =1
    for i in range(n):
        for j in range(i+1,n):
            h, dh_dxi, dh_dxj = robots[i].agent_barrier(robots[j], inter_agent_collision)
            A1.value[inter_count][2*i:2*i+2] = dh_dxi
            A1.value[inter_count][2*j:2*j+2] = dh_dxj
            b1.value[inter_count] = -inter_alpha*h
            inter_count+=1
    obs_collision = 1 + inter_agent

    #Obstacle Collision avoidance
    for i in range(n):
        for j in range(num_obstacles):
            h, dh_dxi, dh_dxj = robots[i].agent_barrier(obstacles[j], obstacles[j].radius+0.1)
            A1.value[obs_collision][2*i:2*i+2] = dh_dxi
            b1.value[obs_collision] = -obs_alpha*h
            obs_collision+=1  
    
    print("eigenvalue:", algebraic)

    #Solve the CBF-QP and get the control input
    cbf_controller.solve(solver=cp.GUROBI)

    # implement control input
    for i in range(n):
        robots[i].step( u1.value[2*i:2*i+2]) 
    
    #If all robots have reached the exits, terminate
    for aa in robots_location:
        if aa[1]<=4.0:

            break
    else:
        break
    counter+=1
    
counter+=1
print("time:", counter*0.02)
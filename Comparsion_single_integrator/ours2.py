import numpy as np
import cvxpy as cp
import eigenvalue as eig
import bootstrap_percolation as dp
from random import randint
from models.single_integrator import *
from models.obstacles import *
import matplotlib.pyplot as plt
from jax import lax, jit, jacrev, hessian
from jax import numpy as jnp


plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-5,5),ylim=(-5,7)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Sim Parameters 
inter_agent_collision =0.3
epsilon = 0.0001
R =3
num_steps = 2000
leaders = 4
inter_alpha = 1.5
obs_alpha = 1.5
r = leaders-1
F = 1

#Initialize the robots
broadcast_value = randint(0,1000)
y_offset =-1.5
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


####################################### 1: CBF Controller###################################
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
goal.append(np.array([0, 100]).reshape(2,-1))
goal.append(np.array([0, 100]).reshape(2,-1))
goal.append(np.array([0, 100]).reshape(2,-1))
goal.append(np.array([0, 100]).reshape(2,-1))


#Build the parametrized sigmoid functions
q_A = 0.02
q = 0.02
s_A = 1.3
s = 1.5
sigmoid_A = lambda x: (1+q_A)/(1+(1/q_A)*jnp.exp(-s_A*x))-q_A
sigmoid = lambda x: (1+q)/(1+(1/q)*jnp.exp(-s*x))-q

######################Computes the \bar {\pi}_{\mathcal F}######################
@jit 
def barrier_func(x):
    def AA(x):
        A = jnp.zeros((n,n))
        def body_i(i, inputs1):
            def body_j(j, inputs):
                dis = R**2-jnp.sum((x[i]-x[j])**2)
                return lax.cond(dis>=0,lambda x: inputs.at[i,j].set(sigmoid_A(dis**2)), lambda x: inputs.at[i,j].set(0), dis) 
            return lax.fori_loop(0, n, body_j, inputs1)
        A = lax.fori_loop(0, n, body_i, A)

        def bodyD(i, inputs):
            return inputs.at[i,i].set(0.0)
        return lax.fori_loop(0, n, bodyD,A)
    
    def body(i, inputs):
        temp_x = A @ jnp.append( jnp.ones((leaders,1)), inputs, axis=0 )
        state_vector = sigmoid(temp_x[leaders:]-r)
        return state_vector
    
    state_vector = jnp.zeros((n-leaders,1))
    A = AA(x)
    delta = 4
    x = lax.fori_loop(0, delta, body, state_vector) 
    return x[:,0]

barrier_grad = jit(jacrev(barrier_func))

def smoothened_strongly_r_robust_simul(robots, R, r):   
    h = barrier_func(robots)
    h_dot = barrier_grad(robots)
    return h, h_dot
###############################################################################

#Set the weight vector \mathbf w
weight = np.array([7]*(n-leaders))

#Compiled the construction of robust maintenance CBF
compiled = jit(smoothened_strongly_r_robust_simul)

#Initialize the counter
counter = 0

while True:
    robots_location = np.array([aa.location for aa in robots])
    
    #Compute the actual robustness
    A = np.zeros((n, n))
    dp.unsmoothened_adjacency(R, A, robots_location)
    delta = np.count_nonzero(A)
    dp.strongly_r_robust(A,leaders, delta)

    #Get the nominal control input \mathbf u_{nom}
    for i in range(n):
        vector = goal[i % leaders] - robots[i].location 
        vector = vector/np.linalg.norm(vector)
        u1_ref.value[2*i] = vector[0][0]
        u1_ref.value[2*i+1] = vector[1][0]

    if counter/20 % 1==0:
        #Agents form a network
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
            
    # h_{3,c} and gradient
    x, der_  = compiled(robots_location, R, r)
    x=np.asarray(x);der_=np.asarray(der_)
    print("t:",counter*0.02," and edges:", delta)
    print(x)


    #Initialize the constraint of QP
    A1.value[0,:] = [0]*(2*n)

    #Calculate the Robustness CBF
    inter_count = 1
    hs = []
    for i in range(n-leaders):
        A1.value[0,:]=weight[i]*np.exp(-weight[i]*(x[i]-epsilon))*der_[i][:].reshape(1,-1)[0]
        hs.append(np.exp(-weight[i]*(x[i]-epsilon)))
    b1.value[0]= -2*(1-sum(hs))

    #Inter-agent collision avoidance 
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

    #Solve the CBF-QP and get the control input \mathbf u
    cbf_controller.solve(solver=cp.GUROBI)

    # implement control input \mathbf u and plot the trajectory
    for i in range(n):
        robots[i].step( u1.value[2*i:2*i+2]) 

        #Colors the trajectories in their current LED colors
        # if counter>0:
        #     plt.plot(robots[i].locations[0][counter-1:counter+1], robots[i].locations[1][counter-1:counter+1], color = robots[i].LED, zorder=0)            

    #Plots the environment and robots
    fig.canvas.draw()
    fig.canvas.flush_events()  

    #If all robots have reached the exits, terminate  
    for aa in robots_location:
        if aa[1]<=4.0:
            break
    else:
        break
    counter+=1
    
counter+=1
print("time:", counter*0.02)
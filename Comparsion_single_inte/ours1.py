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

y_offset =-1.5
F = 1
broadcast_value = randint(600,1000)

robots=[]
robots.append( Leaders(broadcast_value, np.array([-0.7,y_offset]),'b',1.0, ax,F))
robots.append( Leaders(broadcast_value, np.array([0,y_offset]),'b',1.0, ax, F))
robots.append( Leaders(broadcast_value, np.array([-0.3,y_offset - 2.1]),'b',1.0, ax, F))
robots.append( Leaders(broadcast_value, np.array([1.1,y_offset - 2.4]),'b',1.0, ax, F))
robots.append( Agent(np.array([-1.1,y_offset - 0.5]),'g',1.0, ax, F))
robots.append( Agent(np.array([-0.7,y_offset - 1.0]),'g',1.0 , ax, F))
robots.append( Agent(np.array([0.8,y_offset - 1.2]),'g',1.0 , ax, F))
robots.append( Malicious([0,500],np.array([1.1,y_offset - 1.7]),'r',1.0 , ax, F))
robots.append( Agent(np.array([-1.0,y_offset - 2.1]),'g',1.0 , ax, F))
robots.append( Agent(np.array([0.5,y_offset - 1.6]),'g',1.0 , ax, F))
robots.append( Agent(np.array([-0.4,y_offset - 1.9]),'g',1.0 , ax, F))
robots.append( Agent(np.array([-0.8,y_offset - 1.9]),'g',1.0 , ax, F))
robots.append( Agent(np.array([-0.5,y_offset - 2.4]),'g',1.0 , ax, F))
robots.append( Agent(np.array([0.1,y_offset - 1.3]),'g',1.0 , ax, F))
robots.append( Agent(np.array([0.5,y_offset - 2.2]),'g',1.0 , ax, F))

n= len(robots)
inter_agent = int(n*(n-1)/2)


########################## Make Obatacles ###############################
obstacles = []
index = 0
x1 = -1.2#-1.0
x2 = 1.2 #1.0
radius = 0.6
y_s = 0
y_increment = 0.3
for i in range(int( 5/y_increment )):
    obstacles.append( circle( x1,y_s,radius,ax,0 ) ) # x,y,radius, ax, id
    obstacles.append( circle( x2,y_s,radius,ax,1 ) )
    y_s = y_s + y_increment
obstacles.append(circle(-0.9,0.5,radius,ax,0))
obstacles.append(circle(0.9,1.0,radius,ax,0))
obstacles.append(circle(-0.9,1.9,radius,ax,0))
obstacles.append(circle(0.9,2.5,radius,ax,0))
obstacles.append(circle(0.0,4.0,.3,ax,0))

obstacles.append(circle(-1.4, 0.1,radius,ax,0))
obstacles.append(circle(1.4, 0.1,radius,ax,0))
obstacles.append(circle(-1.6, -0.1,radius,ax,0))
obstacles.append(circle(1.6, -0.1,radius,ax,0))

num_obstacles = len(obstacles)
########################################################################

total = 1
# ##################################### 1: CBF Controller###################################
u1 = cp.Variable((2*n,1))
u1_ref = cp.Parameter((2*n,1),value = np.zeros((2*n,1)) )
num_constraints1  = total + inter_agent + num_obstacles*n
A1 = cp.Parameter((num_constraints1,2*n),value=np.zeros((num_constraints1,2*n)))
b1 = cp.Parameter((num_constraints1,1),value=np.zeros((num_constraints1,1)))
const1 = [A1 @ u1 >= b1]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref  ) )
cbf_controller = cp.Problem( objective1, const1 )
###################################################################################################

inter_agent_collision =0.3
obtacle=0.7
dif =2.5
num_steps = 2000
leaders = 4
goal = []
goal.append(np.array([0, 100]).reshape(2,-1))
goal.append(np.array([0, 100]).reshape(2,-1))
goal.append(np.array([0, 100]).reshape(2,-1))
goal.append(np.array([0, 100]).reshape(2,-1))
alpha = 10
inter_alpha = 2
obs_alpha = 2

num_edges =[]
counter = 0
r = leaders-1

q1 = 0.02
p1 = jnp.log(1/q1)
q2 = 0.02
p2 = jnp.log(1/q2)
s_A = 8
s = 3
relu = lambda x: (1+q1)/(1+jnp.exp(-s_A*x+p1))-q1
relu2 = lambda x: (1+q2)/(1+jnp.exp(-s*x+ p2))-q2

@jit 
def barrier_func(x):
    def AA(x):
        A = jnp.array([[0.0 for i in range(n)] for j in range(n)]) 
        for i in range(n):
            for j in range(i+1, n):
                dis = dif-jnp.linalg.norm(x[i]-x[j])
                A = A.at[j,i].set(relu(dis))
                A = A.at[i,j].set(relu(dis))  
        return A
    def body(i, inputs):
        temp_x = A @ jnp.concatenate([jnp.array([1.0 for p in range(leaders)]),inputs])
        state_vector = relu2(temp_x[leaders:]-r)
        return state_vector
    
    state_vector = jnp.array([0.0 for p in range(n-leaders)])
    A = AA(x)
    delta = 5
    x = lax.fori_loop(0, delta, body, state_vector) 
    return x

barrier_grad = jit(jacrev(barrier_func))

def smoothened_strongly_r_robust_simul(robots, dif, r):   
    h = barrier_func(robots)
    h_dot = barrier_grad(robots)
    return h, h_dot


weight = np.array([5]*(n-leaders))
compiled = jit(smoothened_strongly_r_robust_simul)

while True:
    robots_location = np.array([aa.location for aa in robots])
    A = np.zeros((n, n))
    unsmoothened_adjacency(dif, A, robots_location)
    delta = np.count_nonzero(A)
    dp.strongly_r_robust(A,leaders, delta)


    for i in range(n):
        vector = goal[i % leaders] - robots[i].location 
        vector = vector/np.linalg.norm(vector)
        u1_ref.value[2*i] = vector[0][0]
        u1_ref.value[2*i+1] = vector[1][0]
    #Perform W-MSR
    if counter/25 % 1==0:
        for i in range(n):
            for j in range(i+1,n):
                if A[i,j] ==1:
                    robots[i].connect(robots[j])
                    robots[j].connect(robots[i])
        for aa in robots:
            aa.propagate()
        for aa in robots:
            aa.w_msr()
        for aa in robots:
            aa.set_color()
            
    x, der_  = compiled(robots_location, dif, r)
    x=np.asarray(x);der_=np.asarray(der_)
    print("t:",counter*0.02," and edges:", delta)
    print(x)
    inter_count = total
    hs = []
    A1.value[0,:] = [0]*(2*n)

    for i in range(n-leaders):
        A1.value[0,:]=weight[i]*np.exp(-weight[i]*x[i])*der_[i][:].reshape(1,-1)[0]
        hs.append(np.exp(-weight[i]*x[i]))
    b1.value[0]= -0.5*(1-sum(hs))
    for i in range(n):
        for j in range(i+1,n):
            h, dh_dxi, dh_dxj = robots[i].agent_barrier(robots[j], inter_agent_collision)
            A1.value[inter_count][2*i:2*i+2] = dh_dxi
            A1.value[inter_count][2*j:2*j+2] = dh_dxj
            b1.value[inter_count] = -inter_alpha*h
            inter_count+=1
    obs_collision = total + inter_agent
    for i in range(n):
        for j in range(num_obstacles):
            h, dh_dxi, dh_dxj = robots[i].agent_barrier(obstacles[j], obstacles[j].radius+0.1)
            A1.value[obs_collision][2*i:2*i+2] = dh_dxi
            b1.value[obs_collision] = -obs_alpha*h
            obs_collision+=1  



    # for i in range(n-leaders):
    #     A1.value[0,:]=weight[i]*np.exp(-weight[i]*x[i])*der_[i][:].reshape(1,-1)[0]
    #     hs.append(-weight[i]*x[i])
    # inter_count =[];obs=[]
    # for i in range(n):
    #     for j in range(i+1,n):
    #         weighted =12
    #         h, dh_dxi, dh_dxj = robots[i].agent_barrier(robots[j], inter_agent_collision)
    #         A1.value[0][2*i:2*i+2]+= dh_dxi.reshape(1,-1)[0]*np.exp(-weighted*h)*weighted
    #         A1.value[0][2*j:2*j+2]+= dh_dxj.reshape(1,-1)[0]*np.exp(-weighted*h)*weighted
    #         inter_count.append(-weighted*h)
    # for i in range(n):
    #     for j in range(num_obstacles):
    #         weighted = 12
    #         h, dh_dxi, dh_dxj = robots[i].agent_barrier(obstacles[j], obstacles[j].radius+0.1)
    #         A1.value[0][2*i:2*i+2]+= dh_dxi.reshape(1,-1)[0]*np.exp(-weighted*h)*weighted
    #         obs.append(-weighted*h)
    # b1.value[0] = -1.5*(1-np.sum(np.exp(hs + inter_count + obs) ))

    cbf_controller.solve(solver=cp.GUROBI)
    for i in range(n):
        robots[i].step( u1.value[2*i:2*i+2]) 
        

    fig.canvas.draw()
    fig.canvas.flush_events()    
    for aa in robots_location:
        if aa[1]<=4.0:
            break
    else:
        break
    counter+=1
    

counter+=1
print("time:", counter*0.02)
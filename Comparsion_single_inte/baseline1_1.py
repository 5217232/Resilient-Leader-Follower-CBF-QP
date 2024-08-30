import numpy as np
import cvxpy as cp
import eigenvalue as eig
import bootstrap_percolation as dp
from random import randint
from models.single_integrator import *
from models.obstacles import *
import matplotlib.pyplot as plt
import networkx as nx

def adjacency_undirected(edges,A):
    for (i,j) in edges:
        A[i][j]=1
        A[j][i]=1

def generate_random_strongly_r_g(n, leaders, AA):
    G = nx.gnp_random_graph(n, 0.3)
    A = np.zeros((n,n))
    adjacency_undirected(G.edges,A)
    while True:
        nice = False
        if dp.strongly_r_robust(A,leaders, int(np.sum(A)/2)) == leaders-1:
            for (i,j) in G.edges:
                nice = (AA[i,j]==A[i,j])
                if not nice:
                    break
            else:
                break
        G = nx.gnp_random_graph(n, 0.3)
        A = np.zeros((n,n))
        adjacency_undirected(G.edges,A)
    return G.edges



def unsmoothened_adjacency(R, A, robots):
    n= len(robots)
    for i in range(n):
        for j in range(i+1, n):
            norm = np.linalg.norm(robots[i]-robots[j])
            if norm <= R:
                A[i,j] =1
                A[j,i] =1


y_offset =-1.5
F = 0
leaders = 4
R =2.5
inter_agent_collision =0.3

num_steps = 2500
goal = []
goal.append(np.array([0, 100]).reshape(2,-1))
goal.append(np.array([0, 100]).reshape(2,-1))
goal.append(np.array([0, 100]).reshape(2,-1))
goal.append(np.array([0, 100]).reshape(2,-1))
inter_alpha = 1.5
robustnfess = 4
obs_alpha = 1.5
times_exceeded =0
container =[]

for ii in range(50):

    broadcast_value = randint(0,1000)
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(xlim=(-5,5),ylim=(-5,10)) 
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
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

    robots=[]
    robots.append( Leaders(broadcast_value, np.array([-0.7,y_offset]),'b',1.0, ax,F))
    robots.append( Leaders(broadcast_value, np.array([0,y_offset]),'b',1.0, ax, F))
    robots.append( Leaders(broadcast_value, np.array([-0.3,y_offset - 2.1]),'b',1.0, ax, F))
    robots.append( Leaders(broadcast_value, np.array([1.1,y_offset - 2.4]),'b',1.0, ax, F))
    robots.append( Agent(np.array([-1.1,y_offset - 0.5]),'g',1.0, ax, F))
    robots.append( Agent(np.array([-0.7,y_offset - 1.0]),'g',1.0 , ax, F))
    robots.append( Agent(np.array([0.8,y_offset - 1.2]),'g',1.0 , ax, F))
    robots.append( Malicious([0,1000],np.array([1.1,y_offset - 1.7]),'r',1.0 , ax, F))
    robots.append( Agent(np.array([-1.0,y_offset - 2.1]),'g',1.0 , ax, F))
    robots.append( Agent(np.array([0.5,y_offset - 1.6]),'g',1.0 , ax, F))
    robots.append( Agent(np.array([-0.4,y_offset - 1.9]),'g',1.0 , ax, F))
    robots.append( Agent(np.array([-0.8,y_offset - 1.9]),'g',1.0 , ax, F))
    robots.append( Agent(np.array([-0.5,y_offset - 2.4]),'g',1.0 , ax, F))
    robots.append( Agent(np.array([0.1,y_offset - 1.3]),'g',1.0 , ax, F))
    robots.append( Agent(np.array([0.5,y_offset - 2.2]),'g',1.0 , ax, F))

    n= len(robots)
    inter_agent = int(n*(n-1)/2)
    robots_location = np.array([aa.location for aa in robots])

    A = np.zeros((n, n))
    unsmoothened_adjacency(R, A, robots_location)
    randomly_generated_fixed_topology = generate_random_strongly_r_g(n, leaders, A)


    total = len(randomly_generated_fixed_topology)
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

    counter=0

    while True:
        robots_location = np.array([aa.location for aa in robots])
        A = np.zeros((n, n))
        unsmoothened_adjacency(R, A, robots_location)
        delta = np.count_nonzero(A)
        dp.strongly_r_robust(A,leaders, delta)
        print(ii, "-t:",counter*0.02," and edges:", delta)

        for i in range(leaders):
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
            for aa in robots:
                aa.propagate()
            for aa in robots:
                aa.w_msr()
            for aa in robots:
                aa.set_color()
                
        connections = 0
        for (i,j) in randomly_generated_fixed_topology:
            h = R - np.linalg.norm(robots_location[i]-robots_location[j])
            dh_dxi= -2*(robots_location[i]-robots_location[j]).T
            A1.value[connections][2*i:2*i+2] = dh_dxi
            A1.value[connections][2*j:2*j+2] = -dh_dxi
            b1.value[connections] = -0.9*h
            connections+=1
        inter_count = total

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

        cbf_controller.solve(solver=cp.GUROBI)
        for i in range(n):
            robots[i].step( u1.value[2*i:2*i+2]) 
            # if counter>0:
            #     plt.plot(robots[i].locations[0][counter-1:counter+1], robots[i].locations[1][counter-1:counter+1], color = robots[i].LED, zorder=0)            

        fig.canvas.draw()
        fig.canvas.flush_events()    
        for aa in robots_location:
            if aa[1]<=4.0:
                break
        else:
            break
        if counter >= num_steps:
            break
        counter+=1

            # writer.grab_frame()

    counter+=1
    if counter>2500:
        times_exceeded+=1
    else:
        container.append(counter)
    print("So far: exceeded:", times_exceeded, "and", container)

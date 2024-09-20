import numpy as np
import cvxpy as cp
import eigenvalue as eig
import bootstrap_percolation as dp
from random import randint
from models.single_integrator import *
from models.obstacles import *
import matplotlib.pyplot as plt
import networkx as nx



'''
This code implements Baseline 1 in Environment 1. In Baseline 1, it uses a CBF-QP controller where CBFs encode 
the maintenance of a given graph topology. We run simulations in Environment 1 100 times with randomly generated strongly 
r-robust graph. The simulation terminates when either 1) all robots reach the exits or 2) go over the 
given time thresold (set as 50 s). 
'''

#Computes the unweighted adjacency matrix for undirected graphs.
def adjacency_undirected(edges,A):
    for (i,j) in edges:
        A[i][j]=1
        A[j][i]=1

#Randomly generate a strongly r-robust graph.
def generate_random_strongly_r_g(n, leaders, AA):
    #Randomly generate a graph and compute its adjancy matrix.
    G = nx.gnp_random_graph(n, 0.3)
    A = np.zeros((n,n))
    adjacency_undirected(G.edges,A)
    while True:
        graph_fit = False
        #Check if the randomly generated graph is strongly (leader-1)-robust.
        if dp.strongly_r_robust(A,leaders, int(np.sum(A)/2)) == leaders-1:
            #Check if the randomly generated graph fits the initial positions of robots. If so, break. Otherwise, go back.
            for (i,j) in G.edges:
                graph_fit = (AA[i,j]==A[i,j])
                if not graph_fit:
                    break
            else:
                break
        G = nx.gnp_random_graph(n, 0.3)
        A = np.zeros((n,n))
        adjacency_undirected(G.edges,A)
    return G.edges


#computes the distance-based adjacency (proximity) matrix with communication range R and positions of robots.
def unsmoothened_adjacency(R, A, robots):
    n= len(robots)
    for i in range(n):
        for j in range(i+1, n):
            norm = np.linalg.norm(robots[i]-robots[j])
            if norm <= R:
                A[i,j] =1
                A[j,i] =1


#sim parameters
y_offset =-1.5
F = 1
leaders = 4
R =2.5
inter_agent_collision =0.3
num_steps = 2500
inter_alpha = 2
robustnfess = 4
obs_alpha = 2
times_exceeded =0
alpha = 0.5

#Setting the goal positions
goal = []
goal.append(np.array([0, 100]).reshape(2,-1))
goal.append(np.array([0, 100]).reshape(2,-1))
goal.append(np.array([0, 100]).reshape(2,-1))
goal.append(np.array([0, 100]).reshape(2,-1))

container =[]

#Run 100 simulations
for ii in range(50):
    broadcast_value = randint(0,1000)
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

        #Set the nominal control input
        for i in range(leaders):
            vector = goal[i % leaders] - robots[i].location 
            vector = vector/np.linalg.norm(vector)
            u1_ref.value[2*i] = vector[0][0]
            u1_ref.value[2*i+1] = vector[1][0]

        #Perform W-MSR
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


        #Graph topology maintenace CBF
        connections = 0
        for (i,j) in randomly_generated_fixed_topology:
            h = R - np.linalg.norm(robots_location[i]-robots_location[j])
            dh_dxi= -2*(robots_location[i]-robots_location[j]).T
            A1.value[connections][2*i:2*i+2] = dh_dxi
            A1.value[connections][2*j:2*j+2] = -dh_dxi
            b1.value[connections] = -alpha*h
            connections+=1
        inter_count = total


        #Inter-agent collision avoidance
        for i in range(n):
            for j in range(i+1,n):
                h, dh_dxi, dh_dxj = robots[i].agent_barrier(robots[j], inter_agent_collision)
                A1.value[inter_count][2*i:2*i+2] = dh_dxi
                A1.value[inter_count][2*j:2*j+2] = dh_dxj
                b1.value[inter_count] = -inter_alpha*h
                inter_count+=1
        obs_collision = total + inter_agent

        #Obstacle Collision avoidance
        for i in range(n):
            for j in range(num_obstacles):
                h, dh_dxi, dh_dxj = robots[i].agent_barrier(obstacles[j], obstacles[j].radius)
                A1.value[obs_collision][2*i:2*i+2] = dh_dxi
                b1.value[obs_collision] = -obs_alpha*h
                obs_collision+=1  

        #Solve the CBF-QP and get the control input
        cbf_controller.solve(solver=cp.GUROBI)

        #Implement the control input
        for i in range(n):
            robots[i].step( u1.value[2*i:2*i+2]) 
        
        #If all the robots have reached the exists, terminate 
        for aa in robots_location:
            if aa[1]<=4.0:
                break
        else:
            break
        #If over time, terminate
        if counter >= num_steps:
            break
        counter+=1
  
    counter+=1
    if counter>2500:
        times_exceeded+=1
    else:
        container.append(counter)
    print("So far: exceeded:", times_exceeded, "and", container)

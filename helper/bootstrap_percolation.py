import numpy as np
from random import randint


def adjacency_undirected(edges,A):
    for (i,j) in edges:
        A[i][j]=1
        A[j][i]=1

def sigmoid(value,v):
    return 1/(1+np.exp(-5.5*(value-v)))


def smoothened_adjacency(dif, A, robots):
    sigmit = lambda x: np.exp(-10*(np.sqrt(((x[0]-x[2])**2+(x[1]-x[3])**2))-x[4]))/(1+np.exp(-10*(np.sqrt(((x[0]-x[2])**2+(x[1]-x[3])**2))-x[4])))
    n = len(A)
    for i in range(n):
        for j in range(i+1, n):
            A[j][i] = sigmit([robots[j][0], robots[j][1], robots[i][0],robots[i][1], dif-0.5])
            A[i][j] = sigmit([robots[i][0], robots[i][1], robots[j][0],robots[j][1], dif-0.5]) 

def unsmoothened_adjacency(dif, A, robots):
    n = len(A)
    for i in range(n):
        for j in range(i+1, n):
            norm = np.linalg.norm(robots[i]-robots[j])
            if norm <= dif:
                A[i][j] =1
                A[j][i] = 1

def rdeg(n,r):
    groups = int(n/r)
    edges=[]
    for i in range(groups-1):
        for j in range(r):
            for z in range(r):
                edges.append((r*i+j, r*(i+1)+z))
    for i in range(r*(groups-1), r*(groups)):
        for j in range(r*groups, n):
            edges.append((i,j))
    return edges


def strongly_r_robust(A,leaders, delta):
    n= len(A)
    max_r = leaders
    ans_r =0
    max_r+=1
    for r in range(1,max_r+1):
        x = np.array([1 for p in range(leaders)] + [0 for p in range(n-leaders)])
        for i in range(delta):
            temp_x = A @ x
            temp_x = np.array([np.heaviside(temp_x[k]-r,1) for k in range(n)])
            x = x + temp_x
            x= [np.heaviside(x[k]-1,1) for k in range(n)]
        if (x>=np.array([1 for i in range(n)])).all():
            ans_r=r

                
    print("real_r_robustness:",ans_r)
    return ans_r



# edges = [(5, 4), (5, 3), (5, 2), (5, 1), (4, 3), (4, 2), (4, 1), (3, 2), (2, 1), (3, 0), (2, 0), (1, 0)]
# while True:
#     temp = []
#     for n in range(6):
#         temp.append(np.array([randint(-100,100)/10,randint(-100,100)/10]))
#     for i in range(6):
#         for j in range(i+1,6):
#             if np.linalg.norm(temp[i]-temp[j])<0.3:
#                 breake = True
#                 break
#             if (i,j) in edges or (j,i) in edges:
#                 if np.linalg.norm(temp[i]-temp[j])>3:
#                     breake = True
#                     break
#             else:
#                 if np.linalg.norm(temp[i]-temp[j])<3:
#                     breake = True
#                     break
#             if breake:
#                 break
#         if breake:
#             break
#     if breake:
#         continue
#     print(temp)
#     break




def smoothened_strongly_r_robust(A,leaders):
    n= len(A)
    max_r = int(n/2) 
    ans_r =0
    if n % 2==1:
        max_r+=1
    delta = np.sum(A>=0.5)
    delta = int(n*(n-1)/2)
    for r in range(1,max_r+1):
        x = np.array([1 for p in range(leaders)] + [0 for p in range(n-leaders)])
        for i in range(delta):
            temp_x = A @ x
            temp_x = np.array([sigmoid(temp_x[k],r) for k in range(n)])
            x = x + temp_x
            x = [sigmoid(x[k],0.5) for k in range(n)]
            if (x >= np.array([0.5 for i in range(n)])).all():
                break
        if r==4:
            print(x)
        if (x >= np.array([0.5 for i in range(n)])).all():
            ans_r=r
    print(ans_r)
    return ans_r


# leaders = 4
# r = 2
# num_robots =n= 30
# dif =6
# robots=[]
# robots.append(np.array([10,0]))
# robots.append(np.array([10,3]))
# robots.append(np.array([10,4]))
# robots.append(np.array([8,9]))
# robots.append(np.array([6,10]))
# e = rdeg(n,r)
# A = np.array([[0 for i in range(n)] for j in range(n)], dtype=float)
# adjacency_undirected(e,A)
# #unsmoothened_adjacency(dif, A, e)
# smoothened_strongly_r_robust(A, leaders)
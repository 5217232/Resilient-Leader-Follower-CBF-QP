import numpy as np

#Constructs a sigmoid function 
def sigmoid(value,v):
    return 1/(1+np.exp(-5.5*(value-v)))

#Computes the "discrete" adjacency matrix with the radis R, empty n by n matrix A, and list of robots' poisitons
def unsmoothened_adjacency(R, A, robots):
    n= len(robots)
    for i in range(n):
        for j in range(i+1, n):
            norm = np.linalg.norm(robots[i]-robots[j])
            if norm < R:
                A[i][j] =1
                A[j][i] = 1

#Computes the strong r robustness of a graph with its adjacency matrix A, number of leaders, and number of iterations delta
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

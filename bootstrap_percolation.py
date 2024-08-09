import numpy as np

#Calculates the adjacency matrix 
def unsmoothened_adjacency(dif, A, robots):
    n= len(robots)
    for i in range(n):
        for j in range(i+1, n):
            norm = np.linalg.norm(robots[i]-robots[j])
            if norm <= dif:
                A[i,j] =1
                A[j,i] =1

#Calculates the real strong r robustness with an adjacency matrix
def strongly_r_robust(A,leaders):
    n= len(A)
    max_r = int(n/2) 
    ans_r =0
    if n % 2==1:
        max_r+=1
    for r in range(1,max_r+1):
        x = np.zeros((n-leaders,1))
        while True:
            before_x = x
            temp_x = A @ np.concatenate([np.ones((leaders,1)),x])
            x = np.heaviside(temp_x[leaders:]-r,1)
            if np.all(x==before_x):
                if np.all(x==np.ones((n-leaders,1))):
                    ans_r =r
                break
    print("real_r_robustness:",ans_r)
    return ans_r


#Calculates the real strong r robustness with robots' locations
def strongly_r_robust_loc(leaders,robots_location, dif):
    n = len(robots_location)
    A = np.zeros((n, n))
    unsmoothened_adjacency(dif, A ,robots_location)
    max_r = int(n/2) 
    ans_r =0
    if n % 2==1:
        max_r+=1
    for r in range(1,max_r+1):
        x = np.zeros((n-leaders,1))
        while True:
            before_x = x
            temp_x = A @ np.concatenate([np.ones((leaders,1)),x])
            x = np.heaviside(temp_x[leaders:]-r,1)
            if np.all(x==before_x):
                if np.all(x==np.ones((n-leaders,1))):
                    ans_r =r
                break
    print("real_r_robustness:",ans_r)
    return ans_r
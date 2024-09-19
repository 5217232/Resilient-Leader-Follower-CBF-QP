import numpy as np
from scipy import linalg as LA


'''This is a helper library for the baseline2_1 and baseline2_2. '''

def smoothened_adjacency(robots,R,n):
    A = np.zeros((n,n))
    temp =[]
    for i in range(n):
        for j in range(i+1, n):
            dist = (R**2-np.linalg.norm(robots[i]-robots[j])**2)**2
            if np.linalg.norm(robots[i]-robots[j])>R:
                dist = 1
            temp.append(dist)
    temp.sort()  
    largest= temp[-1]
    sigma = largest/np.log(2)
    if np.exp(largest)<=2:
        sigma = 1
    for i in range(n):
        for j in range(i+1, n):
            dist = (R**2-np.linalg.norm(robots[i]-robots[j])**2)**2
            A[i][j]=np.exp(dist/sigma)-1
            A[j][i]=np.exp(dist/sigma)-1
            if np.linalg.norm(robots[i]-robots[j])>R:
                A[i][j]=0
                A[j][i]=0
    return A,sigma


slope = 6
sigmoid = lambda x: 1/(1+np.exp(-slope*x))

def smoothened_adjacency(robots,R,n):
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            dist = R-np.linalg.norm(robots[i]-robots[j])
            A[i][j]=sigmoid(dist)
            A[j][i]=sigmoid(dist)
    return A


#Computes the unweighted adjacency matrix based on the distances of robots, communication range R, and number of robots n.
def unsmoothened_adjacency(robots,R,n):
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            norm = np.linalg.norm(robots[i]-robots[j])
            if norm <= R:
                A[i,j] =1.0
                A[j,i] =1.0
    return A

#Computes the derivative of algebraic connectivity with respect to the positions of robots.
def mu_m(robots_location,R):
    n = len(robots_location)
    # A, sigma =smoothened_adjacency(robots_location,R,n)
    A =smoothened_adjacency(robots_location,R,n)
    AA = unsmoothened_adjacency(robots_location,R,n)
    D = np.diag( np.sum( A, axis = 1 ) )
    L = D-A
    e_vals, e_vecs = LA.eig(L)
    e_vals = np.asarray(e_vals);e_vecs = np.real(np.asarray(e_vecs))
    idx = e_vals.argsort()[::-1]   
    e_vals = e_vals[idx][-2]
    e_vecs = e_vecs[:,idx][-2]
    e_vecs = e_vecs/np.linalg.norm(e_vecs)
    beta = []
    for i in range(n):
        sum=np.array([[0.0],[0.0]])
        for j in range(n):
            if i==j:
                continue
            if AA[i][j]==1:
                dist = R-np.linalg.norm(robots_location[i]-robots_location[j])
                der = slope*np.exp(-slope*dist)/(1+np.exp(-slope*dist))**2
                temp = -(robots_location[i]-robots_location[j])/np.linalg.norm(robots_location[i]-robots_location[j])
                if np.linalg.norm(robots_location[i]-robots_location[j]) ==0:
                    temp =0
                sum+=der*temp*(e_vecs[i]-e_vecs[j])**2
        beta.append(sum[0]);beta.append(sum[1])
    return e_vals, np.array(beta).reshape(1,-1)




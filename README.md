# Resilient-Leader-Follower-CBF-QP-Controller

This repository contains the implementation of Strong $r$-robustness maintenance CBF-QP Controller. This guarantees the multi-robot network to maintain the sufficient number of connections among robots such a way that the robots can perform leader-follower consensus despite the presence of misbehaving agents. Contrary to the previous works that either maintain graph topologies with known robustness or control indirect measures to control robustness, our CBF directly addresses robustness without fixed formations. The algorithm also includes inter-agent and obstacle collision avoidances encoded as additional HOCBFs. For more information, see our paper.


# Dependencies
This repository requires the following libraries: **numpy**, **matplotlib**, **jax**, and **jaxlib**.

For solving the CBF-QP, we need the following libraries: **cvxpy** and **gurobi**

# How to Run
In this repository, we have three different simulations: `spread_out_sim`, `complex_sim1`, and `complex_sim2`.
1) `spread_out_sim` shows the simulation of robots spreading out in an open space.
2) `complex_sim1` and `complex_sim2` show the simulation of robots going through some complex environments with obstacles.
   
To run each simulation, you just need to run the corresponding python file. 


# spread_out_sim
https://github.com/user-attachments/assets/2f664a28-7aa2-4345-9de8-e39ead7725a8

# complex_sim1
https://github.com/user-attachments/assets/97d9329a-348e-464c-903e-4e6657dbe9bb
 
# complex_sim2
https://github.com/user-attachments/assets/6485eaa9-4f78-4334-ada4-f47b82e0a8cb








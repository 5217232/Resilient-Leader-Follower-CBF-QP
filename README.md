# Resilient-Leader-Follower-CBF-QP-Controller

This repository contains the implementation of Strong $r$-robustness maintenance Control Barrier Function-based Quadratic Programming (CBF-QP) Controller. This guarantees the multi-robot network to maintain the sufficient number of connections among robots such a way that the robots can achieve leader-follower consensus despite the presence of misbehaving agents. Contrary to the previous works that assume robots can maintain predetermined graph topologies with known robustness properties, our CBF directly addresses robustness without imposing any fixed topology. This allows the robots to flexibely form reconfigurable communication networks with desired levels of strong $r$-robustness while navigating in spatially restrictive environments. The algorithm also includes inter-agent and obstacle collision avoidances encoded as additional HOCBFs. We present three different scenarios, whose videos are shown below. For more information, please refer to our paper.

# Hardware Experiment
Hardware experiments can be seen at here.


# Dependencies
This repository requires the following libraries: **numpy**, **matplotlib**, **jax**, and **jaxlib**.

For solving the CBF-QP, we need the following libraries: **cvxpy** and **gurobi**

# How to Run
In this repository, we have three different simulations: `spread_out_sim`, `complex_sim1`, and `complex_sim2`.
1) `spread_out_sim` shows the simulation of robots spreading out in an open space.
2) `complex_sim1` and `complex_sim2` show the simulation of robots going through some complex environments with obstacles.
   
To run each simulation scenario, you just need to run the corresponding python files. 

# spread_out_sim
https://github.com/user-attachments/assets/2f664a28-7aa2-4345-9de8-e39ead7725a8

# complex_sim1
https://github.com/user-attachments/assets/97d9329a-348e-464c-903e-4e6657dbe9bb
 
# complex_sim2
https://github.com/user-attachments/assets/6485eaa9-4f78-4334-ada4-f47b82e0a8cb

# Baselines
You can test the baseline algorithms:

Baseline 1 implements a CBF-QP controller whose CBF encodes maintenance of randomly genreated strong $r$-robust graph topologies.

Baseline 2 implements a CBF-QP controller whose CBF encodes maintenace of $r$-robustness. Check [this](https://ieeexplore.ieee.org/document/10354416) for more information.

We have implemented the codes in the settings of the `complex_sim1` and `complex_sim2`. Note the simulations for comparisons are all implemented in single integrator dynamics.
The python files can be found at `Comparsion_single_integrator` folder. 










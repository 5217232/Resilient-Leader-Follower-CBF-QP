# Resilient-Leader-Follower-CBF-QP-Controller

This repository contains the implementation of Strong $r$-robustness maintenance CBF-QP Controller. This guarantees the multi-robot network to maintain the sufficient number of connections among robots such a way that the robots can perform leader-follower consensus despite the presence of misbehaving agents. Contrary to the previous works that either maintain graph topologies with known robustness or control indirect measures to control robustness, our CBF directly addresses robustness without fixed formations. The algorithm also includes inter-agent and obstacle collision avoidances HOCBFs. For more information, see our paper.


# Dependencies
This repository requires the following libraries: **numpy**, **matplotlib**, **jax**, and **jaxlib**.

For solving the CBF-QP, we need the following libraries: **cvxpy** and **gurobi**

# How to Run
In this repository, we have three different simulations: `spread_out_sim`, `complex_sim1`, and `complex_sim2`.
1) `spread_out_sim` shows the simulation of robots spreading out in an open space.
2) `complex_sim1` and `complex_sim2` show the simulation of robots going through some complex environments with obstacles.
   
To run each simulation, you just need to run the corresponding python file. 

|Name |Simulation|
| -----------------| ------------------|
| `spread_out_sim` | ![spread_out](https://github.com/user-attachments/assets/ddd7d555-b7ff-4532-a9d6-981a93473c33)|
| `complex_sim1`   | ![complex1](https://github.com/user-attachments/assets/523627f3-56a5-43f5-bf99-92539215da26)|
| `complex_sim2`   |![complex2](https://github.com/user-attachments/assets/281d881e-0c69-4457-b790-d3c3cde4077a)|







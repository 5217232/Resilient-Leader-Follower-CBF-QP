# Resilient-Leader-Follower-CBF-QP-Controller

This repository contains the implementation of Strong $r$-robustness maintenance CBF-QP Controller. This guarantees the multi-robot network to maintain the sufficient number of connections among robots such a way that the robots can perform leader-follower consensus despite the presence of misbehaving agents. The algorithm also includes inter-agent and obstacle collision avoidances HOCBFs. For more information, see our paper.


# Dependencies
This respository requires the following libraries: **numpy**, **matplotlib**, **jax**, and **jaxlib**.

For solving the CBF-QP, we need the following libraries: **cvxpy** and **gurobi**

# How to Run
In this repository, we have two different simulations: `corridor_sim` and `spread_out_sim`. 
1) `corridor_sim` shows the simulation of $11$ robots going through a narrow space.
2) `spread_out_sim` shows the simulation of $14$ robots spreading out in an open space.
   
To run each simulation, you just need to run the corresponding python file. 

| `corridor_sim` | `spread_out_sim` |
| ------------- | ------------- |
| ![corridor_color_gif](https://github.com/user-attachments/assets/2a4defbf-c0c3-4e02-b54f-f816ef6ce434) |![spread_color-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/af534f0a-8f6c-447d-9262-f3dfd12ef501)|







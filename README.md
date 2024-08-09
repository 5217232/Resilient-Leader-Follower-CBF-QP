# Resilient-Leader-Follower-CBF-QP-Controller

This repository contains the implementation of Strong $r$-robustness maintenance CBF-QP Controller. This guarantees the multi-robot network to maintain the sufficient number of connections among robots such a way that the robots can perform leader-follower consensus despite the presence of misbehaving agents. The algorithm also includes inter-agent and obstacle collision avoidances HOCBFs. For more information, see our paper.


# Dependencies
This respository requires the following libraries: $\textbf{numpy}$, $\textbf{gray}{matplotlib}$, $\textbf{gray}{jax}$, and $\textbf{gray}{jaxlib}$.
For solving the CBF-QP, we need the following libraries: $\colorbox{gray}{cvxpy}$ and $\colorbox{gray}{gurobi}$

# How to Run
In this repository, we have two different simulations: $\colorbox{gray}{corridor_sim}$ and $\colorbox{gray}{spread_out_sim}$. 
1) $\colorbox{gray}{corridor_sim}$ shows the simulation of $11$ robots going through a narrow space.
2) $\colorbox{gray}{spread_out_sim}$ shows the simulation of $14$ robots spreading out in an open space.
To run each simulation, you just need to run the corresponding python file. 

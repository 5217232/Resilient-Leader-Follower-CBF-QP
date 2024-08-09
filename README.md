# Resilient-Leader-Follower-CBF-QP-Controller

This repository contains the implementation of Strong $r$-robustness maintenance CBF-QP Controller. This guarantees the multi-robot network to maintain the sufficient number of connections among robots such a way that the robots can perform leader-follower consensus despite the presence of misbehaving agents. The algorithm also includes inter-agent and obstacle collision avoidances HOCBFs. For more information, see our paper.


# Dependencies
This respository requires the following libraries: $\textbf{numpy}$, $\textbf{matplotlib}$, $\textbf{jax}$, and $\textbf{jaxlib}$.
For solving the CBF-QP, we need the following libraries: $\textbf{cvxpy}$ and $\textbf{gurobi}$

# How to Run
In this repository, we have two different simulations: $\textbf{corridor_sim}$ and $\textbf{spread_out_sim}$. 
1) $\textbf{corridor_sim}$ shows the simulation of $11$ robots going through a narrow space.
2) $\textbf{spread_out_sim}$ shows the simulation of $14$ robots spreading out in an open space.
To run each simulation, you just need to run the corresponding python file. 

# OCModel Network
Final Project for GS3754, Opinion Convergence Model Network.
 
The Opinion Convergence Model is the Opinion Dynamics method that applies the Friedkin-Johnsen method. This is a network model for observing the movement of model opinions according to the initial opinion distribution.

Here is the equation for Opinion Convergence Model Network.

$$
G = (V, E), \\
V = \{1, 2, \dots, N\}, \\ 
E = \{(i, j) \mid A_{ij} = 1, \, i \neq j \}, \\ 
A \in \{0, 1\}^{N \times N}, \\ 
T = \{1, 2, \dots, T_{end}\}, \\
\beta \in [0,1], \\ 
x_t \in \mathbb{R}^{1 \times N}, x_{t} \in [-1, 1] \quad \forall t \in T \cup \{0\}  \\
\lambda_{i} = \beta x_{0i}, \forall i \in N \\
W \in \mathbb{R}^{N \times N}, 
x_t = \lambda A x_{t-1} + (1-\lambda)x_0. \quad\forall t \in T
$$

# How to Run this code

### For Simulation one time.
```bash
python opinion.py
```

### For Simulation multiple times.
```bash
python sim.py
```
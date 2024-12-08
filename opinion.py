import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize



def opinion_with_dynamic_stubborn_coeff(G, initial_opinions, stubborn_coeffs, max_iter=500, tol=1e-3):
    opinions = np.array([initial_opinions[n] for n in G.nodes()])
    stubborn_coeffs_array = np.array([stubborn_coeffs[n] for n in G.nodes()])
    adjacency_matrix = nx.to_numpy_array(G, dtype=float)
    degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
    weights = np.linalg.inv(degree_matrix) @ adjacency_matrix

    opinion_history = [opinions.copy()]
    stubborn_coeffs_history = [stubborn_coeffs_array.copy()]

    for iteration in range(max_iter):
        new_opinions = opinions.copy()
        for i in range(len(opinions)):
            neighbor_opinion = weights[i] @ opinions 
            new_opinions[i] = (
                stubborn_coeffs_array[i] * initial_opinions[i] + 
                (1 - stubborn_coeffs_array[i]) * neighbor_opinion  
            )
        opinion_history.append(new_opinions.copy())
        if np.linalg.norm(new_opinions - opinions) < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break
        opinions = new_opinions

    return np.array(opinion_history), np.array(stubborn_coeffs_history)

num_nodes = 1000 
k = 4   
p = 0.2  

G = nx.watts_strogatz_graph(n=num_nodes, k=k, p=p)

m1, v1 = 0.5, 0.3
m2, v2 = -0.5, 0.1


initial_opinions = {}
for c,i in enumerate(G.nodes()):
    if np.random.rand() < 0.5:
        initial_opinions[i] = np.random.normal(m1, np.sqrt(v1))
    else:
        initial_opinions[i] = np.random.normal(m2, np.sqrt(v2))


    initial_opinions[i] = max(min(1,initial_opinions[i]),-1)

stubborn_coeffs = {i:min(abs(initial_opinions[i]) / 4 + np.random.rand() / 2 , 1) for i in G.nodes()}
opinion_history, stubborn_coeffs_history = opinion_with_dynamic_stubborn_coeff(
    G, initial_opinions, stubborn_coeffs
)

initial_opinion_values = np.array([initial_opinions[i] for i in G.nodes()])
final_opinion_values = opinion_history[-1]
stubborn_coeff_values = np.array([stubborn_coeffs[i] for i in G.nodes()])
opinion_differences = np.abs(initial_opinion_values - final_opinion_values)

pos = nx.spring_layout(G)
norm = Normalize(vmin=-1, vmax=1)
cmap = plt.cm.coolwarm

nx.draw_networkx_nodes(
    G, pos,
    node_color=[cmap(norm(value)) for value in initial_opinion_values],
    node_size=100,
)
nx.draw_networkx_edges(G, pos, alpha=0.2)
plt.title("Small-World Network with Initial Opinions (-1 to 1)", fontsize=16)
plt.axis('off')
plt.savefig('test1.png',dpi=300)
plt.show()

nx.draw_networkx_nodes(
    G, pos,
    node_color=[cmap(norm(value)) for value in final_opinion_values],
    node_size=100,
)
nx.draw_networkx_edges(G, pos, alpha=0.2)
plt.title("Small-World Network with Final Opinions (-1 to 1)", fontsize=16)
plt.axis('off')
plt.savefig('test2.png',dpi=300)
plt.show()

count_differences = []
for t in range(opinion_history.shape[0]):
    count_minus1_0 = np.sum(opinion_history[t] <= 0)
    count_0_1 = np.sum(opinion_history[t] > 0)
    count_differences.append(count_0_1 - count_minus1_0)

opn = list(initial_opinions.values())

n, bins, patches = plt.hist(opn, bins=20, alpha=0.7, edgecolor='black')

bin_centers = 0.5 * (bins[:-1] + bins[1:])
bin_colors = [cmap(norm(center)) for center in bin_centers]

for patch, color in zip(patches, bin_colors):
    patch.set_facecolor(color)

plt.title("Initial Opinion Histogram with Bin-based Color Gradient")
plt.xlabel("Opinion Value")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.savefig('test3.png',dpi=300)
plt.show()



plt.figure(figsize=(10, 6))
plt.plot(count_differences, label="Count Difference (0-1) - (-1-0)", color="purple")
plt.title("Count Difference Between Ranges by Step (-1 to 1)", fontsize=16)
plt.xlabel("Iteration Step")
plt.ylabel("Count Difference")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

average_opinions = opinion_history.mean(axis=1)
plt.figure(figsize=(10, 6))
plt.plot(average_opinions, label="Average Opinion", color="blue")
plt.title("Average Opinion Over Iteration Steps", fontsize=16)
plt.xlabel("Iteration Step")
plt.ylabel("Average Opinion")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

sorted_indices = np.argsort([final_opinion_values[i] for i in G.nodes()])
sorted_opinion_history = opinion_history[:, sorted_indices]
plt.figure(figsize=(10, 6))
plt.imshow(sorted_opinion_history.T, aspect='auto', cmap='coolwarm', norm=norm, origin='lower')
plt.title("Opinion Evolution Over Time (Sorted by Initial Opinions)", fontsize=16)
plt.xlabel("Iteration Step")
plt.ylabel("Node Index (Sorted by Initial Opinions)")
plt.colorbar(label="Opinion Value")
plt.savefig('test4.png',dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(stubborn_coeff_values, opinion_differences, alpha=0.6, edgecolor='black')
plt.title("Initial vs. Converged Opinion Difference by Stubborn Coefficient", fontsize=16)
plt.xlabel("Stubborn Coefficient")
plt.ylabel("Opinion Difference (Initial - Converged)")
plt.grid(alpha=0.3)
plt.show()

different_sign_indices = [
    i for i, (init, final) in enumerate(zip(initial_opinion_values, final_opinion_values))
    if np.sign(init) != np.sign(final)
]
initial_different_sign = initial_opinion_values[different_sign_indices]
final_different_sign = final_opinion_values[different_sign_indices]
sorted_indices = np.argsort(final_different_sign)
sorted_initial_different_sign = initial_different_sign[sorted_indices]
sorted_final_different_sign = final_different_sign[sorted_indices]
sorted_highlight_colors = ['red' if value > 0 else 'blue' for value in sorted_final_different_sign]
plt.figure(figsize=(10, 6))
plt.scatter(
    range(len(sorted_indices)), sorted_initial_different_sign,
    label="Initial Opinions", alpha=0.6, edgecolor="black", color="gray"
)
plt.scatter(
    range(len(sorted_indices)), sorted_final_different_sign,
    label="Final Opinions", alpha=0.8, edgecolor="black", c=sorted_highlight_colors
)
plt.title("Scatter Plot of Nodes with Different Initial and Final Opinion Signs (Sorted)", fontsize=16)
plt.xlabel("Node Index (Sorted by Final Opinions)")
plt.ylabel("Opinion Value")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.legend()
plt.grid(alpha=0.3)
plt.show()


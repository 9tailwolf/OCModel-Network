import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import json
from tqdm import tqdm

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
            break
        opinions = new_opinions

    return np.array(opinion_history), np.array(stubborn_coeffs_history)

def simulation(k,p,same=False):
    iteration = 100
    dataset = []
    for v1 in tqdm(range(10,51)):
        v1 = float(v1/100)
        for r in range(iteration):
            np.random.seed(r)
            num_nodes = 1000  

            G = nx.watts_strogatz_graph(n=num_nodes, k=k, p=p)

            m1 = 0.5
            m2, v2 = -0.5, 0.1
            initial_opinions = {}
            for c,i in enumerate(G.nodes()):
                if np.random.rand() < 0.5:
                    initial_opinions[i] = np.random.normal(m1, np.sqrt(v1))
                    if same:
                        initial_opinions[i] = min(1,max(0.001, initial_opinions[i]))
                else:
                    initial_opinions[i] = np.random.normal(m2, np.sqrt(v2))
                    if same:
                        initial_opinions[i] = min(-0.001,max(-1, initial_opinions[i]))
                initial_opinions[i] = min(0,max(-1, initial_opinions[i]))
                
            stubborn_coeffs = {i:min(abs(initial_opinions[i]) / 2, 1) for i in G.nodes()}
            opinion_history, stubborn_coeffs_history = opinion_with_dynamic_stubborn_coeff(
                G, initial_opinions, stubborn_coeffs
            )
            initial_opinion_values = np.array([initial_opinions[i] for i in G.nodes()])
            final_opinion_values = opinion_history[-1]
            stubborn_coeff_values = np.array([stubborn_coeffs[i] for i in G.nodes()])
            opinion_differences = np.abs(initial_opinion_values - final_opinion_values)
            count_differences = []
            for t in range(opinion_history.shape[0]):
                count_minus1_0 = np.sum(opinion_history[t] <= 0)
                count_0_1 = np.sum(opinion_history[t] > 0)
                count_differences.append(count_0_1 - count_minus1_0)
            average_opinions = opinion_history.mean(axis=1)


            res = { 'r':r,
                    'k':k,
                    'p':p,
                    'm1':m1,
                    'm2':m2,
                    'v1':v1,
                    'v2':v2,
                    'init_opinion': int(count_differences[0]),
                'final_opinion': int(count_differences[-1]),
                    'init_average_opinion': float(average_opinions[0]),
                    'final_average_opinion': float(average_opinions[-1])}
            
            dataset.append(res)
    return dataset

k = 4
p = 0.2
res = simulation(k,p)
with open('results/sim_' + str(k) + '_' + str(int(round(p * 10))) + '.json', 'w') as f : 
    json.dump(res, f, indent=4)
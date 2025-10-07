import numpy as np
import pandas as pd

def calculate_pagerank(TN_t_c):
    """Calculates PageRank for a given period's DataFrame slice."""
    edges = TN_t_c[['reporterISO', 'partnerISO', 'W_ij']]

    countries = sorted(list(set(edges['reporterISO']).union(edges['partnerISO'])))
    country_index = {country: i for i, country in enumerate(countries)}
    n = len(countries)

    if n == 0:
        return pd.DataFrame(columns=['Country', 'PageRank']) # Return empty if no nodes

    W = np.zeros((n, n))
    for index, row in edges.iterrows():
        i = country_index[row['reporterISO']]
        j = country_index[row['partnerISO']]
        W[i, j] += row['W_ij']

    row_sums = W.sum(axis=1)
    dangling_nodes_mask = (row_sums == 0)

    M_row_stochastic = np.divide(W, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis] != 0)
    if np.sum(dangling_nodes_mask) > 0:
        M_row_stochastic[dangling_nodes_mask, :] = 1.0 / n

    P = M_row_stochastic.T # Column-stochastic matrix for PageRank

    alpha = 0.85
    v = np.ones(n) / n
    r = np.ones(n) / n
    epsilon = 1e-8
    delta = 1.0
    iteration = 0
    max_iterations = 1000

    while delta > epsilon and iteration < max_iterations:
        r_new = alpha * P @ r + (1 - alpha) * v
        delta = np.linalg.norm(r_new - r, 1)
        r = r_new
        iteration += 1
    
    pagerank_t_c = pd.DataFrame({
        'Country': countries,
        'PageRank': r
    }).sort_values(by='PageRank', ascending=False).reset_index(drop=True)
    
    return pagerank_t_c

# This is still work in progress and need to be worked on to finalize a version of TradeRank

def calculate_pagerank_export(TN_t_c):
    """Calculates Export PageRank for a given period's DataFrame slice."""
    edges = TN_t_c[['reporterISO', 'partnerISO', 'W_ij']]

    countries = sorted(list(set(edges['reporterISO']).union(edges['partnerISO'])))
    country_index = {country: i for i, country in enumerate(countries)}
    n = len(countries)

    if n == 0:
        return pd.DataFrame(columns=['Country', 'PageRank']) # Return empty if no nodes

    W = np.zeros((n, n))
    for index, row in edges.iterrows():
        i = country_index[row['reporterISO']]
        j = country_index[row['partnerISO']]
        W[i, j] += row['W_ij']

    row_sums = W.sum(axis=1)
    dangling_nodes_mask = (row_sums == 0)

    # M_row_stochastic where M_ij = W_ij / sum(W_ik for k)
    # This represents the probability of exporting from i to j
    M_row_stochastic = np.divide(W, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis] != 0)
    
    # Handle dangling nodes (countries with no outgoing exports)
    if np.sum(dangling_nodes_mask) > 0:
        M_row_stochastic[dangling_nodes_mask, :] = 1.0 / n

    # For export PageRank, the transition matrix P should be M_row_stochastic itself
    # because the formula sums over j (receiving country) for each i (exporting country)
    # weighted by W_ij / sum(W_ik).
    # If r is a column vector, then M_row_stochastic @ r is NOT the correct operation for this formula.
    # The formula is r_i = alpha * SUM_j (M_ij * r_j)
    # This means (P_export @ r)_i = sum_j (P_export)_ij * r_j
    # So P_export should be M_row_stochastic
    # However, numpy's matmul (@) is row vector @ matrix or matrix @ column vector.
    # The standard PageRank formulation expects P to be column-stochastic, and r to be column vector.
    # If P is column stochastic (P_ji = prob of j -> i), then r_i = sum_j P_ji * r_j.
    # Our M_row_stochastic has M_ij = W_ij / sum_k W_ik. This is the probability of i -> j.
    # So to get sum_j M_ij r_j, we would need to multiply a row vector r by M_row_stochastic.
    # OR, we need a column-stochastic matrix which is M_row_stochastic.T.
    # Let's verify with the formula.
    # The formula looks like a left multiplication by r_j.
    # r_new_i = alpha * sum_j (M_ij * r_j) + (1-alpha)/n
    # If r is a column vector, this is (M_row_stochastic * r)_i
    # So, P should be M_row_stochastic.
    P = M_row_stochastic 

    alpha = 0.85
    v = np.ones(n) / n
    r = np.ones(n) / n
    epsilon = 1e-8
    delta = 1.0
    iteration = 0
    max_iterations = 1000

    while delta > epsilon and iteration < max_iterations:
        # r_new = alpha * P @ r + (1 - alpha) * v
        # Since P is M_row_stochastic (P_ij = W_ij / sum_k W_ik), and r is a column vector,
        # (P @ r)_i = sum_j P_ij * r_j. This exactly matches the desired formula's summation.
        r_new = alpha * P @ r + (1 - alpha) * v
        delta = np.linalg.norm(r_new - r, 1)
        r = r_new
        iteration += 1
    
    pagerank_t_c = pd.DataFrame({
        'Country': countries,
        'PageRank': r
    }).sort_values(by='PageRank', ascending=False).reset_index(drop=True)
    
    return pagerank_t_c
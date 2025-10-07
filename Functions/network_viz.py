import pandas as pd
import numpy as np  
import networkx as nx
import matplotlib.pyplot as plt
import math


def visualize_network(TN_t_c, pagerank_t_c, top_n=10, period_val=2000, cap=0, ax=None):
    """
    Visualizes the trade network for a given period and capability.
    If 'ax' is provided, plots on the given matplotlib axis (for subplots).
    If 'ax' is None, creates a new figure and axis and plots directly (for single network visualization).
    """
    G = nx.DiGraph()

    # Add edges with weights
    for index, row in TN_t_c.iterrows():
        reporter = row['reporterISO']
        partner = row['partnerISO']
        weight = row['W_ij']
        G.add_edge(reporter, partner, weight=weight)

    if not G.nodes():
        if ax:
            ax.set_title(f"No data for Cap {cap}, Period {period_val}", size=10)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            print(f"No network data found for period {period_val}. Cannot visualize.")
        return

    # Filter out nodes from PageRank_df that are not in the current graph
    active_nodes = list(G.nodes())
    pagerank_t_c_active = pagerank_t_c[pagerank_t_c['Country'].isin(active_nodes)].copy()

    # Identify top N and other countries
    if len(pagerank_t_c_active) > top_n:
        top_countries_df = pagerank_t_c_active.head(top_n)
        other_countries_df = pagerank_t_c_active.iloc[top_n:].copy()
        other_countries_df = other_countries_df.sort_values(by='PageRank', ascending=False)
    else:
        top_countries_df = pagerank_t_c_active
        other_countries_df = pd.DataFrame(columns=['Country', 'PageRank']) # Empty

    # --- Calculate Custom Positions ---
    pos = {}
    center_radius = 1
    outer_radius = 2.0
    num_top = len(top_countries_df)
    num_others = len(other_countries_df)

    # Position for top countries (small circle in the center)
    for i, country in enumerate(top_countries_df['Country']):
        angle = 2 * math.pi * i / num_top if num_top > 0 else 0
        x = center_radius * math.cos(angle)
        y = center_radius * math.sin(angle)
        pos[country] = np.array([x, y])

    # Position for other countries (larger circle)
    for i, country in enumerate(other_countries_df['Country']):
        angle = 2 * math.pi * i / num_others if num_others > 0 else 0
        x = outer_radius * math.cos(angle)
        y = outer_radius * math.sin(angle)
        pos[country] = np.array([x, y])

    # Node sizes
    node_pageranks = pagerank_t_c_active.set_index('Country')['PageRank']
    max_pr = node_pageranks.max() if not node_pageranks.empty else 1
    min_pr = node_pageranks.min() if not node_pageranks.empty else 0
    node_sizes = [
        1 + 2000 * ((node_pageranks[node] - min_pr) / (max_pr - min_pr + 1e-9))
        for node in G.nodes()
    ] if len(node_pageranks) > 1 else [2000 for _ in G.nodes()]

    # Node colors: Differentiate top countries
    node_colors = ['red' if node == "CHN" else 'skyblue' for node in G.nodes()]

    # Edge widths based on W_ij
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    min_weight = min(edge_weights) if edge_weights else 0
    edge_widths = [
        0.01 + 20 * ((w - min_weight) / (max_weight - min_weight + 1e-9))
        for w in edge_weights
    ] if len(edge_weights) > 1 else [2.0 for _ in G.edges()]

    # Drawing
    if ax is None:
        plt.figure(figsize=(15, 15))
        draw_args = {}
    else:
        draw_args = {'ax': ax}

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.6, **draw_args)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=G.edges(),
        edge_color='grey',
        arrows=True,
        arrowstyle='-|>',
        arrowsize=5,
        width=edge_widths,
        connectionstyle="arc3,rad=0.2",
        alpha=0.5,
        min_source_margin=0,
        min_target_margin=10,
        **draw_args
    )
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight="normal", alpha=0.8, **draw_args)

    if ax is None:
        plt.title(f'Trade Network of Capability {cap} for {period_val}', size=16)
        plt.axis('off')
        plt.show()
    else:
        ax.set_title(f'Cap {cap}, Period {period_val}', size=10)
        ax.axis('off')



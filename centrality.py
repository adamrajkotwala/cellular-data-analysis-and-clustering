import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from itertools import combinations

# Compute the normalized linkage between two genomic windows
def normalized_linkage(window1, window2, df, normalized_linkage_vals):

    if window1 == window2:
        normalized_linkage_vals.at[window1, window2] = 0 
        return 0

    NP_count = len(df) - 3
    AB = 0
    A = 0
    B = 0

    # using vectorized calculations

    # storing values from columns of specific windows, after row three for summation purposes
    val1 = df.iloc[3:, window1].values
    val2 = df.iloc[3:, window2].values  
    
    # summing the indentifications of these combinations and dividing them by the total number of nuclear profiles
    AB = ((val1 == 1) & (val2 == 1)).sum() / NP_count # cosegregation
    A = (val1 == 1).sum() / NP_count # detection frequency of window A
    B = (val2 == 1).sum() / NP_count # detection fequency of window B

    linkage_value = AB - (A*B)

    if linkage_value < 0:
        if A*B < ((1-A)*(1-B)):
            theoretical_maximum = A*B
        else:
            theoretical_maximum = ((1-A)*(1-B))
    else:
        if (B*(1-A)) < (A*(1-B)):
            theoretical_maximum = (B*(1-A))
        else:
            theoretical_maximum = (A*(1-B))

    if theoretical_maximum == 0:
        normalized_linkage_value = 0
    else:
        normalized_linkage_value = linkage_value / theoretical_maximum

    normalized_linkage_vals.at[window1, window2] = normalized_linkage_value
    
    return normalized_linkage_value

# build the graph by adding edges with normalized linkage above Q3
def build_graph(df):

    windows = list(df.columns)
    n = len(windows)
    norm_vals = pd.DataFrame(np.zeros((n, n)), index=windows, columns=windows)

    # fill the symmetric normalized linkage matrix
    for window1, window2 in combinations(windows, 2):
        value = normalized_linkage(window1, window2, df, norm_vals)
        norm_vals.at[window2, window1] = value

    # compute upper triangle and Q3
    triu = norm_vals.values[np.triu_indices(n, k=1)] # this function returns the indices for the upper triangular portion of an n x n matrix, excluding the diagonal (k=1 means start above the diagonal)
    Q3 = np.percentile(triu, 75)
    G = nx.Graph()
    G.add_nodes_from(windows)

    # add an edge if the normalized linkage is above Q3
    for window1, window2 in combinations(windows, 2):
        if norm_vals.at[window1, window2] > Q3:
            G.add_edge(window1, window2, weight=norm_vals.at[window1, window2])

    return G, Q3

# plot graph using plotly
def plot_graph(G):

    # setup layout with networkx
    pos = nx.spring_layout(G)

    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]

    # edge trace
    edge_x = []
    edge_y = []

    for u, v in G.edges():
        x0, y0 = G.nodes[u]['pos']
        x1, y1 = G.nodes[v]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, 
        line=dict(width=0.5, color='#888'), 
        hoverinfo='none', 
        mode='lines'
    )

    # node trace
    node_x = []
    node_y = []
    node_color = []
    node_text = []

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        conn = G.degree[node]
        node_color.append(conn)
        node_text.append(f'{node}: {conn} connections')

    node_trace = go.Scatter(
        x=node_x, y=node_y, 
        mode='markers', 
        hoverinfo='text', 
        text=node_text,
        marker=dict(
            showscale=True, 
            colorscale='YlGnBu', 
            reversescale=True, 
            color=node_color, 
            size=10, 
            line_width=2,
            colorbar=dict(title='Connections')
        )
    )

    # create graph
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Network Graph of Genomic Interactions in the Hist1 Region",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    fig.show()

# analyze degree centrality and print stats
def analyze_centrality(G):
    centrality = nx.degree_centrality(G)
    avg_centrality = sum(centrality.values()) / len(centrality)
    min_centrality = min(centrality.values())
    max_centrality = max(centrality.values())

    print("\nDegree Centrality Analysis:")
    print("- Average: {:.4f}".format(avg_centrality))
    print("- Min: {:.4f}".format(min_centrality))
    print("- Max: {:.4f}".format(max_centrality))

    ranked = sorted(centrality.items(), key=lambda x: x[1])
    print("\nRanked list of windows by degree centrality (ascending):")
    for window, value in ranked:
        print("{}: {:.4f}".format(window, value))


# load data, build graph, visualize and analyze
def main():
    df = pd.read_csv('filtered_data.csv')
    df = df.T
    G, Q3 = build_graph(df)
    print("Q3 threshold for edge inclusion:", Q3)
    plot_graph(G)
    analyze_centrality(G)

if __name__ == "__main__":
    main()

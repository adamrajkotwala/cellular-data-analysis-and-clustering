import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from itertools import combinations

def normalized_linkage(window1, window2, df, normalized_linkage_vals):
    if window1 == window2:
        normalized_linkage_vals.at[window1, window2] = 0 
        return 0

    NP_count = len(df) - 3
    val1 = df.iloc[3:, window1].values
    val2 = df.iloc[3:, window2].values  

    AB = ((val1 == 1) & (val2 == 1)).sum() / NP_count  # co-segregation
    A  = (val1 == 1).sum() / NP_count  # detection frequency for window1
    B  = (val2 == 1).sum() / NP_count  # detection frequency for window2

    linkage_value = AB - (A * B)

    if linkage_value < 0:
        theoretical_maximum = A * B if A * B < ((1 - A) * (1 - B)) else ((1 - A) * (1 - B))
    else:
        theoretical_maximum = B * (1 - A) if (B * (1 - A)) < (A * (1 - B)) else (A * (1 - B))

    normalized_linkage_value = linkage_value / theoretical_maximum if theoretical_maximum != 0 else 0

    normalized_linkage_vals.at[window1, window2] = normalized_linkage_value
    return normalized_linkage_value

def build_graph(df):
    windows = list(df.columns)
    normalized_linkage_values = pd.DataFrame(np.zeros((len(windows), len(windows))), index=windows, columns=windows)

    # fill matrix
    for window1, window2 in combinations(windows, 2):
        value = normalized_linkage(window1, window2, df, normalized_linkage_values)
        normalized_linkage_values.at[window2, window1] = value

    # use the upper triangle, excluding the diagonal (k=1)
    triu = normalized_linkage_values.values[np.triu_indices(len(windows), k=1)]
    Q3 = np.percentile(triu, 75)
    
    # build a graph, including an edge only if the normalized linkage is above the Q3 threshold
    G = nx.Graph()
    G.add_nodes_from(windows)
    for window1, window2 in combinations(windows, 2):
        if normalized_linkage_values.at[window1, window2] > Q3:
            G.add_edge(window1, window2, weight=normalized_linkage_values.at[window1, window2])
    return G, Q3, normalized_linkage_values

def plot_graph(G):
    pos = nx.spring_layout(G) # computes x and y posiitons for every mode
    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]

    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = G.nodes[u]['pos']
        x1, y1 = G.nodes[v]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x = []
    node_y = []
    node_color = []
    node_text = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        conn = G.degree[node]
        node_color.append(conn) # using the number of connections for the color
        node_text.append(f'{node}: {conn} connections')

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text, marker=dict(showscale=True, colorscale='YlGnBu', reversescale=True, color=node_color, size=10, line_width=2, colorbar=dict(title='Connections')))

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title="Network Graph of Genomic Interactions in the Hist1 Region", showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    fig.show()


def analyze_centrality(G):
    nodes = list(G.nodes())
    n = len(nodes)
    centrality = {}
    for node in nodes:
        deg = G.degree[node]
        centrality[node] = deg / (n - 1) if n > 1 else 0.0
    values = list(centrality.values())
    avg = sum(values) / n

    print("\nDegree Centrality Analysis:")
    print("Average: {:.4f}".format(avg))
    print("Min: {:.4f}".format(min(centrality.values())))
    print("Max: {:.4f}".format(max(centrality.values())))

    sorted_list = sorted(centrality.items(), key=lambda x: x[1]) # sort by centrality score (second item in the tuple)
    print("\nWindows ranked by centrality (ascending):")
    for window, value in sorted_list:
        print(f"{window}: {value:.4f}")

    return centrality

def plot_community_graph(G_sub, global_centrality, title="Community Graph"):
    pos = nx.spring_layout(G_sub)
    for node in G_sub.nodes():
        G_sub.nodes[node]['pos'] = pos[node]

    edge_x, edge_y = [], []
    for u, v in G_sub.edges():
        x0, y0 = G_sub.nodes[u]['pos']
        x1, y1 = G_sub.nodes[v]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none')

    node_x, node_y, node_sizes, node_text, node_color = [], [], [], [], []
    for node in G_sub.nodes():
        x, y = G_sub.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        # scale node size proportional to its global degree centrality.
        node_sizes.append(global_centrality[node] * len(global_centrality))
        node_text.append(f'{node}: {global_centrality[node]:.4f}')
        node_color.append(global_centrality[node])
        
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', text=node_text, hoverinfo='text', marker=dict(showscale=True,  colorscale='YlGnBu', reversescale=True, color=node_color, size=node_sizes,  line_width=2))
    
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title=title, showlegend=False,hovermode='closest',margin=dict(b=20, l=5, r=5, t=40),xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    fig.show()

def plot_community_heatmap(normalized_linkage_values, df, community_nodes, title="Community Heatmap"):
    all_windows = list(df.columns)
    # fill only cells where both windows are in the community
    heatmap_matrix = np.full((len(all_windows), len(all_windows)), np.nan)
    for i, win1 in enumerate(all_windows):
        for j, win2 in enumerate(all_windows):
            if win1 in community_nodes and win2 in community_nodes:
                heatmap_matrix[i, j] = normalized_linkage_values.loc[win1, win2]

    # compute global min and max from the full normalized linkage matrix for consistent color scales
    global_min = np.nanmin(normalized_linkage_values.values)
    global_max = np.nanmax(normalized_linkage_values.values)

    fig = go.Figure(data=go.Heatmap(z=heatmap_matrix, x=all_windows, y=all_windows, colorscale='Viridis', zmin=global_min, zmax=global_max, colorbar=dict(title="Linkage Value")))
    fig.update_layout(title=title, width=700, height=700)
    fig.show()

def analyze_communities(G, normalized_linkage_values, df, global_centrality, hist1_dict, lad_dict):
    # identify the top 5 hubs
    best_hubs = sorted(global_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nHub nodes in descending oorder:")
    for hub, cent in best_hubs:
        print(f"{hub}: {cent:.4f}")

    # for each hub, define the community and report stats
    for hub, cent in best_hubs:
        community_nodes = [hub] + list(G.neighbors(hub))
        community_size = len(community_nodes)

        # compute feature presence percentages

        # hist1
        hist1_count = 0
        for node in community_nodes:
            if hist1_dict.get(node, False): # returns true if present, false otherwise
                hist1_count += 1
        pct_hist1 = (hist1_count / community_size) * 100

        # lad
        lad_count = 0
        for node in community_nodes:
            if lad_dict.get(node, False):
                lad_count += 1
        pct_lad   = (lad_count / community_size) * 100

        print("\n")
        print(f"Hub {hub} Centrality: {cent:.4f}")
        print(f"Size of community: {community_size}")
        print(f"Percentage with hist1 gene: {pct_hist1:.2f}%")
        print(f"Percentage with LAD: {pct_lad:.2f}%")
        print(f"Nodes in community: {community_nodes}")

        # plot community graphs and heatmaps
        G_sub = G.subgraph(community_nodes).copy()
        plot_community_graph(G_sub, global_centrality, title=f"Community Graph for Hub {hub}")
        plot_community_heatmap(normalized_linkage_values, df, community_nodes, title=f"Community Heatmap for Hub {hub}")

def main():
    df = pd.read_csv("filtered_data.csv")
    df = df.T
    features_table = pd.read_csv("Hist1_region_features.csv", index_col=0)

    # checking if each window is in the feature table, and whether it has a LAD and/or a Hist1
    n_windows = len(df.columns) 
    hist1_dict = {}
    lad_dict   = {}
    for i in range(n_windows):
        if i < len(features_table):
            hist1_dict[i] = features_table.iloc[i]['Hist1'] > 0
            lad_dict[i]   = features_table.iloc[i]['LAD']   > 0
        else:
            hist1_dict[i] = False
            lad_dict[i]   = False

    # original netowrk graph
    graph, Q3, normalized_linkage_values = build_graph(df)
    print("Q3 threshold: ", Q3)
    plot_graph(graph)

    # get centrality and print out stats
    centrality_of_all_nodes = analyze_centrality(graph)

    # print community report, graph results
    analyze_communities(graph, normalized_linkage_values, df, centrality_of_all_nodes, hist1_dict, lad_dict)

if __name__ == "__main__":
    main()

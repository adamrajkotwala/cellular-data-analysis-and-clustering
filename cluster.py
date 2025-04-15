import pandas as pd
import numpy as np
import random
from itertools import product
import matplotlib.pyplot as plt

# my insights : 
# if j-distance = 0, two nps share a conserved subset of mutations, meaning they are probably linked
# if j-distance = 1, two nps share no active regions, meaning they might be independent
# i thought of this because i was noticing 1s that didn't make sense, but realized if np a only has 1s
# where np b does, then the denominator will = m11, marking them as 100% similar when in reality
# only the active region of the np is what's being considered in those cases
# this causes one cluster to grow bigger than the others typically, as the distance values are smaller, attracting more points

# purpose: perform clustering over multiple iterations and find the best clustering solution
# n (int): number of iterations to run
# NPs (list): list of np labels
# k (int): number of clusters
# j_distances (dict): dictionary storing precomputed jaccard distances
# return (dict, dict, int): best clusters, best medoids, and iteration count of the best solution
def iterate_program(n, NPs, k, j_distances):
    iteration_count = 0
    best_iteration_count = 0
    best_clusters = None
    best_cluster_points = None
    best_quality = float("inf")

    for i in range(n):
        print(f"\nrunning iteration {i+1}")

        iteration_count += 1

        min_data_index = 0
        max_data_index = len(NPs)

        # randomly select first medoids
        cluster_points = {}

        for i in range(k):
            random_np = NPs[random.randint(min_data_index, max_data_index - 1)]
            cluster_points[i] = random_np

        # recluster until medoids don't change
        clusters, cluster_points = stabilize(NPs, cluster_points, j_distances, k)

        # compute cluster quality
        intra_dist = compute_cluster_quality(clusters, j_distances)

        # compare qualities to find best clusters
        if intra_dist < best_quality:
            best_quality = intra_dist
            best_clusters = clusters
            best_cluster_points = cluster_points
            best_iteration_count = iteration_count
            

    return best_clusters, best_cluster_points, best_iteration_count

# purpose: compute the average intra-cluster distance for quality assessment
# clusters (dict): cluster assignments
# j_distances (dict): dictionary storing precomputed jaccard distances
# return (float): average intra-cluster distance
def compute_cluster_quality(clusters, j_distances):
    total_distance = 0
    count = 0

    # computing average distance for all clusters together
    for cluster_id, np_list in clusters.items():
        cluster_nps = list(np_list.keys())

        # iterate over every possible combination of NPs without repeats
        for i in range(len(cluster_nps)):
            for j in range(i+1, len(cluster_nps)):
                np1, np2 = cluster_nps[i], cluster_nps[j]
                j_dist = j_distances.get((np1, np2), j_distances.get((np2, np1)))

                total_distance += j_dist
                count += 1

    avg_distance = total_distance / count if count > 0 else 0

    return avg_distance

# purpose: calculate jaccard distance between two nps
# np1 (str): first np
# np2 (str): second np
# df (dataframe): dataset containing np mutation data
# j_distances (dict): dictionary storing precomputed jaccard distances
# return (float): jaccard distance between np1 and np2
def j_distance(np1, np2, df, j_distances):

    if (np1, np2) in j_distances:
        return j_distances[(np1, np2)]
    
    if np1 == np2:
        j_distances[(np1, np2)] = 0 
        return 0

    m11 = 0
    m10 = 0
    m01 = 0

    # using vectorized calculations

    # storing values from columns of specific NPs, after row one for summation purposes
    val1 = df.loc[1:, np1].values
    val2 = df.loc[1:, np2].values  
    
    # summing the indentifications of these combinations
    m11 = ((val1 == 1) & (val2 == 1)).sum()
    m10 = ((val1 == 1) & (val2 == 0)).sum()
    m01 = ((val1 == 0) & (val2 == 1)).sum()


    if m10 < m01:
        denominator = m10 + m11
    else:
        denominator = m01 + m11

    if denominator != 0:
        j_similarity = m11 / denominator
    else: 
        j_similarity = 0
    
    j_dist = 1 - j_similarity

    j_distances[(np1, np2)] = j_dist

    return j_dist

# purpose: assign nps to the closest medoid
# NPs (list): list of np labels
# cluster_points (dict): current medoid assignments
# j_distances (dict): dictionary storing jaccard distances
# k (int): number of clusters
# return (dict): clusters with assigned nps and their distances
def assign_clusters(NPs, cluster_points, j_distances, k):

    clusters = {}
    for i in range(k):
        clusters[i] = {}

    # j distances are already calculated, so iterate through nps and find the closest medoid
    for np in NPs:
        if np in cluster_points.values():  # medoids get automatically assigned
            continue

        closest_cluster = None
        min_distance = float("inf")

        # compare np against all medoids
        for cluster_id in range(k):
            medoid = cluster_points[cluster_id]
            j_dist = j_distances.get((medoid, np), j_distances.get((np, medoid))) # checking both combos for safety

            if j_dist < min_distance:
                min_distance = j_dist
                closest_cluster = cluster_id

        clusters[closest_cluster][np] = min_distance

    for cluster_id in cluster_points:
        medoid = cluster_points[cluster_id]
        if medoid not in clusters[cluster_id]:
            clusters[cluster_id][medoid] = 0  # distance to itself is 0

    return clusters

# purpose: find new medoids
# clusters (dict): current np clusters
# j_distances (dict): dictionary storing jaccard distances
# return (dict): updated medoid assignments
def assign_medoids(clusters, j_distances):

    new_medoids = {}

    for cluster_id in clusters:
        cluster_nps = list(clusters[cluster_id].keys())

        # handling cluster that is only one np
        if len(cluster_nps) < 2:
            new_medoids[cluster_id] = cluster_nps[0]
            continue

        avg_distances = {}

        # compute the average J distance for each np in the cluster
        for np1 in cluster_nps:
            total_distance = 0

            for np2 in cluster_nps:
                distance = j_distances.get((np1, np2), j_distances.get((np2, np1))) # checking both combos for safety
                total_distance += distance
            
            average = total_distance / len(cluster_nps)
            avg_distances[np1] = average

        # select the np with the minimum average distance as the new medoid
        best_new_medoid = min(avg_distances, key=avg_distances.get)
        new_medoids[cluster_id] = best_new_medoid


    return new_medoids

# purpose: perform clustering iterations until medoids stabilize
# NPs (list): list of np labels
# cluster_points (dict): current medoid assignments
# j_distances (dict): dictionary storing jaccard distances
# k (int): number of clusters
# return (dict): final cluster assignments and medoids
def stabilize(NPs, cluster_points, j_distances, k):
    previous_medoids = {}
    medoid_history = []
    max_iterations = 100
    iteration = 0

    while previous_medoids != cluster_points:
        
        medoid_tuple = tuple(sorted(cluster_points.values()))  # convert to a hashable format

        if medoid_tuple in medoid_history:
            print("stabilize: detected cycling medoids")
            break

        medoid_history.append(medoid_tuple)

        if iteration >= max_iterations:
            print("stabilize: max iterations reached")
            break

        previous_medoids = cluster_points.copy()
        
        clusters = assign_clusters(NPs, cluster_points, j_distances, k)
        
        cluster_points = assign_medoids(clusters, j_distances)
        
        iteration += 1

    return clusters, cluster_points

# purpose: calculate the variability of the average cluster distances
# avg_distances (list): list of average distances for each cluster
# return (float): normlizaed measure of variability
# this calculates how spread out the cluster distances are by comparing 
# the biggest and smallest values to the average. dividing by the average 
# helps keep the result meaningful no matter the scale of the distances.
def calculate_variability(clusters, cluster_points, j_distances):

    avg_distances = []

    for cluster_id in cluster_points:
        medoid = cluster_points[cluster_id]
        total_distance = 0

        for NP in clusters[cluster_id]:
            total_distance += j_distances.get((medoid, NP), j_distances.get((NP, medoid)))

        avg_distance = total_distance / len(clusters[cluster_id])
        avg_distances.append(avg_distance)
        print("\ncluster ", cluster_id+1, " medoid(", medoid, "): avg j distance = ", avg_distance)

    mean_dist = np.mean(avg_distances)
    std_dev = np.std(avg_distances, ddof=0)

    if mean_dist == 0:
        return 0 

    return std_dev / mean_dist

# purpose: compute feature presence percentages for each NP in each cluster
# feature_table (DataFrame): DataFrame containing feature presence data
# df (DataFrame): dataset containing np mutation data
# clusters (dict): dictionary of cluster assignments
# cluster_points (dict): dictionary of medoid assignments
# return (dict): dictionary containing feature percentages per cluster per NP
def calculate_cluster_feature_frequencies(feature_table, df, clusters, cluster_points):
    # identify indices of relevant windows for each feature
    LAD_windows = feature_table.index[feature_table["LAD"] > 0].tolist()
    hist1_windows = feature_table.index[feature_table["Hist1"] > 0].tolist()
    vmn_windows = feature_table.index[feature_table["Vmn"] > 0].tolist()
    RNAPII_S2P_windows = feature_table.index[feature_table["RNAPII-S2P"] > 0].tolist()
    RNAPII_S5P_windows = feature_table.index[feature_table["RNAPII-S5P"] > 0].tolist()
    RNAPII_S7P_windows = feature_table.index[feature_table["RNAPII-S7P"] > 0].tolist()
    enhancer_windows = feature_table.index[feature_table["Enhancer"] > 0].tolist()
    H3K9me3_windows = feature_table.index[feature_table["H3K9me3"] > 0].tolist()
    H3K20me3_windows = feature_table.index[feature_table["H3K20me3"] > 0].tolist()
    H3K27me3_windows = feature_table.index[feature_table["h3k27me3"] > 0].tolist()
    H3K36me3_windows = feature_table.index[feature_table["H3K36me3"] > 0].tolist()
    NANOG_windows = feature_table.index[feature_table["NANOG"] > 0].tolist()
    pou5f1_windows = feature_table.index[feature_table["pou5f1"] > 0].tolist()
    sox2_windows = feature_table.index[feature_table["sox2"] > 0].tolist()
    CTCF_windows = feature_table.index[feature_table["CTCF-7BWU"] > 0].tolist()

    cluster_feature_percentages = {}

    for cluster_id, np_list in clusters.items():
        np_names = list(np_list.keys()) 
        medoid = cluster_points[cluster_id]

        # medoid isnt included in the cluster container but it is part of the cluster still
        if medoid not in np_names:
            np_names.append(medoid)

        # to hold percentages for each NP
        cluster_feature_percentages[cluster_id] = {}

        for np in np_names:
            total_ones = df[np].sum()

            # sum 1s in relevant windows for the respective NP
            LAD_ones = df.loc[LAD_windows, np].sum()
            Hist1_ones = df.loc[hist1_windows, np].sum()
            Vmn_ones = df.loc[vmn_windows, np].sum()
            RNAPII_S2P_ones = df.loc[RNAPII_S2P_windows, np].sum()
            RNAPII_S5P_ones = df.loc[RNAPII_S5P_windows, np].sum()
            RNAPII_S7P_ones = df.loc[RNAPII_S7P_windows, np].sum()
            enhancer_ones = df.loc[enhancer_windows, np].sum()
            H3K9me3_ones = df.loc[H3K9me3_windows, np].sum()
            H3K20me3_ones = df.loc[H3K20me3_windows, np].sum()
            H3K27me3_ones = df.loc[H3K27me3_windows, np].sum()
            H3K36me3_ones = df.loc[H3K36me3_windows, np].sum()
            NANOG_ones = df.loc[NANOG_windows, np].sum()
            pou5f1_ones = df.loc[pou5f1_windows, np].sum()
            sox2_ones = df.loc[sox2_windows, np].sum()
            CTCF_ones = df.loc[CTCF_windows, np].sum()

            # calculate percentages
            LAD_percent = (LAD_ones / total_ones) * 100 if total_ones > 0 else 0
            Hist1_percent = (Hist1_ones / total_ones) * 100 if total_ones > 0 else 0
            Vmn_percent = (Vmn_ones / total_ones) * 100 if total_ones > 0 else 0
            RNAPII_S2P_percent = (RNAPII_S2P_ones / total_ones) * 100 if total_ones > 0 else 0
            RNAPII_S5P_percent = (RNAPII_S5P_ones / total_ones) * 100 if total_ones > 0 else 0
            RNAPII_S7P_percent = (RNAPII_S7P_ones / total_ones) * 100 if total_ones > 0 else 0
            enhancer_percent = (enhancer_ones / total_ones) * 100 if total_ones > 0 else 0
            H3K9me3_percent = (H3K9me3_ones / total_ones) * 100 if total_ones > 0 else 0
            H3K20me3_percent = (H3K20me3_ones / total_ones) * 100 if total_ones > 0 else 0
            H3K27me3_percent = (H3K27me3_ones / total_ones) * 100 if total_ones > 0 else 0
            H3K36me3_percent = (H3K36me3_ones / total_ones) * 100 if total_ones > 0 else 0
            NANOG_percent = (NANOG_ones / total_ones) * 100 if total_ones > 0 else 0
            pou5f1_percent = (pou5f1_ones / total_ones) * 100 if total_ones > 0 else 0
            sox2_percent = (sox2_ones / total_ones) * 100 if total_ones > 0 else 0
            CTCF_percent = (CTCF_ones / total_ones) * 100 if total_ones > 0 else 0

            cluster_feature_percentages[cluster_id][np] = {
                'LAD %': LAD_percent,
                'Hist1 %': Hist1_percent,
                'Vmn %': Vmn_percent,
                'RNAPII-S2P %': RNAPII_S2P_percent,
                'RNAPII-S5P %': RNAPII_S5P_percent,
                'RNAPII-S7P %': RNAPII_S7P_percent,
                'Enhancer %': enhancer_percent,
                'H3K9me3 %': H3K9me3_percent,
                'H3K20me3 %': H3K20me3_percent,
                'H3K27me3 %': H3K27me3_percent,
                'H3K36me3 %': H3K36me3_percent,
                'NANOG %': NANOG_percent,
                'Pou5f1 %': pou5f1_percent,
                'Sox2 %': sox2_percent,
                'CTCF %': CTCF_percent
            }

    return cluster_feature_percentages

# purpose: compute the average feature percentages for each cluster
# cluster_feature_percentages (dict): dictionary containing feature percentages per NP per cluster
# return (dict): dictionary containing the average feature percentages per cluster
def calculate_cluster_averages(cluster_feature_percentages):
    cluster_averages = {}

    for cluster_id, np_data in cluster_feature_percentages.items():
        feature_totals = {}
        np_count = len(np_data)

        # total the feature counts for the nps in the respective cluster
        for np, features in np_data.items():
            for feature, value in features.items():
                if feature not in feature_totals:    # appends the feature for the first time per cluster
                    feature_totals[feature] = 0
                feature_totals[feature] += value

        # calculate the average for each feature
        cluster_averages[cluster_id] = {}
        for feature, total in feature_totals.items():
            cluster_averages[cluster_id][feature] = total / np_count

    return cluster_averages

# purpose: generate and display radar charts for feature averages per cluster
# cluster_averages (dict): dictionary containing the average feature percentages per cluster
def plot_radar_chart(cluster_averages):
    for cluster_id, averages in cluster_averages.items():
        features = list(averages.keys())
        values = list(averages.values())

        # close the radar chart loop by adding the first value back at the end
        values.append(values[0])

        # linspace automatically creates evenly spaced angles
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist() + [0] # start point, end point, num of labels, disabling 2pi label, converting array to list and appending first element again

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True)) # 6 by 6 inches and circular instead of cartesian
        ax.plot(angles, values, marker='o')
        ax.fill(angles, values, alpha=0.3) # transparency

        ax.set_xticks(angles[:-1]) # excludes last duplicate angle
        ax.set_xticklabels(features, fontsize=10)
        ax.set_title(f'Cluster {cluster_id+1} Feature Averages', fontsize=12, fontweight='bold', y=1.1)

        plt.show()

# purpose: print final cluster results, including medoids and average distances
# clusters (dict): dictionary with cluster assignments
# cluster_points (dict): dictionary with medoid assignments
# j_distances (dict): precomputed jaccard distances
def output(clusters, cluster_points, variability, best_iteration):
    print("\nbest iteration: ", best_iteration)
    print("\nfinal clusters")

    for cluster_id in clusters:
        print("\ncluster ", cluster_id+1, " medoid: ", cluster_points[cluster_id], '\n')
        for np in clusters[cluster_id]:
            print(' ', np, ":", clusters[cluster_id][np])

    print("\nvariability:", variability)


def main():
    df = pd.read_csv('filtered_data.csv')
    NPs = df.columns[3:]
    k = 3
    num_iterations = 1000
    j_distances = {}

    # precompute all J distances for all np pairs
    for np1, np2 in product(NPs, repeat=2):
        j_distance(np1, np2, df, j_distances)

    best_clusters, best_cluster_points, best_iteration_count = iterate_program(num_iterations, NPs, k, j_distances)

    variability = calculate_variability(best_clusters, best_cluster_points, j_distances)

    output(best_clusters, best_cluster_points, variability, best_iteration_count)

    feature_table = pd.read_csv('Hist1_region_features.csv')

    cluster_feature_percentages = calculate_cluster_feature_frequencies(feature_table, df, best_clusters, best_cluster_points)

    cluster_averages = calculate_cluster_averages(cluster_feature_percentages)

    plot_radar_chart(cluster_averages)

if __name__ == "__main__":
    main()

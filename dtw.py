from cluster import load_data, fragment_signals
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import pyabf
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.io import loadmat
import scipy
from Pelt.detection import get_events
import math
import traceback
import json
import warnings
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from bayes_opt import BayesianOptimization, UtilityFunction
from functools import partial
from sklearn.decomposition import PCA
import shutil
import time
import warnings
import traceback

def read_last_line(output_file):
    last_i, last_j = 0, 0  
    with open(output_file, 'r') as f:
        lines = f.readlines() 
        if len(lines) >= 2: 
            last_line = lines[-1].strip().split(',') 
            last_i, last_j = int(last_line[0]), int(last_line[1])
    return last_i, last_j

def compute_dtw_distances(segments):
    output_file = 'dtw_matrix1.txt'
    num_segments = len(segments)
    dtw_matrix = np.zeros((num_segments, num_segments))
    lines_written = 0  # Track the number of lines written
    last_i, last_j = read_last_line('dtw_matrix1.txt')

    # Open the file in append mode to continue writing
    with open(output_file, 'a') as f:
        for i in range(last_i, num_segments):
            print("Index is:", i)
            for j in range((last_j + 1) if i == last_i else i + 1, num_segments):
                segment_i = np.ravel(segments[i])
                segment_j = np.ravel(segments[j])
                distance, _ = fastdtw(segment_i, segment_j, dist=lambda x, y: euclidean([x], [y]))
                
                # Store the calculated distance in the matrix
                dtw_matrix[i, j] = distance
                dtw_matrix[j, i] = distance

                # Write the current distance to the file
                f.write(f"{i},{j},{distance:.4f}\n")

                # Increment the line counter
                lines_written += 1

                # Flush the buffer every 100 lines
                if lines_written >= 1000:
                    f.flush()  # Force buffer to write to disk
                    lines_written = 0  # Reset the counter

    return dtw_matrix

def compute_within_cluster_distance(segments, labels, n_clusters, dtw_matrix):
    # Ensure dtw_matrix is a NumPy array
    dtw_matrix = np.array(dtw_matrix)
    
    within_cluster_distance = 0
    for cluster in range(n_clusters):
        cluster_indices = [i for i in range(len(labels)) if labels[i] == cluster]
        
        # Skip clusters with only one element
        if len(cluster_indices) <= 1:
            continue

        try:
            # Compute pairwise distances within the cluster
            cluster_distances = [dtw_matrix[i, j] for i in cluster_indices for j in cluster_indices if i != j]
        except Exception as e:
            print(f"Error computing pairwise distances for cluster {cluster} with indices {cluster_indices}: {e}")
            return np.inf  # or another appropriate fallback value to signal failure

        # Sum up the distances and compute the average
        within_cluster_distance += sum(cluster_distances) / (len(cluster_indices) * (len(cluster_indices) - 1))

    return within_cluster_distance


def compute_between_cluster_distance(segments, labels, n_clusters):
    cluster_centroids = []
    for cluster in range(n_clusters):
        cluster_segments = [segments[i] for i in range(len(labels)) if labels[i] == cluster]
        if cluster_segments:
            centroid = np.mean(cluster_segments, axis=0)
            # Flatten the centroid to 1-D if necessary
            if centroid.ndim > 1:
                centroid = centroid.flatten()
            cluster_centroids.append(centroid)
    
    between_cluster_distance = 0
    if len(cluster_centroids) > 1:
        for i in range(len(cluster_centroids)):
            for j in range(i + 1, len(cluster_centroids)):
                try:
                    # Convert centroids to ensure they are float64 arrays
                    centroid_i = np.asarray(cluster_centroids[i], dtype=np.float64)
                    centroid_j = np.asarray(cluster_centroids[j], dtype=np.float64)
                    centroid_i= centroid_i.reshape(1, -1)
                    centroid_j= centroid_j.reshape(1, -1)
                    # Debug: Print shapes and types
                    print(f"Centroid {i} (converted) shape: {centroid_i.shape}, dtype: {centroid_i.dtype}")
                    print(f"Centroid {j} (converted) shape: {centroid_j.shape}, dtype: {centroid_j.dtype}")

                    # Check for NaNs or Infs
                    if np.any(np.isnan(centroid_i)) or np.any(np.isnan(centroid_j)):
                        print(f"NaN values detected in centroids for clusters {i} or {j}.")
                        continue
                    if np.any(np.isinf(centroid_i)) or np.any(np.isinf(centroid_j)):
                        print(f"Infinite values detected in centroids for clusters {i} or {j}.")
                        continue

                    # Compute distance using fastdtw
                    distance, _ = fastdtw(centroid_i, centroid_j, dist=euclidean)
                    between_cluster_distance += distance
                except Exception as e:
                    print(f"Error computing DTW distance between cluster {i} and cluster {j}: {e}")
                    return np.inf  # Return a large distance to signify an error

        # Compute the average between-cluster distance
        between_cluster_distance /= len(cluster_centroids) * (len(cluster_centroids) - 1) / 2

    return between_cluster_distance

def objective_function(n_clusters, dtw_matrix, my_events_fragments):
    start_time = time.time()
    n_clusters = int(n_clusters)
    try:
        clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
        labels = clustering_model.fit_predict(dtw_matrix)
        within_cluster_dist = compute_within_cluster_distance(my_events_fragments, labels, n_clusters, dtw_matrix)
        between_cluster_dist = compute_between_cluster_distance(my_events_fragments, labels, n_clusters)
        result = -(between_cluster_dist - within_cluster_dist)
        print(f"For {n_clusters} clusters: Within/Between Cluster Distance = {-result}, Time Taken = {time.time() - start_time} seconds")
        return result
    except Exception as e:
        print(f"Error in objective_function for {n_clusters} clusters: {e}")
        return -np.inf  # Returning a very negative value in case of failure

def read_dtw_matrix(segments):
    output_file = 'dtw_matrix1.txt'
    num_segments = len(segments)
    dtw_matrix = np.zeros((num_segments, num_segments))
    computed_pairs = set()
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line_number, line in enumerate(f):
                i, j, dist = line.strip().split(',')
                i, j = int(i), int(j)
                computed_pairs.add((i, j))
                dtw_matrix[i, j] = float(dist)
                dtw_matrix[j, i] = float(dist)
    return dtw_matrix

def optimize_n_clusters(dtw_matrix, my_events_fragments):
    objective_partial = partial(objective_function, dtw_matrix=dtw_matrix, my_events_fragments=my_events_fragments)
    
    optimizer = BayesianOptimization(
        f=objective_partial,
        pbounds={'n_clusters': (10, 100)},
        random_state=42,
        allow_duplicate_points=False  # Ensure duplicates are not allowed
    )
    
    # Set Gaussian process parameters if needed
    optimizer.set_gp_params(alpha=1e-3)

    # Create a utility function instance for the acquisition function
    utility = UtilityFunction(kind='ucb', kappa=2.5, xi=0.01)  # Lower xi to encourage exploration

    n_iterations = 20
    tested_clusters = set()  # To keep track of tested n_clusters
    losses = {}  # Store losses for each tested n_clusters
    repeated_attempts = 0  # Count for repeated attempts

    with tqdm(total=n_iterations, desc="Optimizing n_clusters") as pbar:
        def callback(_):
            pbar.update(1)

        try:
            # Perform the optimization
            for _ in range(n_iterations):
                next_point = optimizer.suggest(utility)
                n_clusters = int(np.round(next_point['n_clusters']))  # Ensure n_clusters is an integer
                
                print("Proposed n_clusters:", n_clusters)

                # If n_clusters has been tested, continue to sample new unique values
                while n_clusters in tested_clusters:
                    print(f"Already tested n_clusters: {n_clusters}, suggesting new...")
                    next_point = optimizer.suggest(utility)
                    n_clusters = int(np.round(next_point['n_clusters']))  # Recalculate n_clusters
                    repeated_attempts += 1

                    # If repeated attempts exceed a threshold, stop Bayesian Optimization
                    if repeated_attempts > 5:  # Adjust this threshold as necessary
                        print("Too many repeated attempts. Switching to random sampling.")
                        break
                
                # If we've decided to switch to random sampling
                if repeated_attempts > 5:
                    break

                # Register the tested n_clusters
                tested_clusters.add(n_clusters)

                # Evaluate the objective function
                target = objective_partial(n_clusters=n_clusters)  # Pass n_clusters to the objective function
                optimizer.register(params={'n_clusters': n_clusters}, target=target)
                losses[n_clusters] = target  # Store the loss for the tested n_clusters

            print("Tested clusters:", tested_clusters)

        except Exception as e:
            warnings.warn(f"Optimization process encountered an error: {e}")
            print(traceback.format_exc())
            return None  # Return None if the optimization fails

    # Random sampling for the remaining iterations
    remaining_iterations = n_iterations - len(tested_clusters)
    for _ in range(remaining_iterations):
        n_clusters = np.random.randint(10, 101)  # Randomly sample between 10 and 100
        while n_clusters in tested_clusters:  # Ensure uniqueness
            n_clusters = np.random.randint(10, 101)
        
        # Evaluate the objective function
        target = objective_partial(n_clusters=n_clusters)
        optimizer.register(params={'n_clusters': n_clusters}, target=target)
        losses[n_clusters] = target  # Store the loss for the sampled n_clusters

    # Find the n_clusters with the minimum loss
    best_n_clusters = min(losses, key=losses.get)  # Get the n_clusters with the lowest loss
    print(f"Best n_clusters found by Bayesian Optimization (after random sampling): {best_n_clusters} with loss: {losses[best_n_clusters]}")
    
    return best_n_clusters
def plot_dendrogram(dtw_matrix, method='ward'):
    # Perform hierarchical clustering
    Z = linkage(dtw_matrix, method=method)
    
    plt.figure(figsize=(12, 8))
    dendrogram(Z)
    plt.title('Dendrogram for Agglomerative Clustering')
    plt.xlabel('Segment Index')
    plt.ylabel('Distance')
    plt.savefig("plots/clusters/dendogram.png")

def plot_representative_patterns(segments, labels, n_clusters):
    # Ensure the directory exists
    os.makedirs("plots/clusters", exist_ok=True)

    for cluster in range(n_clusters):
        plt.figure(figsize=(10, 6))
        cluster_segments = [segments[i] for i in range(len(labels)) if labels[i] == cluster]
        
        if cluster_segments:
            mean_pattern = np.mean(cluster_segments, axis=0)
            plt.plot(mean_pattern, label=f'Cluster {cluster + 1}')
            plt.title(f'Representative Pattern of Cluster {cluster + 1}')
            plt.xlabel('Time')
            plt.ylabel('Signal Value')
            plt.legend()
            # Save the plot for the specific cluster
            plt.savefig(f"plots/clusters/cluster_{cluster + 1}_pattern_for_{n_clusters}.png")
            plt.close()  # Close the plot to free memory
        else:
            print(f"No segments found for Cluster {cluster + 1}.")

def find_patterns_in_events():
    my_events, _, _ = load_data()
    my_events_fragments = fragment_signals(my_events, 30)
    dtw_matrix = read_dtw_matrix(my_events_fragments[0:10000])

    # Get the optimal number of clusters
    best_n_clusters = optimize_n_clusters(dtw_matrix, my_events_fragments)
    print(f"Using {best_n_clusters} clusters for Agglomerative Clustering.")  # Debugging statement

    # Ensure n_clusters is an integer
    best_n_clusters = int(best_n_clusters)

    # Initialize the clustering model
    clustering_model = AgglomerativeClustering(n_clusters=best_n_clusters, metric='precomputed', linkage='average')

    with tqdm(total=len(dtw_matrix), desc="Clustering") as pbar:
        labels = clustering_model.fit_predict(dtw_matrix)
        pbar.update(len(dtw_matrix))

    plot_representative_patterns(my_events_fragments, labels, best_n_clusters)
    plot_dendrogram(dtw_matrix)

def copy_file(source_file, destination_file):
    source_file_i, source_file_j = read_last_line(source_file)
    dest_file_i, dest_file_j = read_last_line(destination_file)
    
    if source_file_i < dest_file_i or (source_file_i == dest_file_i and source_file_j < dest_file_j):
        print(f"No new data to copy. Destination file is up-to-date.")
        return

    try:
        with open(source_file, 'r') as src, open(destination_file, 'a') as dest:
            src.seek(0, os.SEEK_SET)
            lines = src.readlines()
            
            start_line = dest_file_i * (len(lines) // source_file_i) + dest_file_j
            lines_to_copy = lines[start_line:]

            # Write the lines to the destination file
            dest.writelines(lines_to_copy)
        
        print(f"Successfully copied new content from '{source_file}' to '{destination_file}'.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

find_patterns_in_events()
#copy_file("dtw_matrix1.txt", "copy_dtw_matrix.txt")

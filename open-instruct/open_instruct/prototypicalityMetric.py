import numpy as np
import torch
import argparse
import faiss, json
import pandas as pd
parser = argparse.ArgumentParser(
    description="Process perplexities csv files into LP(1) subsets."
)
parser._action_groups.pop()
required = parser.add_argument_group("Required Arguments")
optional = parser.add_argument_group("Optional Arguments")


required.add_argument(
    "--embedding_file",
    help="embedding file.",
    required=True,
)


required.add_argument(
    "--main_file",
    help="file.",
    required=True,
)

required.add_argument(
    "--clusters",
    help="clusters.",
    required=True,
)

required.add_argument(
    "--dataset",
    help="dataset.",
    required=True,
)


args = parser.parse_args()
full_embedding_file=args.embedding_file
k = args.clusters
dataset = args.dataset

embed_dict_active_full = torch.load(full_embedding_file, map_location=torch.device("cpu"))
ids_active_full = np.array(list(embed_dict_active_full.keys()))
'introduced squueze to ensure that the torch.Size([1119, 1, 4096]) becomes torch.Size([1119, 4096])'
vectors = torch.stack([embed_dict_active_full[k].squeeze(0) for k in ids_active_full]).to(torch.float32)
# GPU kmeans

kmeans = faiss.Kmeans(d=vectors.shape[1], k=k, niter=20, verbose=True, gpu=True)
kmeans.train(vectors)
print('kmeans total square error  ',kmeans.obj)
# Assign points to closest centroid
D, I = kmeans.index.search(vectors, 1)
centroids = kmeans.centroids
percentage = [1,5,10,25,50,75]
total_data = vectors.shape[0]
print('total_data ',total_data)



def get_farthest_points_per_cluster(data, cluster_id, assignments, centroid):
    """
    Returns the farthest point (index and distance) from the centroid in the given cluster.
    
    Parameters:
    - data: (N, d) array of points
    - kmeans: trained faiss.Kmeans object
    
    Returns:
    - (point_idx, squared_distance)}
    """
    farthest_points = {}

    # Get all points assigned to this cluster
    indices = np.where(assignments == cluster_id)[0]
    cluster_points = data[indices]

    # Compute squared distances from centroid
    dists = np.sum((cluster_points - centroid)**2, axis=1)
    max_idx = np.argmax(dists)
    farthest_points = (indices[max_idx], dists[max_idx])

    return farthest_points

def precompute_sorted_cluster_points(data, assignments, centroids):
    """
    For each cluster, return a sorted list of (distance, point_index) from farthest to closest.
    """
    from collections import defaultdict
    cluster_dist_lists = defaultdict(list)

    for cluster_id in range(centroids.shape[0]):
        indices = (assignments == cluster_id).nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            continue

        cluster_points = data[indices]
        centroid = centroids[cluster_id]

        dists = torch.sum((cluster_points - centroid) ** 2, dim=1)
        sorted_indices = torch.argsort(dists, descending=True)

        # Store (distance, original_index_in_data)
        cluster_dist_lists[cluster_id] = [
            (dists[i].item(), indices[i].item()) for i in sorted_indices
        ]

    return cluster_dist_lists
def round_robin_farthest_selection(cluster_dist_lists, budget):
    from collections import defaultdict

    selected_ids = set()
    selected_count = 0
    cursor_by_cluster = defaultdict(int)  # Keeps track of which farthest point was last used per cluster

    while selected_count < budget:
        exhausted = True  # Track if any selection was made this round

        for cluster_id in cluster_dist_lists:
            candidates = cluster_dist_lists[cluster_id]
            cursor = cursor_by_cluster[cluster_id]

            # Skip if no more candidates in this cluster
            if cursor >= len(candidates):
                continue

            _, point_idx = candidates[cursor]

            if point_idx not in selected_ids:
                selected_ids.add(point_idx)
                cursor_by_cluster[cluster_id] += 1
                selected_count += 1
                exhausted = False

                if selected_count >= budget:
                    break

        if exhausted:
            # All clusters exhausted
            break

    return selected_ids


cluster_dist_lists = precompute_sorted_cluster_points(
    torch.from_numpy(vectors), torch.from_numpy(I.squeeze()), torch.from_numpy(kmeans.centroids)
)
selected_ids_dict = {}
for p in percentage:
    budget = int((p / 100) * total_data)
    selected_ids = round_robin_farthest_selection(cluster_dist_lists, budget)
    selected_ids_dict[p] = selected_ids
        
# Read JSON Lines files
main_file_name_df = pd.read_json(args.main_file, lines=True)
for p in percentage:
    selected_rows = main_file_name_df[main_file_name_df['id'].isin(selected_ids_dict[p])]
    selected_rows.to_json(f'/path/to/scratch/data/processed/llama3/{dataset}/optimality_opt_125M/{p}_optimality_opt_125M_{dataset}.json', orient='records', lines=True)





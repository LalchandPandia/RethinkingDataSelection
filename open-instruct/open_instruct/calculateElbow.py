import numpy as np
import torch
import argparse
import faiss, json
from collections import defaultdict
    
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

inertias = []
k_values = [5, 10, 25, 50, 75, 100, 180, 200, 300, 500, 1000, 2000, 4000, 6000, 8000, 10000]

d = vectors.shape[1]

for k in k_values:
    kmeans = faiss.Kmeans(d=d, k=k, niter=20, verbose=False, gpu=False)  # set gpu=True if needed
    kmeans.train(vectors)
    inertias.append(kmeans.obj[-1])  # last objective value
print(inertias)
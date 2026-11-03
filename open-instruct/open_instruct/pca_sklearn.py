import numpy as np
import torch
from sklearn.decomposition import PCA

# Load your data
full_embedding_file = '/path/to/scratch/embedding_nvidia.pt'
full_embedding_file ='/path/to/scratch/data/opt_125M.pt'
full_embedding_file='/path/to/scratch/data/llama3_arc_challenge_8B_mean.pt'
embed_dict_active_full = torch.load(full_embedding_file, map_location=torch.device("cpu"))
ids_active_full = np.array(list(embed_dict_active_full.keys()))
'introduced squueze to ensure that the torch.Size([1119, 1, 4096]) becomes torch.Size([1119, 4096])'
vectors = torch.stack([embed_dict_active_full[k].squeeze(0) for k in ids_active_full]).to(torch.float32)
vectors -=vectors.mean(dim=0)
vectors/= vectors.std(dim=0)

# Perform PCA
n_components = 3  # or any number <= n_features
pca = PCA(n_components=n_components)
tr = pca.fit_transform(vectors)

# Get explained variance
explained_variance = pca.explained_variance_  # Variance of each PC
explained_variance_ratio = pca.explained_variance_ratio_  # % variance explained



# Save PCA values
np.savez("/path/to/scratch/embedding_llama3_arc_challenge_8B_mean_sklearn_pca.npz", ids=ids_active_full, pca=tr, variance=explained_variance, variance_ratio=explained_variance_ratio)
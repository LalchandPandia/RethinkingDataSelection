import torch, faiss
import numpy as np
import argparse
#import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(
    description="Process perplexities csv files into LP(1) subsets."
)
parser._action_groups.pop()
required = parser.add_argument_group("Required Arguments")
optional = parser.add_argument_group("Optional Arguments")



args = parser.parse_args()
input_dir='/path/to/scratch/activeSFT/embeddings/'
full_embedding_file = '/path/to/scratch/embedding_nvidia.pt'
embed_dict_active_full = torch.load(full_embedding_file, map_location=torch.device("cpu"))
print('embed_dict_active_full shape ',embed_dict_active_full.shape)
ids_active_full = list(embed_dict_active_full.keys())


# Build a mapping from id to index
id_to_index = {id_: i for i, id_ in enumerate(ids_active_full)}
full_indices = [id_to_index[id_] for id_ in ids_active_full]

#perform pca on the entire 6M dataset
vectors = torch.stack([embed_dict_active_full[k] for k in ids_active_full])
vectors-=vectors.mean(axis=0)
mat = faiss.PCAMatrix(4096, 2)
mat.train(vectors)

tr = mat.apply(vectors)
np.save('/path/to/scratch/embedding_nvidia_pca_normalized.npy', tr)




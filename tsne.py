
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
import pickle
import numpy as np
#with open('/mnt/beegfs/work/cvcs_2023_group16/clusters_1024.pickle', 'rb') as f:
with open('./data/cluster_test.pickle', 'rb') as f:
    clusters_data = pickle.load(f)

images_paths, features ,cluster_labels =clusters_data
features = np.array(list(features))
images_paths = list(images_paths)

# Reduce dimensionality using t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=len(set(images_paths))-2)
embedded_points = tsne.fit_transform(features)

# # Assuming you have a list of image file paths, load the images
# images = [(Image.open(path)).resize((25,25)) for path in images_paths]


# # Create a scatter plot of the embedded points
# plt.figure(figsize=(10, 8))
# plt.scatter(embedded_points[:, 0], embedded_points[:, 1], c=cluster_labels)
#

cdict = {
    0: "red",
    1: "blue",
    2: "green",
    3: "purple",
    4: "orange",
    5: "pink",
    6: "brown",
    7: "teal",
    8: "gray",
    9: "cyan",
    10: "magenta",
    11: "lime",
    12: "indigo",
    13: "olive",
    14: "maroon",
    15: "navy",
    16: "turquoise"
}

from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def getImage(path, zoom=0.05):
    return OffsetImage(plt.imread(path), zoom=zoom)




fig, ax = plt.subplots(figsize=(10, 8))
for g in np.unique(cluster_labels):
    ix = np.where(cluster_labels == g)
    ax.scatter(embedded_points[ix,0], embedded_points[ix,1], c = cdict[g])

for x0, y0, path in zip(embedded_points[:,0], embedded_points[:,1], images_paths):
    ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
    ax.add_artist(ab)

plt.title("t-SNE Visualization of Images in 2D")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.savefig('tsne_img_test.png')


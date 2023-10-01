import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter
import os
import os.path
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import dlib
import bz2
import urllib.request
import sys
import shutil

from utils import *
#main

folder_path = './photos'

image_files = sorted(os.listdir(folder_path))
image_files = [os.path.join(folder_path, imf) for imf in image_files]

print(image_files)
images_path=[]
for x in image_files:
    images_path.append(x)

faces, images_path = find_cnn_faces(image_files,images_path)
print(len(faces))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet50()
model.fc = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.fc.in_features, 4096)
)

print("weights loading ...")
try:
    if not torch.cuda.is_available():
        checkpoint = torch.load("./weights/contrastive_loss_big_more_epocs.pth", map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load("./weights/contrastive_loss_big_more_epocs.pth")
except Exception as e:
    print("Error loading state_dict:", e)
else:
    # Print the keys from the loaded state_dict

    # Now try to load the state_dict into the model
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print("Error loading state_dict into the model:", e)

del model.fc[0]

if torch.cuda.device_count() > 1:
    print("Utilizzo più di una GPU")
    model = nn.DataParallel(model)
model.to(device)
model.eval()
print(torch.cuda.get_device_name())


features = []
feat_tmp = extract_features(faces,model,device)
features = feat_tmp.reshape(-1, feat_tmp.shape[-1])
print(features.shape)
features = features.tolist()

dist_matrix = pairwise_distances(features, metric='euclidean')


# Applica l'algoritmo di clustering agglomerativo n_clusters=22
clustering = AgglomerativeClustering(distance_threshold=0.755*np.max(dist_matrix,axis=0).mean(), metric='euclidean', linkage='ward', compute_distances=True, n_clusters=None)

cluster_labels = clustering.fit_predict(features)

with open('./data/cluster_test.pickle', 'wb') as f:
    pickle.dump([images_path, features, cluster_labels], f)


# Find numbers of occurence for each value
counts = Counter(cluster_labels)
print(max(cluster_labels)+1)
to_be_plotted = []
# Print the result
for value, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
    print(f"{value}: {count}")
    cnt = 0
    for i, c in enumerate(cluster_labels):
        if c == value:
            print('{}'.format(images_path[i]))
            cnt+=1

# FOLDERS ORGANIZATION

with open('./data/cluster_test.pickle', 'rb') as f:
    clusters_data = pickle.load(f)

images_path, features, cluster_labels = clusters_data

features = np.array(list(features))
images_path = list(images_path)

# Create a folder for each cluster if it contains more than 5 images
for i in range(max(cluster_labels) + 1):
    cluster_images = [img for img, label in zip(images_path, cluster_labels) if label == i]
    if len(cluster_images) >= 5:
        cluster_dir = f"CLUSTERS/cluster_{i}"
        if os.path.exists(cluster_dir):
            shutil.rmtree(cluster_dir)
        os.makedirs(cluster_dir)

# Copy the images corresponding to each cluster into the respective folder
for i in range(len(cluster_labels)):
    src_path = images_path[i]  # path of the original image
    dst_path = f"CLUSTERS/cluster_{cluster_labels[i]}"  # path of the destination folder

    # Check if the destination folder exists (created earlier) and copy the image
    if os.path.exists(dst_path):
        shutil.copy(src_path, dst_path)

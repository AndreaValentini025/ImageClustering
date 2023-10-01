import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import os
import dlib
import bz2
import urllib.request
import sys
from scipy.spatial.distance import cosine,euclidean

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from utils import *


def get_max_intradist(clusters,features):

  intradist=[]
  max_intradist = []
  for c in range(max(clusters)+1):
    for e,i in enumerate(np.where(np.array(clusters) == c)[0]):

      if len(np.where(np.array(clusters) == c)[0])>1:
        toloop=np.where(np.array(clusters) == c)[0][e+1:]
      else:
        toloop=np.where(np.array(clusters) == c)[0]
      for x in toloop:
        intradist.append(euclidean(features[i],features[x]))

    max_intradist.append(max(intradist))
    intradist=[]
  return max(max_intradist)

def nearest_neighbors(features, new_features,max_dist):
  """
  Compute the most similar image
  Args:
      features: features of each face in the collection
      new_features: features of the input faces

  Returns:
      distances: Array representing the lengths to points
      indices: Indices of the nearest points in the population matrix.

  """
  features_added = np.append(features,new_features,axis=0)
  print(features_added.shape)
  pca = PCA(n_components=features_added.shape[0])
  pca_output = pca.fit_transform(features_added)
  new_img_pca = pca_output[-1]
  n_neighbors = 1  # number of nearest neighbors to search for
  nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean').fit(pca_output[:-1])
  # Find the index of the nearest neighbor(s) in the existing dataset to the new sample
  distances, indices = nbrs.kneighbors([new_img_pca])
  if distances > max_dist :
    return [], []
  return distances, indices


if len(sys.argv) > 1:
    filename = sys.argv[1]
    with open('./data/cluster_test.pickle', 'rb') as f:
        clusters_data = pickle.load(f)

    images_paths,  features, cluster_labels= clusters_data
    images_paths = list(images_paths)
    cluster_labels = list(cluster_labels)
    features = list(features)


    # controllo che l'immagine caricata non sia già presente nel pickle
    # in quel caso avrei gia il cluster a cui appartiene
    if filename in images_paths:
        clusters_found = []
        for i, x in enumerate(images_paths):
            if x == filename:
                clusters_found.append(cluster_labels[i])
        print(clusters_found)
        output = get_cluster_elements(cluster_labels, clusters_found, images_paths)
        print(output)
    else:
        to_be_clustered = []
        to_be_clustered.append(filename)
        img_pth = []
        img_pth.append(filename)
        new_img, img_pth = find_cnn_faces(to_be_clustered, img_pth)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = models.resnet50()
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.fc.in_features, 4096)
        )
        try:
            if not torch.cuda.is_available():
                checkpoint = torch.load("./weights/contrastive_loss_big_more_epocs.pth", map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load("./weights/contrastive_loss_big_more_epocs.pth")
        except Exception as e:
            print("Error loading state_dict:", e)
        else:
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
        new_features = extract_features(new_img,model,device)
        #get neighbors
        clusters_found =[]
        max_intradist = get_max_intradist(cluster_labels,features)
        for nf in new_features:
          distances, indices = nearest_neighbors(features,nf.reshape(1,-1),max_intradist)
          if len(indices) > 0:
            clusters_found.append(cluster_labels[indices[0][0]])

        print('CLUSTER DI APPARTENENZA: {}'.format( clusters_found))

        if len(clusters_found)>1:
            found = []
            for i, c in enumerate(cluster_labels):
                if c in clusters_found:
                    found.append(images_paths[i])

            print(ottieni_stringhe_n_volte(found, len(clusters_found)))

else:
    print(' SPECIFICARE UN\'IMMAGINE IN INPUT')
    print(' -> eseguire con \'python nearest_neighbor.py <percorso_immagine>')
import os
import sys
from math import ceil

import clip
import torch
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from rembg import remove
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from torchvision.models import ResNet50_Weights
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
import pickle


def extract_features_background(dataset):
    """
      Calcola le features del background di una serie di immagini rgba utilizzando  ViT di CLIP
      Args:
          dataset: tensore delle immagini rgba
      Returns:
          features: tensore delle features

      """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load('RN50', device)
    masked_tensor = dataset[:, :3, :, :]
    with torch.no_grad():
        features = model.encode_image(masked_tensor.to(device))
    return features


def save_features_with_pickle(image_folder, destination_file):
    """
    Estrae le features tramite CLIP RSN50 e le salva utilizzando pickle.

    Args:
        image_folder: Cartella contenente le immagini.
        destination_file: Nome del file in cui si vogliono salvare le features.
    """

    image_path = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if
                  img.endswith(('.jpg', '.jpeg', '.png', '.JPEG', '.JPG'))]

    image_tensors = []
    total_features = np.empty((0, 1024))

    # utilizzo delle epoche per evitare sovraccarico del server

    batch_size = 200  # Dimensione del batch per la GPU
    epoc = [0,1,2]
    for i in epoc:
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(image_path))

        for img_name in image_path[start_index:end_index]:
            tensor = get_background_image(img_name)
            image_tensors.append(tensor)

        dataset = torch.cat(image_tensors, dim=0)
        image_tensors = []

        features = extract_features_background(dataset)
        total_features = np.vstack((total_features, features.detach().cpu().numpy()))

    with open(destination_file, 'wb') as pickle_file:
        pickle.dump(total_features, pickle_file)


def get_background_image(image_path):
    """
  Estrae l'immagine di backround
  Args:
      image_path: percorso all'immagine specificata

  Returns:
      img_tensor: tensore contenente l'immagine specificata in formato RGBA,
                dove A=0 se il pixel non appartiene al background

  """

    # Define the preprocessing transformation
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = plt.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    foreground = remove(img)
    background = np.where(foreground[..., 3][:, :, np.newaxis] == 0, img, 0)

    img_tensor = preprocess(background).unsqueeze(0)

    return img_tensor


def get_features(image_folder):
    image_path = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if
                  img.endswith(('.jpg', '.jpeg', '.png', '.JPEG', '.JPG'))]

    image_tensors = []
    total_features = np.empty((0, 1024))

    # utilizzo delle epoche per evitare sovraccarico del server

    batch_size = 200  # Dimensione del batch per la GPU
    i = 0
    while True:
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(image_path))
        if start_index >= end_index:
            break

        for img_name in image_path[start_index:end_index]:
            tensor = get_background_image(img_name)
            image_tensors.append(tensor)

        dataset = torch.cat(image_tensors, dim=0)
        image_tensors = []

        features = extract_features_background(dataset)
        total_features = np.vstack((total_features, features.detach().cpu().numpy()))

        i += 1

    return total_features


def save_features_with_pickle(image_folder, destination_file):
    """
    Estrae le features tramite CLIP RSN50 e le salva utilizzando pickle.

    Args:
        image_folder: Cartella contenente le immagini.
        destination_file: Nome del file in cui si vogliono salvare le features.
    """

    total_features = get_features(image_folder)

    with open(destination_file, 'wb') as pickle_file:
        pickle.dump(total_features, pickle_file)


def dbscan_clustering(features, eps=0.28):
    clustering = DBSCAN(eps=eps, min_samples=2, metric='cosine')

    n_components = min(features.shape[0], 128)

    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)

    cluster_labels = clustering.fit_predict(features_pca)

    from collections import Counter

    counts = Counter(cluster_labels)
    # Stampa il risultato con un carattere di fine riga dopo ogni virgola
    for value, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        print(f"{value}: {count}")

    num_clusters = len(np.unique(cluster_labels)) - 1
    print(num_clusters)
    cluster_sizes = np.zeros(num_clusters)
    for i in range(num_clusters):
        cluster_sizes[i] = np.sum(cluster_labels == i)

    return cluster_labels, cluster_sizes


def print_background_clusters(cluster_labels, image_paths, cluster_sizes, n_cluster=5):

   lowest_indices = np.argsort(cluster_sizes)[::-1][:n_cluster]

    for cluster in lowest_indices:
        # Seleziona le immagini del cluster corrente
        cluster_images = image_paths[cluster_labels == cluster]

        num_images = len(cluster_images)
        plt.figure(figsize=(10, 10))
        print('CLUSTER')
        for i, image_path in enumerate(cluster_images):

            img = plt.imread(image_path)
            plt.subplot(ceil(num_images / 5), 5, i + 1)

            plt.imshow(img)
            plt.axis('off')

        # Mostra i subplot per le immagini del cluster corrente
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    #MODIFICARE I SEGUENTI PARAMETRI
    image_folder = ''   #cartella contenente le immagini
    destinatio_folder = '....../bg_features.pkl' #percorso del file dove vengono salvate le features

    '''
    Se si vuole testare il salvataggio su file decommentare questa parte 
    e commentare la funzione get_features
    
    save_features_with_pickle(image_folder, dest_file)
    features = load_features_from_file(dest_file)
    
    '''

    image_path = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if
                  img.endswith(('.jpg', '.jpeg', '.png', '.JPEG', '.JPG'))]

    features = get_features(image_folder)

    cluster_labels, cluster_size = dbscan_clustering(features)
    print_background_clusters(cluster_labels, np.array(image_path), cluster_size, n_cluster=5)


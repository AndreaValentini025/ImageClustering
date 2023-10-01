import os

import clip
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def clip_top_images(input_text, image_folder, threshold=19):
    """

    :param input_text: stringa che voglio cercare nelle immagini
          image_folder: cartella contenente le immagini tra cui cercare
    :return: una tuple (sorted_logits, sorted_image_paths, lim_positive_indices) che contiene i
            i logits ordinati in ordine decrescente sulla base del match con il testo
            i percorsi delle immagini ordinati per logits decresenti
            indice dell'ultima immagine mostrata
    """

    # Load the model & images
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('RN50', device)

    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if
                   img.endswith(('.jpg', '.jpeg', '.png', '.JPEG', '.JPG'))]

    image_dataset = []
    for image_path in image_paths:
        image_tensor = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_dataset.append(image_tensor)
        image_dataset_tensor = torch.cat(image_dataset, dim=0).to(device)

    text_tensor = clip.tokenize(input_text).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image_dataset_tensor, text_tensor)

    logits_per_image = logits_per_image.cpu()
    logits_per_text = logits_per_text.cpu()

    sorted_indices = np.argsort(-logits_per_text)
    sorted_logits = logits_per_text[0, sorted_indices]

    sorted_image_paths = [image_paths[i] for i in sorted_indices[0]]

    positive_indices = np.where(sorted_logits > threshold)[1]

    if len(positive_indices) > 10:
        lim_positive_indices = positive_indices[:10]
    else:
        lim_positive_indices = positive_indices

    # Mostra le immagini
    count = 1
    for i in lim_positive_indices:
        image_path = sorted_image_paths[i]
        logits = sorted_logits[:, i]
        # Mostra l'immagine utilizzando plt
        image = plt.imread(image_path)
        plt.imshow(image)
        plt.title(f"{input_text}: {count}°")
        plt.axis('off')
        plt.show()
        count += 1

    last_index = 0
    if len(lim_positive_indices) > 0:
        last_index = lim_positive_indices[-1]

    return sorted_logits, sorted_image_paths, last_index


def load_other_images(sorted_logits, sorted_image_paths, last_index, n_logits):
    """
    :param sorted_logits:
    :param sorted_image_paths:
    :param last_index: indice dell'ultima immagine mostrata
    :param n_logits: ultimo intero considerato utilizzato
    :return: una tupla (last index, n_logits)
    """

    if last_index == len(sorted_logits) - 1:
        print("Non sono disponibili altre immagini")
        return last_index, n_logits

    # controllo gli indici che sono già stati mostrati
    positive_indices = np.where(sorted_logits > n_logits)[1]
    lim_positive_indices = positive_indices

    # ho già mostrato tutte le immagini che dovevo mostrare
    if len(lim_positive_indices) == 0 or lim_positive_indices[-1] == last_index:

        # Continuo ad abbassare n_logits di uno alla volta finché non c'è almeno un elemento
        while True:
            n_logits -= 1
            lim_positive_indices = np.where((sorted_logits >= n_logits) & (sorted_logits < n_logits + 1))[1]
            if len(lim_positive_indices) != 0:
                break

    # devo ancora mostrare una parte di immagini
    else:
        lim_positive_indices = positive_indices[10:]

    # Mostra le immagini
    for i in lim_positive_indices:
        image_path = sorted_image_paths[i]
        logits = sorted_logits[:, i]

        # Mostra l'immagine utilizzando plt
        image = plt.imread(image_path)
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    return lim_positive_indices[-1], n_logits


if __name__ == '__main__':
    # MODIFICARE I PROSSIMI PARAMETRI PRIMA DI LANCIARE IL PROGRAMMA
    input_text = 'a soccer jersey'  # testo da cercare nelle immagini
    image_folder = 'drive/MyDrive/CV dataset'  # cartella contenente le immagini

    th = 16
    sorted_logits, sorted_image_paths, last_index = clip_top_images(input_text, image_folder, threshold=th)
    if last_index == -1:
        print("Nessuna immagine ha avuto un riscontro positivo.")

    load_again = True
    while load_again:
        answer_incorrect = True
        while answer_incorrect:
            answer = input("Caricare altre immagini? (Y/N)").lower()
            if answer == 'y':
                last_index, th = load_other_images(sorted_logits, sorted_image_paths, last_index, th)
                answer_incorrect = False
            elif answer == 'n':
                load_again = False
                answer_incorrect = False
            else:
                print("La risposta non era corretta. Inserire Y se si vuole caricare altre immagini, N altrimenti")
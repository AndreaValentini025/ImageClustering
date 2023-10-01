import numpy as np
import cv2
import os
import dlib
import bz2
import urllib.request
import pickle
import matplotlib.pyplot as plt
import sys
from utils import *

#algorithm for face detection using dlib library and resize of the cropped faces to (224,224)

def find_cnn_faces(images):
    """
          extract faces from given images
          Args:
              images: list of images

          Returns:
              out_faces: list of faces detected
              images_path: list of images paths where faces has been detected
          """

    file_path = "./models/mmod_human_face_detector.dat"
    if not os.path.isfile(file_path):
        url = "https://github.com/davisking/dlib-models/raw/master/mmod_human_face_detector.dat.bz2"
        urllib.request.urlretrieve(url, file_path)

    out_faces = []
    dnnFaceDetector = dlib.cnn_face_detection_model_v1("./models/mmod_human_face_detector.dat")
    target_size = (224, 224)
    truth_labels = []
    addedpaths = 0
    images_path = [item[0] for item in data]
    for c, (x, y) in enumerate(images):
        print(c)
        img = cv2.imread(x, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        rects = dnnFaceDetector(img, 1)
        if len(rects) > 1 or len(rects) == 0:
            images_path.remove(x)
            continue
        for (i, rect) in enumerate(rects):
            x1 = rect.rect.left()
            y1 = rect.rect.top()
            x2 = rect.rect.right()
            y2 = rect.rect.bottom()
            # Rectangle around the face
            if x1 - 10 < 0 or y1 - 10 < 0 or x2 + 10 > width or y2 + 10 > height:
                images_path.remove(x)
                continue
            else:
                new_img = align_face(img, x1, x2, y1, y2)  # img[y1-10:y2+10, x1-10:x2+10]
                if rect.confidence > 0.9:
                    if i > 0:
                        images_path.insert(c + 1 + addedpaths, images_path[c + addedpaths])
                        addedpaths += 1
                    new_img = cv2.resize(new_img, target_size)
                    out_faces.append([new_img, y])
                    if c % 1000 == 0 and c>0:
                        with open('./data/faces_{}.pickle'.format(batch_idx), 'wb') as f:
                            pickle.dump([out_faces, images_path], f)
                else:
                    images_path.remove(x)

    return out_faces, images_path

if __name__=="__main__":

    batch_idx = int(sys.argv[1])

    
    start_batch_idx = batch_idx*10000
    end_batch_idx = start_batch_idx+10000


    folder_path = './data/img_align_celeba'
    labels_file_path = './data/identity_CelebA.txt'


    image_files = sorted(os.listdir(folder_path))

    #images initialization for each job
    images = image_files[start_batch_idx:end_batch_idx]

    with open(labels_file_path, 'r') as f:
        rows = f.read().splitlines()

    r = rows[start_batch_idx:end_batch_idx]
    data = []
    for row in r:
        images, labels = row.strip().split(' ')
        data.append([os.path.join(folder_path, images),int(labels)])


    faces, images_path = find_cnn_faces(data)

    with open('./data/faces_{}.pickle'.format(batch_idx), 'wb') as f:
        pickle.dump([faces, images_path], f)

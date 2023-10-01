import os
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import cv2
import dlib
import bz2
import urllib.request
import sys
import random

from PIL import Image
import matplotlib.image as mpimg
import torch.nn.functional as F

LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom

def extract_eye(shape, eye_indices):
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)

def extract_eye_center(shape, eye_indices):
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6

def extract_left_eye_center(shape):
    return extract_eye_center(shape, LEFT_EYE_INDICES)

def extract_right_eye_center(shape):
    return extract_eye_center(shape, RIGHT_EYE_INDICES)

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M

def crop_image(image, det):
    left, top, right, bottom = rect_to_tuple(det)
    return image[top:bottom, left:right]


def align_face(img,x1,x2,y1,y2):
    """
          Align the eyes-line of a face horizontally
          Args:
              img: the initial image
              x1: top left corner coordinate
              x2: bottom right corner coordinate
              y1: top left corner coordinate
              y2: bottom right corner coordinate

          Returns:
              crop_image(rotated, rect_face): aligned face
    """
    height, width = img.shape[:2]
    dlibPredictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
    rect_face = dlib.rectangle(left=max(x1-10,0),top=max(y1-10,0),right=min(x2+10,width),bottom=min(y2+10,height))
    shape = dlibPredictor(img, rect_face)
    left_eye = extract_left_eye_center(shape)
    right_eye = extract_right_eye_center(shape)

    M = get_rotation_matrix(left_eye, right_eye)
    rotated = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_CUBIC)

    return crop_image(rotated, rect_face)


def find_cnn_faces(images,images_path):
    file_path = "./models/mmod_human_face_detector.dat"
    if not os.path.isfile(file_path):
        url = "https://github.com/davisking/dlib-models/raw/master/mmod_human_face_detector.dat.bz2"
        urllib.request.urlretrieve(url, file_path)

    out_faces = []
    dnnFaceDetector = dlib.cnn_face_detection_model_v1("./models/mmod_human_face_detector.dat")
    target_size = (224, 224)
    addedpaths = 0
    for c, x in enumerate(images):
        print(c)
        if os.path.isdir(x):
            continue
        img = cv2.imread(x, cv2.COLOR_BGR2RGB)
        img = gamma_correct_image(img)
        height, width = img.shape[:2]
        rects = dnnFaceDetector(img, 1)
        if len(rects) == 0:
            images_path.remove(x)
            continue

        for (i, rect) in enumerate(rects):
            x1 = rect.rect.left()
            y1 = rect.rect.top()
            x2 = rect.rect.right()
            y2 = rect.rect.bottom()
            # Rectangle around the face

            new_img = align_face(img, x1, x2, y1, y2)  # img[y1-10:y2+10, x1-10:x2+10]

            if rect.confidence > 0.9:
                if i > 0:

                    if c+1+addedpaths > len(images_path):
                        images_path.append(images_path[-1])
                    else :
                        images_path.insert(c + 1 + addedpaths, images_path[c + addedpaths])
                        addedpaths += 1
                new_img = cv2.resize(new_img, target_size)
                out_faces.append(new_img)
            else:
                if i<1:
                    images_path.remove(x)
                    addedpaths-=1
    return out_faces, images_path


def heaviside(x):
      """
      Implementation of the Heaviside step function (https://en.wikipedia.org/wiki/Heaviside_step_function)
      Args:
      x: Numpy-Array or single Scalar
      Returns:
      x with step values
      """
      if x <= 0:
              return 0
      else:
              return 1


def adaptiveGammaCorrection(v_Channel):

    """https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-016-0138-1#CR14
    Applies adaptive Gamma-Correction to V-Channels of an HSV-image.

      Args:
      v_Channel: Numpy-Array (uint8) representing the "Value"-Channel of the image
      Returns:
        Corrected channel
      """
    #calculate general variables
    I_in = v_Channel/255.0
    I_out = I_in
    sigma = np.std(I_in)
    mean = np.mean(I_in)
    D = 4*sigma
    #low contrast image
    if D <= 1/3:
      gamma = - np.log2(sigma)
      I_in_f = I_in**gamma
      mean_f = (mean**gamma)
      k =  I_in_f + (1 - I_in_f) * mean_f
      c = 1 / (1 + heaviside(0.5 - mean) * (k-1))
      #dark
      if mean < 0.5:
        I_out = I_in_f / ((I_in_f + ((1-I_in_f) * mean_f)))
      #bright
      else:
        I_out = c * I_in_f
    #high contrast image
    elif D > 1/3:
      gamma = np.exp((1- (mean+sigma))/2)
      I_in_f = I_in**gamma
      mean_f = (mean**gamma)
      k =  I_in_f + (1 - I_in_f) * mean_f
      c = 1/ (1 + heaviside(0.5 - mean) * (k-1))
      I_out = c * I_in_f
    else:
      print('Error calculating D')

    I_out = I_out*255

    return I_out.astype(np.uint8)

def gamma_correct_image(img):
  """
  Adaptive Gamma Correction, that automatically assesses
  which gamma level to apply, based on whether an image
  has high or low contrast.

  Args:
      img: immagine corretta

  Returns:
      corrected_image: immagine corretta

  """
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  h, s, v = cv2.split(hsv)

  v = adaptiveGammaCorrection(v)
  #finds border or intensity variation with laplacian gradient
  gradient =  cv2.Laplacian(s,cv2.CV_32F, ksize = 1)  #cv2.Laplacian(s, cv2.CV_32F, ksize = 1)

  #smoothing gradient
  clipped_gradient = gradient * np.exp(-1 * np.abs(gradient) * np.abs(s - 0.5))

  #normalize to [-1...1]
  clipped_gradient =  2*(clipped_gradient - np.max(clipped_gradient))/-np.ptp(clipped_gradient)-1

  clipped_gradient =  0.5 * clipped_gradient #--> 0.5 limits maximum saturation change to 50 %

  factor = np.add(1.0, clipped_gradient)
  s = np.multiply(s, factor)
  s = cv2.convertScaleAbs(s)

  final_CLAHE = cv2.merge((h,s,v))

  corrected_image = cv2.cvtColor(final_CLAHE, cv2.COLOR_HSV2RGB)

  return corrected_image



def preprocess_image(img):
  '''
  Preprocessa le immagini in modo che possano essere passate nel modello

  Args:
    img : immagine da normalizzare

  Returns:
    img : immagine normalizzata
  '''
  img = cv2.resize(img, (224, 224))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = np.expand_dims(img, axis=0)
  return img


def extract_features(faces,model,device):
    num_samples = len(faces)

    if len(faces) > 1:
        preprocessed_faces = np.array([preprocess_image(face) for face in faces])
        x = torch.tensor(np.squeeze(preprocessed_faces), dtype=torch.float).to(device)
    else:
        if len(faces) == 0:
            print('NO FACE DETECTED. PLEASE SELECT ANOTHER PHOTO AND RETRY')
            exit(1)
        x = torch.tensor(preprocess_image(faces[0]), dtype=torch.float).to(device)
    x = x.permute(0, 3, 1, 2)

    print(x.shape)
    feat = model(x)
    return feat.detach().cpu().numpy()



def ottieni_stringhe_n_volte(vettore, n):
    '''
    codice Python che prende un vettore di stringhe
    e restituisce un secondo vettore contenente solo
    le stringhe che compaiono un numero
    specifico di volte
    '''
    conteggio_stringhe = {}
    for stringa in vettore:
        if stringa in conteggio_stringhe:
            conteggio_stringhe[stringa] += 1
        else:
            conteggio_stringhe[stringa] = 1

    vettore_risultato = []
    for stringa in vettore:
        if conteggio_stringhe[stringa] == n and stringa not in vettore_risultato:
            vettore_risultato.append(stringa)

    return vettore_risultato
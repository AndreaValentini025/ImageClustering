# Image Clustering and Retrieval in Large Photo Collection
*This Computer Vision project proposes an algorithm
that aims to detect all the faces in a collection of photos in
order to cluster them by identities and organize them into
dif erent folders. Given a photo, the algorithm will also
recognize the faces in it and search them in the file
system’s images, displaying only the images that contain
the recognized faces of the given one. Moreover, the
application provides the capability to search within the
collection of images for the one that most closely matches
a user-provided input string. Finally, the application
labels the images based solely on their background and
returns the top 5 most significant clusters.*

**[READ THE PAPER HERE](https://github.com/AndreaValentini025/ImageClustering/blob/main/Image%20Clustering%20and%20Retrieval%20in%20Large%20Photo%20Collection.pdf)**

**It is necessary to install the following packages:**

pip install git+https://github.com/openai/CLIP.git

pip install torch

pip install torchvision

pip install opencv-python

pip install numpy

pip install matplotlib

pip install scikit-learn

pip install pandas

pip install pickle-mixin

pip install dlib

pip install scipy


pip install rembg

for gpu support use pip install rembg[gpu]


The Face Detection and Alignment function is implemented in each file using user images, that MUST be stored into the 'photos' folder.


To test *Feature Extraction* , *Face Clustering* and *Folder Organization* functions, run the "cluster_images.py" file. This file reads all images present in the 'photos' folder, extracts and aligns each face detected 
and then apply Feature Extraction on them using the pretrained model.
The Agglomerative Clustering is then applied on the extracted features to return the clusters computed.
(a pickle is saved in 'data' folder containing imgs_paths, features and cluster_id)

At this point, the program copies 'photos' images into the 'CLUSTER' folder.

This folder contains subdirectories for each cluster having at least 5 images in it.

The code automatically searchs the model pretrained weights in the 'weights' folder.

CelebA 4096-embedding space 90-epochs pretrained weights are given in that folder.

To retrain the model:

THE DATASET MUST BE DOWNLOADED AND EXTRACTED INTO THE FOLDER 'data'.

CelebA dataset can be downloaded following this link: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg&usp=sharing present on the official page. (Align&Cropped version)

To extract and crop all faces present in the dataset, an slurm array job has been used. (Code in script.sh, probably must be adjusted or removed, but in this case the "face_extraction.py" must be modified.)

The script parallelize the extraction executing the "face_extraction.py" program on different subset of the dataset.

When the parallel extraction is terminated, "dataset_preparation.py" must be executed.

The dataset is now divided in TrainSet and TestSet, so training can be started using 'train_net.py', which accept 1 parameter (a random value) to set if the net must start from the ImageNet1K weights or from the given pretrained weights.

The given train code let you train the net for 30 epochs, with checkpoint every 5 epochs.


The file "tsne.py" creates a graphical representation of the clusters embedding space.


To test *Image Retrieval* function, run the "nearest_neighbor.py" specifying as the only parameter the input image path (relative or absolute).

The program will extract all the faces in the input image and it will return the relative cluster, or the clusters intersection if more than 1 identities is provided. 



To test *Background Clustering*, navigate to the "main_bg_clustering.py" file. Before running the main script: make the necessary modifications to the two indicated parameters.

The normal execution includes plotting the first 5 clusters found without saving the features to a file. If you want to test feature saving to a file, follow the instructions in the main script.


If you want to test the *Image Search* function, go to the "main_image_search.py" file. Before running the main script, make the necessary modifications to the two indicated parameters.

The program will plot the images most similar to the text, then ask if you want to load more images. Enter Y/N based on whether you want to view more images or terminate the program.

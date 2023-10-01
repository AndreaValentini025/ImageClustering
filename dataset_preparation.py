import pickle
import numpy as np
import cv2
import random


def suddividi_dataset(dataset, percentuale_training):
    images, labels, image_paths = map(list, zip(*dataset))
    a = np.array(images)
    b = np.array(labels)
    c = np.array(image_paths)
    indices = np.arange(a.shape[0])
    np.random.shuffle(indices)
    images = a[indices]
    labels = b[indices]
    image_paths = c[indices]
    lunghezza_training = int(len(images) * percentuale_training)
    training_set = list(
        zip(images[:lunghezza_training], labels[:lunghezza_training], image_paths[:lunghezza_training]))
    test_set = list(
        zip(images[lunghezza_training:], labels[lunghezza_training:], image_paths[lunghezza_training:]))
    return training_set, test_set

def create_train_dict(dataset):
    for i, (img, label, _) in enumerate(dataset):
        if label not in dict_train_index_labels.keys():
            dict_train_index_labels[label] = []
        dict_train_index_labels[label].append(i)

def create_test_dict(dataset):
    for i, (img, label, _) in enumerate(dataset):
        if label not in dict_test_index_labels.keys():
            dict_test_index_labels[label] = []
        dict_test_index_labels[label].append(i)


if __name__=="__main__":
    #Create a list of file paths for the pickle files to be merged
    file_paths = []
    for i in range(21):
        file_paths.append('./data/faces_{}.pickle'.format(i))

    #Initialize a list to store the data from each pickle file
    merged_data = []

    total_images = []
    total_labels = []
    total_paths = []
    #Load data from each pickle file and combine it into merged_data
    for i, file_path in enumerate(file_paths):
        with open(file_path, 'rb') as f:
            print(i)
            faces, paths = pickle.load(f)
            for image, label in faces:
                total_images.append(image)
                total_labels.append(label)
            total_paths.extend(paths)

    merged_data = list(zip(total_images, total_labels, total_paths))

    with open('./data/celeba_faces.pickle', 'wb') as f:
        pickle.dump(merged_data, f)

    # split dataset into trainset and testset 80/20



    trainset_data, testset_data = suddividi_dataset(merged_data, 0.8)

    trainset = np.array(trainset_data, dtype=object)
    testset = np.array(testset_data, dtype=object)

    # convert images from BGR to RGB
    for i, (x, y, z) in enumerate(trainset_data):
        img = x
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        trainset[i] = [img, y, z]

    for i, (x, y, z) in enumerate(testset_data):
        img = x
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        testset[i] = [img, y, z]

    trainset = np.stack(trainset, axis=0)
    testset = np.stack(testset, axis=0)

    # # save total trainset on pickle
    # with open('/mnt/beegfs/work/cvcs_2023_group16/trainset_celeba.pickle', 'wb') as f:
    #     pickle.dump(trainset, f)
    #
    # # save total testset on pickle
    # with open('/mnt/beegfs/work/cvcs_2023_group16/testset_celeba.pickle', 'wb') as f:
    #     pickle.dump(testset, f)

    dict_train_index_labels={}
    dict_test_index_labels={}



    create_train_dict(trainset)
    create_test_dict(testset)
    train_idx_delete =[]
    test_idx_delete =[]
    for k,v in dict_train_index_labels.items():
        if len(v) == 1:
            train_idx_delete.append(v[0])


    l = list(range(len(trainset)))
    for x in train_idx_delete:
        l.remove(x)

    trainset = np.array(trainset)
    trainset = trainset[l]
    trainset = trainset.tolist()


    for k,v in dict_test_index_labels.items():
        if len(v) == 1:
            test_idx_delete.append(v[0])

    l = list(range(len(testset)))
    for x in test_idx_delete:
        l.remove(x)

    testset = np.array(testset)
    testset = testset[l]
    testset = testset.tolist()

    with open('./data/trainset_celeba_cleaned.pickle', 'wb') as f:
        pickle.dump(trainset,f)

    with open('./data/testset_celeba_cleaned.pickle', 'wb') as f:
        pickle.dump(testset,f)

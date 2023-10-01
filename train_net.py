import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.models import ResNet50_Weights
import sys
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F


transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#creation of the train's dictionary
def create_train_dict(dataset):
    for i, (img, label, _) in enumerate(dataset):
        if label not in dict_train_index_labels.keys():
            dict_train_index_labels[label] = []
        dict_train_index_labels[label].append(i)


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = torch.matmul(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.view(-1, 1) - 2.0 * dot_product + square_norm.view(1, -1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.clamp(distances, min=0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have the same label.

    Args:
        labels: torch tensor of shape [batch_size]

    Returns:
        mask: torch tensor of shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    
    indices_equal = torch.eye(labels.size(0)).bool().to(device)
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = (labels.unsqueeze(0) == labels.unsqueeze(1))

    # Combine the two masks
    mask = indices_not_equal & labels_equal

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: torch tensor of shape [batch_size]

    Returns:
        mask: torch tensor of shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = (labels.unsqueeze(0) == labels.unsqueeze(1))

    mask = ~labels_equal

    return mask


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, torch tensor of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have the same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive.to(device)
    mask_anchor_positive = mask_anchor_positive.float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = pairwise_dist * mask_anchor_positive

    # shape (batch_size, 1)
    hardest_positive_dist = torch.max(anchor_positive_dist, dim=1, keepdim=True)[0]

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative.to(device)
    mask_anchor_negative = mask_anchor_negative.float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = torch.min(anchor_negative_dist, dim=1, keepdim=True)[0]
    # tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))
    
    # Combine the biggest d(a, p) and smallest d(a, n) into the final triplet loss
    triplet_loss = torch.max(hardest_positive_dist - hardest_negative_dist + margin, torch.tensor(0.0))
    # Get the final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss


class TripletDataset(torch.utils.data.Dataset):
    '''
        Dataset items
    '''
    def __init__(self, dataset, train=True):
        self.dataset = dataset
        self.train = train

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, item):
        if self.train:
            anchor_img, anchor_label, anchor_path = self.dataset[shuffeled[item]]
        else:
            anchor_img, anchor_label, anchor_path = self.dataset[item]

        return anchor_img, anchor_label, anchor_path


def shuffleData():
    '''
     function to shuffle data before each epoch
    '''
    n = len(trainset)
    global shuffeled
    shuffeled = torch.randperm(n).tolist()

def collate_fn_train(data):
    """
        function to manipulate the batches, adding positive example to compute the triplet loss.
       data: is a list of tuples with (img, label, path)
    """
    imgs, labels, paths = zip(*data)
    imgs = list(imgs)
    labels = list(labels)
    paths = list(paths)
    last_labels = []
    for i in range(4):
        save_cnt = 0
        while True:
            label_idx = random.choice(range(len(labels)))
            if labels[label_idx] not in last_labels:
                last_labels.append(labels[label_idx])
                break
            if save_cnt > 32:
                break
            else:
                save_cnt += 1
        iterations = random.randint(1, len(dict_train_index_labels[labels[label_idx]]) - 1)
        pos_list_idx = []
        pos_list_idx.append(label_idx)
        for j in range(iterations):
            while True:
                idx_rnd = random.choice(range(len(labels)))
                if idx_rnd not in pos_list_idx:
                    break
            save_cnt = 0
            while True:
                choosen = random.choice(dict_train_index_labels[labels[label_idx]])
                if trainset[choosen][2] not in [paths[x] for x in pos_list_idx]:
                    break
                if save_cnt > 200:
                    break
                else:
                    save_cnt += 1
            pos_list_idx.append(idx_rnd)
            imgs[idx_rnd] = trainset[choosen][0]
            labels[idx_rnd] = trainset[choosen][1]
            paths[idx_rnd] = trainset[choosen][2]

    return imgs,labels,paths



with open('./data/trainset_celeba_cleaned.pickle', 'rb') as f:
    trainset = pickle.load(f)
batch_size = 32


dict_train_index_labels={}
shuffeled = []


train_dataset = TripletDataset(trainset,train=True)
create_train_dict(trainset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn = collate_fn_train)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

embedding_space_dim = 4096
optimizer = None

if len(sys.argv) > 1:
    model = models.resnet50()
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.fc.in_features, embedding_space_dim),
    )
    try:
        checkpoint = torch.load("./weights/contrastive_loss_big.pth")
        previous_loss = checkpoint['loss']

    except Exception as e:
        print("Error loading checkpoint:", e)
    else:
        # Print the keys from the loaded state_dict

        # Now try to load the state_dict into the model
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print("Error loading state_dict into the model:", e)
else:
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.fc.in_features, embedding_space_dim),
    )
    init.normal_(model.fc[1].weight, mean=0.0, std=0.01)
    if model.fc[1].bias is not None:
        init.zeros_(model.fc[1].bias)

if torch.cuda.device_count() > 1:
    print("Using {} GPUs for training.".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)

model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
if len(sys.argv) > 1:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


epochs = 30
train_losses = []

#training with triplet loss for 30 epochs
for e in range(epochs):
    shuffleData()
    train_losses = []

    model.train()
    for i, (anchors, labels, paths) in enumerate(train_loader):
        anchors = torch.tensor(anchors, dtype=torch.float).to(device)
        labels = torch.tensor(labels, dtype=torch.int).to(device)

        optimizer.zero_grad()

        anchors_embeddings = model(anchors)

        loss = batch_hard_triplet_loss(labels, anchors_embeddings, margin=0.5)
        train_losses.append(loss.cpu().detach().numpy())
        loss.backward()

        optimizer.step()

    #weights saving every 5 epochs
    if (e+1) % 5 == 0:
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': np.mean(train_losses),
        }, "./weights/contrastive_loss_big_more_epocs.pth")

torch.save({
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': np.mean(train_losses),
            },"./weights/contrastive_loss_big_more_epocs.pth")

from IPython.display import clear_output 
import sys
sys.path.insert(0,'C:/Users/Mey/Documents/PlantDiseaseDiagnosisFewShotLearning/siamese_triplet_net/src/')
import torch
import timm
from dataloaders import get_train_transforms, get_val_transforms, get_siamese_dataloader, get_triplet_dataloader
from networks import SiameseNet, TripletNet 
from models import *
from losses import ContrastiveLoss, TripletLoss
from trainer import fit
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Vision Transformer model
embedding_net = timm.create_model('vit_base_patch16_224', pretrained=True)
embedding_net.head = torch.nn.Identity()  # Remove the classification head

model = TripletNet(embedding_net=embedding_net)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
loss_fn = TripletLoss(1.)
n_epochs = 7  # 100

if torch.cuda.is_available():
    model.cuda()

log_interval = 10  # 100
path_data = 'C:/Users/Mey/Documents/PlantDiseaseDiagnosisFewShotLearning/siamese_triplet_net/src/dataset'
triplet_train_loader = get_triplet_dataloader(root=path_data + '/train/', batch_size=5, transforms=get_train_transforms())
triplet_val_loader = get_triplet_dataloader(root=path_data + '/val/', batch_size=5, transforms=get_val_transforms())

fit(triplet_train_loader, triplet_val_loader, model, loss_fn, optimizer, lr_scheduler, n_epochs, device, log_interval)

import cv2
import numpy as np
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torch.autograd import Variable
import os
import pandas as pd
import seaborn as sns

def generate_embeddings(data_loader, model):
    with torch.no_grad():
        model.eval()
        labels = None
        embeddings = None
        for batch_idx, data in tqdm(enumerate(data_loader)):
            batch_imgs, batch_labels = data
            batch_labels = batch_labels.numpy()
            batch_E = model.get_embedding(batch_imgs)
            batch_E = batch_E.data.cpu().numpy()
            embeddings = np.concatenate((embeddings, batch_E), axis=0) if embeddings is not None else batch_E
            labels = np.concatenate((labels, batch_labels), axis=0) if labels is not None else batch_labels
    return embeddings, labels

test_data = torchvision.datasets.ImageFolder(root=path_data + '/test/', transform=get_val_transforms())
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
val_embeddings_cl, val_labels_cl = generate_embeddings(test_loader, model)
vis_tSNE(val_embeddings_cl, val_labels_cl)

train_data = torchvision.datasets.ImageFolder(root=path_data + '/train/', transform=get_val_transforms())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
test_data = torchvision.datasets.ImageFolder(root=path_data + '/test/', transform=get_val_transforms())
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)
x_train, y_train = generate_embeddings(train_loader, model)
x_test, y_true = generate_embeddings(test_loader, model)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
accuracy = round(accuracy_score(y_true, y_pred) * 100, 2)
precision = round(precision_score(y_true, y_pred, average='macro') * 100, 2)
recall = round(recall_score(y_true, y_pred, average='macro') * 100, 2)
f1 = round(f1_score(y_true, y_pred, average='macro') * 100, 2)

print(f'--- Results for ViT Embeddings on KNN (k = 1) ---')
print(f'Accuracy Score: {accuracy}')
print(f'Precision Score: {precision}')
print(f'Recall Score: {recall}')
print(f'F1 Score: {f1}')

import sys

# sys.path.insert(0, '/content/drive/MyDrive/pg/siamese_triplet_net/src/')
# sys.path.insert(0,'C:/Users/Mey/Documents/pg-coffee-main/siamese_triplet_net/src/')
sys.path.insert(0, 'C:/Users/Mey/Documents/PlantDiseaseDiagnosisFewShotLearning/siamese_triplet_net/src/')
import torch

device = torch.cuda.is_available()
from dataloaders import get_train_transforms, get_val_transforms, get_triplet_dataloader
from networks import TripletNet
from models import *
from losses import TripletLoss
from trainer import fit
import torchvision

# model & optimizer & lr_scheduler
embedding_net = MobileNetv2()
model = TripletNet(embedding_net=embedding_net)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
loss_fn = TripletLoss(1.)
n_epochs = 100

if device:
    model.cuda()

log_interval = 100

# path to data
# path_data = '/content/drive/MyDrive/pg/dataset/'
path_data = 'C:/Users/Mey/Documents/PlantDiseaseDiagnosisFewShotLearning/siamese_triplet_net/src/dataset'

# define siamese train and val loaders
# this loader is implemented for datasets in ImageFolder format (https://pytorch.org/vision/stable/datasets.html#imagefolder)
triplet_train_loader = get_triplet_dataloader(root=path_data + '/train/', batch_size=32,
                                              transforms=get_train_transforms())
triplet_val_loader = get_triplet_dataloader(root=path_data + '/val/', batch_size=32, transforms=get_val_transforms())

fit(triplet_train_loader, triplet_val_loader, model, loss_fn, optimizer, lr_scheduler, n_epochs, device, log_interval)

# from: https://github.com/avilash/pytorch-siamese-triplet/blob/master/tsne.py
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.autograd import Variable


def generate_embeddings(data_loader, model):
    with torch.no_grad():
        #device = 'cuda'
        model.eval()
        #model.to(device)
        labels = None
        embeddings = None
        for batch_idx, data in tqdm(enumerate(data_loader)):
            batch_imgs, batch_labels = data
            batch_labels = batch_labels.numpy()
            #batch_imgs = Variable(batch_imgs.to('cuda'))
            batch_E = model.get_embedding(batch_imgs)
            batch_E = batch_E.data.cpu().numpy()
            embeddings = np.concatenate((embeddings, batch_E), axis=0) if embeddings is not None else batch_E
            labels = np.concatenate((labels, batch_labels), axis=0) if labels is not None else batch_labels
    return embeddings, labels


def vis_tSNE(embeddings, labels, backbone='Convnet'):
    num_samples = embeddings.shape[0]
    X_embedded = TSNE(n_components=2).fit_transform(embeddings[0:num_samples, :])
    plt.figure(figsize=(16, 16))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#1fa7b4', '#fb7f0e', '#27a02c', '#da2758', '#a46abd',
              '#af7bb4', '#fa7fbe', '#2baf2c', '#4f2d28', '#b4f7bd']

    labels_name = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
                   'Potato___healthy', 'Potato___Late_blight', 'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus',
                   'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                   'Tomato_healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                   'Tomato_Spider_mites_Two_spotted_spider_mite']
    for i in range(16):
        inds = np.where(labels == i)[0]
        plt.scatter(X_embedded[inds, 0], X_embedded[inds, 1], alpha=.8, color=colors[i], s=200)
    # plt.title(f't-SNE', fontweight='bold', fontsize=24)
    plt.legend(labels_name, fontsize=30)
    plt.savefig(f'./tsne_{backbone}.png')


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

classifier = KNeighborsClassifier(n_neighbors=1)
# classifier = SVC()
# classifier = SGDClassifier()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

accuracy = round(accuracy_score(y_true, y_pred) * 100, 2)
precision = round(precision_score(y_true, y_pred, average='macro') * 100, 2)
recall = round(recall_score(y_true, y_pred, average='macro') * 100, 2)
f1 = round(f1_score(y_true, y_pred, average='macro') * 100, 2)
print(f'--- Results for MobileNetv2 Embeddings on KNN (k = 1) ---')
print(f'Accuracy Score:{accuracy}')
print(f'Precision Score: {precision}')
print(f'Recall Score: {recall}')
print(f'F1 Score: {f1}')

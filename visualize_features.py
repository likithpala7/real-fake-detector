import argparse
import os
import pdb

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from modules.models.clip_model import CLIPLORA, CLIPDataset, CLIPTeacher2, CLIPTeacher, CLIPTeacherDataset, CLIPLarge
from modules.models.vit_model import ViTModel
from modules.models.resnet_model import ResNet, ResNetDataset
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_dict = {i: model for i, model in
              enumerate(sorted(os.listdir('imagenet_ai_holdout')))}


def params():
    parser = argparse.ArgumentParser()
    most_recent_checkpoint = sorted(os.listdir('results'))[-1]
    parser.add_argument('--model', type=str, default='CLIP',
                        choices=['ResNet', 'CLIP', 'ViT', 'TransformerTeacher', 'TransformerTeacher2', 'CLIP-LORA'])
    parser.add_argument('--checkpoint-file', type=str, default=most_recent_checkpoint)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    return parser.parse_args()


def extract_features():
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():

        for idx, (inputs, labels) in tqdm(enumerate(val_dloader), total=len(val_dloader)):

            with torch.no_grad():
                features = model.get_features(inputs)

            for feature, label in zip(features, labels):
                features_list.append(feature.cpu().detach().numpy())
                labels_list.append(label.cpu().detach().numpy())

    features = np.array(features_list)
    labels = np.array(labels_list)

    tsne = TSNE(n_components=2, random_state=0)
    embeddings = tsne.fit_transform(features)

    plt.figure(figsize=(10, 10))

    for i in np.unique(labels):
        plt.scatter(embeddings[labels == i, 0], embeddings[labels == i, 1], label=label_dict[i])

    title = f't-SNE Visualization of Features from {config.model}'
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.savefig(os.path.join('results', config.checkpoint_file, 'tsne.png'))


if __name__ == '__main__':

    config = params()

    if config.model == 'ResNet':

        # Load processors and models
        model = ResNet()

        val_dset = ResNetDataset(
            d_type='val',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
            ])
        )
    elif config.model == 'ViT':
        model = ViTModel()

        val_dset = ResNetDataset(
            d_type='val',
            transform=
            transforms.Compose([transforms.ToTensor(), transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
        )
    elif config.model == 'TransformerTeacher':
        model = CLIPTeacher(config)
        val_dset = CLIPTeacherDataset(d_type='val')
    elif config.model == 'TransformerTeacher2':
        model = CLIPTeacher2()
        val_dset = CLIPTeacherDataset(d_type='val')
    elif config.model == 'FCNTeacher':
        model = CLIPTeacher2()
        val_dset = CLIPTeacherDataset(d_type='val')
    elif config.model == 'CLIP':
        model = CLIPLarge(num_classes=len(label_dict))
        val_dset = CLIPDataset(d_type='val')
    else:
        model = CLIPLORA(model_name='laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
        val_dset = CLIPDataset(d_type='val')

    # model = torch.load(os.path.join('results', config.checkpoint_file, 'latest_model.pth'))
    model.to(device)
    model.load_state_dict(torch.load(os.path.join('results', config.checkpoint_file,
                                                  'latest_model.pth')))
    val_dloader = DataLoader(val_dset, batch_size=config.batch_size, shuffle=True)

    extract_features()

"""
Docstring for trial;

This file is for trial and test-out components built for the project, no use in actual training or testing;
"""

"""
Creating CSV file for custom data-loaders;
"""
# import os
# import csv

# data_path = "data"

# with open("data_csv.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(['index','image_path','image_caption','image_label'])

#     dataset_directory = os.listdir(data_path)
#     index = 0
#     for directory in dataset_directory:
#         data_label = directory
#         images_directory = os.path.join(data_path, directory, "different")
#         for image in os.listdir(images_directory):
#             image_path = os.path.join(images_directory,image)
#             image_caption = image.split('.')[0].replace('_',' ')
#             writer.writerow([index,image_path, image_caption, data_label])
#             index += 1
#         images_directory = os.path.join(data_path, directory, "image")
#         for image in os.listdir(images_directory):
#             image_path = os.path.join(images_directory,image)
#             image_caption = image.split('.')[0].replace('_',' ')

#             writer.writerow([index,image_path, image_caption, data_label])
#             index += 1
#         images_directory = os.path.join(data_path, directory, "same")
#         for image in os.listdir(images_directory):
#             image_path = os.path.join(images_directory,image)
#             image_caption = image.split('.')[0].replace('_',' ')

#             writer.writerow([index,image_path, image_caption, data_label])
#             index += 1


"""
Plotting samples from dataset;
"""
# import torch
# import matplotlib.pyplot as plt
# from data_loader import MisogynyDataLoader

# dataloaders = MisogynyDataLoader(batch_size=4)
# train_loader = dataloaders.train_loader

# images, captions, labels = next(iter(train_loader))

# def show_images(images, captions, labels):
#     batch_size = images.size(0)
#     plt.figure(figsize=(12, 4))
#     for i in range(batch_size):
#         img = images[i]
#         img = img.permute(1, 2, 0)
#         img = torch.clamp(img, 0, 1)
#         plt.subplot(1, batch_size, i + 1)
#         plt.imshow(img)
#         plt.axis("off")
#         plt.title(f"Label: {labels[i]}")

#     plt.tight_layout()
#     plt.show()


# show_images(images, captions, labels)

"""
Code snippet to try-out final Graph with LDA reduced embeddings;
Lolz might not be much usefull in future;
"""
# from models import BERTEmbedder, OpenClipVitEmbedder, LDALayer, GraphModule
# from data_loader import MisogynyDataLoader
# import numpy as np
# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"

# data_loader = MisogynyDataLoader()
# train_loader = data_loader.train_loader
# test_loader = data_loader.test_loader

# text_embedder = BERTEmbedder().to(device)
# image_embedder = OpenClipVitEmbedder(device=device)

# lda_mean = np.load("weights/combined_lda_mean.npy")
# lda_coef = np.load("weights/combined_lda_coef.npy")
# lda_layer = LDALayer(lda_mean, lda_coef)

# graph_module = GraphModule.load(path="weights/graph_module.pth", device=device)

# sample_images, sample_captions, sample_labels = [], [], []

# for i, (images, captions, labels) in enumerate(train_loader):
#     sample_images.append(images)
#     sample_captions.extend(captions)
#     sample_labels.extend(labels)
#     if i >= 0:
#         break

# sample_images = torch.cat(sample_images, dim=0).to(device)

# with torch.no_grad():
#     text_emb = text_embedder(sample_captions).to("cpu")

# with torch.no_grad():
#     image_emb = image_embedder(sample_images).to("cpu")

# combined_emb = np.concatenate([text_emb.numpy(), image_emb.numpy()], axis=1)
# combined_tensor = torch.tensor(combined_emb, dtype=torch.float32)
# lda_emb = lda_layer(combined_tensor)

# contextualized_embeddings = graph_module(lda_emb, k=5)

# print("Number of samples:", contextualized_embeddings.shape[0])
# print("Contextualized embedding dimension:", contextualized_embeddings.shape[1])
# for i in range(min(5, contextualized_embeddings.shape[0])):
#     print(f"Sample {i} embedding:", contextualized_embeddings[i].detach().numpy())


"""
Code snippet to try-out final Misogyny model;
Ig we never know it will work or not;
"""

# import torch
# from models import MisogynyModel
# from data_loader import MisogynyDataLoader

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data_loader = MisogynyDataLoader()
# sample_loader = data_loader.test_loader

# misogyny_model = MisogynyModel(device=device).to(device)
# misogyny_model.eval()

# images, captions, labels = next(iter(sample_loader))
# images = images.to(device)
# labels = labels.to(device)

# with torch.no_grad():
#     logits = misogyny_model(captions, images)
#     predicted_class = torch.argmax(logits, dim=1)

# print("Predicted label:", predicted_class.tolist())
# print("Ground truth label:", labels.tolist())

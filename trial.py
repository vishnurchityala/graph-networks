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
#         plt.title(f"Label: {labels[i]} Caption:{captions[i]}")

#     plt.tight_layout()
#     plt.show()


# show_images(images, captions, labels)
"""
Loads a trained misogyny classification model and evaluates it on the test dataset.
Computes and prints test loss and overall accuracy.
"""
# import torch
# from torch import nn
# from data_loader import MisogynyDataLoader
# from models import MisogynyModel

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data_loader = MisogynyDataLoader(batch_size=16)
# test_loader = data_loader.test_loader

# model = MisogynyModel(device=device).to(device)

# checkpoint_path = "saved_models/misogyny_model_81.pth"
# model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# model.eval()

# criterion = nn.CrossEntropyLoss()

# test_loss = 0.0
# correct = 0
# total = 0

# with torch.no_grad():
#     for images, captions, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)

#         logits = model(captions, images)
#         loss = criterion(logits, labels)

#         test_loss += loss.item() * images.size(0)
#         _, predicted = torch.max(logits, 1)

#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# avg_test_loss = test_loss / total
# test_accuracy = correct / total

# print(f"Test Loss: {avg_test_loss:.4f}")
# print(f"Test Accuracy: {test_accuracy:.4f}")

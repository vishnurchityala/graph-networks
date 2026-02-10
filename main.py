"""
Docstring for main

This document to put to-gether all components of models created and test out their functioning;
Lolz not getting much good results; kms;

"""
import torch
from torch import nn, optim
from data_loader import MisogynyDataLoader
from models import MisogynyModel
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_loader = MisogynyDataLoader(batch_size=16)
train_loader = data_loader.train_loader
test_loader = data_loader.test_loader

misogyny_model = MisogynyModel(device=device).to(device)
print(misogyny_model.eval())
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(misogyny_model.parameters(), lr=1e-4)

num_epochs = 20
log_file = "training_logs.txt"

with open(log_file, "w") as f:
    f.write("Epoch,Loss,Accuracy\n")

for epoch in range(num_epochs):
    misogyny_model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, captions, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = misogyny_model(captions, images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    log_str = f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
    print(log_str)
    with open(log_file, "a") as f:
        f.write(f"{epoch+1},{epoch_loss:.4f},{epoch_acc:.4f}\n")

misogyny_model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, captions, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = misogyny_model(captions, images)
        loss = criterion(logits, labels)

        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

avg_test_loss = test_loss / total
test_accuracy = correct / total

print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
os.makedirs("saved_models", exist_ok=True)
save_path = f"saved_models/misogyny_model_{int(test_accuracy*100)}.pth"
torch.save(misogyny_model.state_dict(), save_path)
print(f"Model saved to {save_path}")
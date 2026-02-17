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

"""
Loading the best trained model for GraphModel, LDAModel, PCAOnlyModel
"""
import torch
from data_loader import MisogynyDataLoader
from models import MisogynyModel, MisogynyModelNoGraph, MisogynyModelPCAOnly
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test dataset
data_loader = MisogynyDataLoader()
test_loader = data_loader.test_loader
train_loader = data_loader.train_loader

# Models
MODEL_CONFIGS = {
    "graph":MisogynyModel(),
    "no_graph":MisogynyModelNoGraph(),
    "pca_only": MisogynyModelPCAOnly()
}


MODEL_CONFIGS['graph'].load_state_dict(
    torch.load("saved_models/graph_20260217_1150_BEST_ep22_acc0.955_f10.925.pth",weights_only=False)["model_state_dict"]
)

MODEL_CONFIGS['no_graph'].load_state_dict(
    torch.load("saved_models/no_graph_20260217_1208_BEST_ep18_acc0.906_f10.814.pth",weights_only=False)["model_state_dict"]
)

MODEL_CONFIGS['pca_only'].load_state_dict(
    torch.load("saved_models/pca_only_20260217_1217_BEST_ep10_acc0.951_f10.910.pth",weights_only=False)["model_state_dict"]
)

def evaluate_model(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:

            images, captions, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            logits = model(captions, images)

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }


for name, model in MODEL_CONFIGS.items():

    print(f"\n===== {name.upper()} MODEL =====")

    train_metrics = evaluate_model(model, train_loader)
    test_metrics = evaluate_model(model, test_loader)

    print("TRAIN METRICS")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nTEST METRICS")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
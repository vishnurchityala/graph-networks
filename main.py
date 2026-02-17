"""
Unified Trainer

Trains:
    1) Graph model
    2) No-graph model
    3) PCA-only model

Saves only:
    - best checkpoint (highest validation F1)
    - final checkpoint (last epoch)

Nothing is saved during training epochs.
"""

import os
import random
import datetime
import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import f1_score, precision_score, recall_score

from data_loader import MisogynyDataLoader
from models import MisogynyModel, MisogynyModelNoGraph, MisogynyModelPCAOnly

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_loader = MisogynyDataLoader(batch_size=16)
train_loader = data_loader.train_loader
test_loader = data_loader.test_loader

MODEL_CONFIGS = {
    "graph": {
        "class": MisogynyModel,
        "epochs": 25
    },
    "no_graph": {
        "class": MisogynyModelNoGraph,
        "epochs": 20
    },
    "pca_only": {
        "class": MisogynyModelPCAOnly,
        "epochs": 10
    },
}

def compute_metrics(y_true, y_pred):
    acc = (y_true == y_pred).mean()
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return acc, precision, recall, f1_macro, f1_weighted

def evaluate(model, loader, criterion):
    model.eval()
    loss_sum = 0
    preds, targets = [], []

    with torch.no_grad():
        for images, captions, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(captions, images)
            loss = criterion(logits, labels)

            loss_sum += loss.item() * images.size(0)

            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    avg_loss = loss_sum / len(targets)
    acc, prec, rec, f1m, f1w = compute_metrics(targets, preds)

    return avg_loss, acc, prec, rec, f1m, f1w

def train_model(model_name, ModelClass, epochs=30):

    print(f"\n========== Training {model_name.upper()} ==========")

    model = ModelClass(device=device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    use_amp = torch.cuda.is_available()
    amp_device = "cuda" if use_amp else "cpu"

    scaler = torch.amp.GradScaler(device=amp_device, enabled=use_amp)

    os.makedirs("saved_models", exist_ok=True)

    log_file = f"logs_{model_name}.csv"
    with open(log_file, "w") as f:
        f.write(
            "epoch,"
            "train_loss,train_acc,train_prec,train_rec,train_f1_macro,train_f1_weighted,"
            "val_loss,val_acc,val_prec,val_rec,val_f1_macro,val_f1_weighted\n"
        )

    best_f1 = 0.0
    best_state_dict = None
    best_epoch = 0
    best_acc = 0
    history = []
    for epoch in range(1, epochs + 1):

        model.train()
        train_loss = 0
        preds, targets = [], []

        for images, captions, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                logits = model(captions, images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.detach().cpu().numpy())
            targets.extend(labels.detach().cpu().numpy())

        preds = np.array(preds)
        targets = np.array(targets)
        train_loss /= len(targets)

        train_acc, train_prec, train_rec, train_f1m, train_f1w = compute_metrics(targets, preds)

        val_loss, acc, prec, rec, f1m, f1w = evaluate(model, test_loader, criterion)

        with open(log_file, "a") as f:
            f.write(
                    f"{epoch},"
                    f"{train_loss:.4f},{train_acc:.4f},{train_prec:.4f},{train_rec:.4f},{train_f1m:.4f},{train_f1w:.4f},"
                    f"{val_loss:.4f},{acc:.4f},{prec:.4f},{rec:.4f},{f1m:.4f},{f1w:.4f}\n"
                )

        print(
            f"{model_name} | Epoch {epoch}/{epochs} "
            f"| TL {train_loss:.3f} "
            f"| TA {train_acc:.3f} "
            f"| VF1 {f1m:.3f}"
        )


        if f1m > best_f1:
            best_f1 = f1m
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_acc = acc
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_prec": train_prec,
            "train_rec": train_rec,
            "train_f1_macro": train_f1m,
            "train_f1_weighted": train_f1w,
            "val_loss": val_loss,
            "val_acc": acc,
            "val_prec": prec,
            "val_rec": rec,
            "val_f1_macro": f1m,
            "val_f1_weighted": f1w
        })


    date = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    best_name = (
        f"{model_name}_{date}"
        f"_BEST"
        f"_ep{best_epoch}"
        f"_acc{best_acc:.3f}"
        f"_f1{best_f1:.3f}.pth"
    )

    best_path = os.path.join("saved_models", best_name)

    torch.save({
        "model_name": model_name,
        "epoch": best_epoch,
        "model_state_dict": best_state_dict,
        "best_f1": best_f1,
        "best_accuracy": best_acc,
        "history": history
    }, best_path)


    print("Saved best ->", best_path)

    final_name = (
        f"{model_name}_{date}"
        f"_FINAL"
        f"_ep{epochs}.pth"
    )

    final_path = os.path.join("saved_models", final_name)

    torch.save({
        "model_name": model_name,
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
    }, final_path)

    print("Saved final ->", final_path)
    print(f"Best F1 for {model_name}: {best_f1:.4f}")


for name, cfg in MODEL_CONFIGS.items():
    train_model(
        model_name=name,
        ModelClass=cfg["class"],
        epochs=cfg["epochs"]
    )

print("\nAll models trained.")

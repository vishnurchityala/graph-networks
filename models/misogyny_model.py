import torch
import numpy as np
from torch import nn

from .bert_embedder import BERTEmbedder
from .clip_embedder import OpenClipVitEmbedder
from .pca_layer import PCALayer
from .lda_layer import LDALayer
from .graph_layer import GraphModule
from .classification_layer import ClassificationLayer


class MisogynyModel(nn.Module):
    def __init__(self,
                 lda_weights_path=("weights/combined_lda_mean.npy", "weights/combined_lda_coef.npy"),
                 pca_weights_path_text=("weights/bert_pca_components_50.npy", "weights/bert_pca_mean_50.npy"),
                 pca_weights_path_image=("weights/clip_pca_components_50.npy", "weights/clip_pca_mean_50.npy"),
                 graph_weights_path="weights/graph_module.pth",
                 device=None,
                 freeze_non_trainable=True):
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.text_embedder = BERTEmbedder().to(self.device)
        self.image_embedder = OpenClipVitEmbedder().to(self.device)

        lda_mean = np.load(lda_weights_path[0])
        lda_coef = np.load(lda_weights_path[1])
        self.lda_layer = LDALayer(lda_mean, lda_coef).to(self.device)

        text_components = np.load(pca_weights_path_text[0])
        text_mean = np.load(pca_weights_path_text[1])
        self.text_pca_layer = PCALayer(
            mean=text_mean,
            components=text_components
        ).to(self.device)

        image_components = np.load(pca_weights_path_image[0])
        image_mean = np.load(pca_weights_path_image[1])
        self.image_pca_layer = PCALayer(
            mean=image_mean,
            components=image_components
        ).to(self.device)

        self.graph_module = GraphModule.load(
            graph_weights_path,
            device=self.device
        ).to(self.device)

        graph_out_dim = self.graph_module.gat2.out_channels
        input_dim = text_components.shape[0] + image_components.shape[0] + graph_out_dim

        self.classification_layer = ClassificationLayer(
            input_dim=input_dim,
            out_dim=4
        ).to(self.device)

        if freeze_non_trainable:
            for module in [
                self.text_embedder,
                self.image_embedder,
                self.lda_layer,
                self.text_pca_layer,
                self.image_pca_layer,
                self.graph_module,
            ]:
                for param in module.parameters():
                    param.requires_grad = False

            for param in self.graph_module.gat1.parameters():
                param.requires_grad = True
            for param in self.graph_module.gat2.parameters():
                param.requires_grad = True
            for param in self.classification_layer.parameters():
                param.requires_grad = True


    def forward(self, text_inputs, image_inputs):
        text_features = self.text_embedder(text_inputs)
        image_features = self.image_embedder(image_inputs)

        combined_embed = torch.cat([text_features, image_features], dim=1)
        lda_features = self.lda_layer(combined_embed)
        lda_graph_features = self.graph_module(lda_features)

        text_pca_features = self.text_pca_layer(text_features)
        image_pca_features = self.image_pca_layer(image_features)

        combined_features = torch.cat([text_pca_features, image_pca_features, lda_graph_features], dim=1)

        logits = self.classification_layer(combined_features)

        return logits

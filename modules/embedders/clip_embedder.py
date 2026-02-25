import torch
import torch.nn.functional as F
import open_clip


class OpenClipVitEmbedder(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-32",
            pretrained="openai"
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, image_tensor):
        image_tensor = image_tensor.to(self.device)
        image_features = self.model.encode_image(image_tensor)
        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features
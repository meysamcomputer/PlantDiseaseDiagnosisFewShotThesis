import torch
from torch import nn
from torchvision.models import mobilenet_v2
from vit_pytorch import ViT

class CustomViT(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(CustomViT, self).__init__()
        # Load the pretrained MobileNetV2
        self.mobilenet_backbone = mobilenet_v2(pretrained=pretrained).features
        
        # Replace the classifier with a Vision Transformer
        self.vit = ViT(
            image_size = 224,
            patch_size = 32,
            num_classes = num_classes,
            dim = 512,
            depth = 6,
            heads = 8,
            mlp_dim = 1024,
            channels = 3,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        
    def forward(self, x):
        # Use the MobileNet backbone to extract features
        x = self.mobilenet_backbone(x)
        
        # Flatten the features and pass through the transformer
        x = x.flatten(2).transpose(1, 2)
        x = self.vit(x)
        
        return x

# Define the number of classes in your dataset
#num_classes = 10  # Example: 10 classes

# Create the model
#model = CustomViT(num_classes=num_classes)

# Example input tensor
#input_tensor = torch.randn(1, 3, 224, 224)

# Forward pass
#predictions = model(input_tensor)

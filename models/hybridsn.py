import torch.nn as nn
import torch


class HybridSN(nn.Module):
    def __init__(self, num_classes, n_bands, patch_size):
        super(HybridSN, self).__init__()

        # Calculate spectral dimension after 3D convolutions
        # kernel_sizes: 7, 5, 3 -> total reduction = (7-1) + (5-1) + (3-1) = 12
        self.spectral_dim = n_bands - 12

        # 3D Convolution Block (spectral-spatial feature extraction)
        self.block_1_3D = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.ReLU(inplace=True)
        )

        # 2D Convolution Block (spatial feature refinement)
        self.block_2_2D = nn.Sequential(
            nn.Conv2d(in_channels=32 * self.spectral_dim, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

        # Fully Connected Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64 * patch_size * patch_size, out_features=256),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=256, out_features=128),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        y = self.block_1_3D(x)
        y = y.view(-1, y.shape[1] * y.shape[2], y.shape[3], y.shape[4])
        y = self.block_2_2D(y)
        y = y.view(y.size(0), -1)
        y = self.classifier(y)
        return y


def hybridsn(dataset, patch_size):
    if dataset == 'ip':
        return HybridSN(num_classes=16, n_bands=200, patch_size=patch_size)
    elif dataset == 'sa':
        return HybridSN(num_classes=16, n_bands=204, patch_size=patch_size)
    else:
        raise ValueError(f"Dataset '{dataset}' is not supported. Only 'ip' and 'sa' are supported.")
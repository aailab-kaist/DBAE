import torch.nn as nn

try:
    from model.module import AttentionBlock, normalization, View
except ModuleNotFoundError:
    import sys, os
    sys.path.append(os.path.dirname(__file__) + "/../../../")
    from model.module import AttentionBlock, normalization, View

from model.representation.encoder.resnet import resnet10

class CELEBA64Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.latent_dim = kwargs["latent_dim"]

        self.encoder = nn.Sequential(

            nn.Conv2d(3, 64, (3, 3), (2, 2), 1),          # batch_size x 64 x 32 x 32

            normalization(64),
            nn.SiLU(True),
            nn.Conv2d(64, 128, (3, 3), (2, 2), 1),          # batch_size x 128 x 16 x 16

            AttentionBlock(128, 4, -1, False),

            normalization(128),
            nn.SiLU(True),
            nn.Conv2d(128, 128, (3, 3), (2, 2), 1),          # batch_size x 128 x 8 x 8

            normalization(128),
            nn.SiLU(True),
            nn.Conv2d(128, 128, (3, 3), (2, 2), 1),          # batch_size x 128 x 4 x 4

            normalization(128),
            nn.SiLU(True),
            View((-1, 128 * 4 * 4)),                  # batch_size x 2048
            nn.Linear(2048, self.latent_dim)
        )

    # x: batch_size x 3 x 64 x 64
    def forward(self, x):
        # batch_size x latent_dim
        return self.encoder(x)

#class CELEBA64ResNet10(nn.Module):
#    def __init__(self, **kwargs):
#        resnet10_kwargs = dict(
#            block="basic",
#            layers=[1, 1, 1, 1],
#            block_inplanes=[64, 128, 256, 512],
#            spatial_dims=2,
#            feed_forward=(not kwargs["latent_dim"]==512),  # if latent_dim==512, delete the fc layer
#            num_classes=kwargs["latent_dim"],  # if latent_dim!=512, use the fc layer mapping to correct dimension
#        )
#        super().__init__(**resnet10_kwargs)


class CELEBA64ResNet10(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.latent_dim = kwargs["latent_dim"]
        self.encoder = resnet10()

    def forward(self, x):
        return self.encoder(x)

if __name__ == '__main__':
    kwargs = {"latent_dim":512}
    # calculate #params
    encoder = CELEBA64Encoder(**kwargs)
    # encoder = CELEBA64ResNet10(**kwargs)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(n_params)

    # summary (# params)
    # CELEBA64Encoder:  1,487,104
    # CELEBA64ResNet10: 4,906,688
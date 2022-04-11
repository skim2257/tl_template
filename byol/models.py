from torch import Tensor
from torch import nn
from models.resnet import resnet18, resnet34

class Identity(nn.Module):
    """An identity class to replace arbitrary layers in pretrained models.
    Example::
        from pl_bolts.utils import Identity
        model = resnet18()
        model.fc = Identity()
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x
        
def torchvision_ssl_encoder(name: str) -> nn.Module:
    if name == 'resnet18':
        resnet_model = resnet18
    elif name == 'resnet34':
        resnet_model = resnet34
    else:
        raise NotImplementedError
    resnet_model.linear = Identity()
    return resnet_model

class MLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_size=4096, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class SiameseArm(nn.Module):
    def __init__(self, encoder="resnet18", encoder_out_dim=2048, projector_hidden_size=4096, projector_out_dim=256):
        super().__init__()

        if isinstance(encoder, str):
            encoder = torchvision_ssl_encoder(encoder)
        # Encoder
        self.encoder = encoder
        # Projector
        self.projector = MLP(encoder_out_dim, projector_hidden_size, projector_out_dim)
        # Predictor
        self.predictor = MLP(projector_out_dim, projector_hidden_size, projector_out_dim)

    def forward(self, x):
        y = self.encoder(x)[0]
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h
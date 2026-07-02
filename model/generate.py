"""Cat image generation using trained DCGAN generator"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
from io import BytesIO

GENERATOR_PATH = Path(__file__).parent / "generator.pth"
LATENT_DIM = 100
OUTPUT_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GeneratorV2(nn.Module):
    """128x128 generator (LeakyReLU, 6 layers)"""
    def __init__(self, latent_dim=LATENT_DIM, img_channels=3, fm=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, fm * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fm * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(fm * 8, fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(fm * 4, fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(fm * 2, fm, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(fm, fm // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(fm // 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


class GeneratorV1(nn.Module):
    """64x64 original generator (ReLU, 5 layers)"""
    def __init__(self, latent_dim=LATENT_DIM, img_channels=3, fm=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, fm * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fm * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(fm * 8, fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(fm * 4, fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(fm * 2, fm, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm),
            nn.ReLU(True),
            nn.ConvTranspose2d(fm, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


GENERATOR_CLASSES = [GeneratorV2, GeneratorV1]

_generator = None


def load_generator():
    global _generator
    if _generator is not None:
        return _generator
    if not GENERATOR_PATH.exists():
        return None

    state = torch.load(GENERATOR_PATH, map_location=DEVICE, weights_only=True)

    for cls in GENERATOR_CLASSES:
        try:
            netG = cls().to(DEVICE)
            netG.load_state_dict(state, strict=True)
            netG.eval()
            _generator = netG
            return _generator
        except Exception:
            continue

    return None


def generate_cat(seed=None, upscale_to=OUTPUT_SIZE):
    netG = load_generator()
    if netG is None:
        return None
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn(1, LATENT_DIM, 1, 1, device=DEVICE)
    with torch.no_grad():
        fake = netG(noise).detach().cpu()[0]
    img_tensor = fake.mul(0.5).add(0.5).clamp(0, 1)
    img_pil = transforms.ToPILImage()(img_tensor)
    if upscale_to and upscale_to > img_pil.width:
        img_pil = img_pil.resize((upscale_to, upscale_to), Image.Resampling.LANCZOS)
    return img_pil


def generate_cat_bytes(seed=None, upscale_to=OUTPUT_SIZE):
    img = generate_cat(seed=seed, upscale_to=upscale_to)
    if img is None:
        return None
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return buf.read()

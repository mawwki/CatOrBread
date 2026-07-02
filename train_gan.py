"""Train DCGAN to generate cat images"""
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

DATA_DIR = Path(__file__).parent / "data" / "cat"
MODEL_DIR = Path(__file__).parent / "model"
GENERATOR_PATH = MODEL_DIR / "generator.pth"
os.makedirs(MODEL_DIR, exist_ok=True)

LATENT_DIM = 100
IMG_SIZE = 128
IMG_CHANNELS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, img_channels=IMG_CHANNELS, fm=64):
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


class Discriminator(nn.Module):
    def __init__(self, img_channels=IMG_CHANNELS, fm=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(img_channels, fm, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fm, fm * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fm * 2, fm * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fm * 4, fm * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fm * 8, fm * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fm * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fm * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class CatDataset(Dataset):
    def __init__(self, root_dir, img_size=IMG_SIZE, transform=None):
        self.paths = sorted(Path(root_dir).glob("*.*"))
        self.transform = transform or transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def add_noise(tensor, sigma=0.05):
    return tensor + torch.randn_like(tensor) * sigma


def train(args):
    print(f"Device: {DEVICE}")

    netG = Generator().to(DEVICE)
    netD = Discriminator().to(DEVICE)
    netG.apply(weights_init)
    netD.apply(weights_init)

    if args.resume and GENERATOR_PATH.exists():
        state = torch.load(GENERATOR_PATH, map_location=DEVICE, weights_only=True)
        own_state = netG.state_dict()
        filtered = {k: v for k, v in state.items() if k in own_state and v.shape == own_state[k].shape}
        if filtered:
            netG.load_state_dict(filtered, strict=False)
            print(f"Resumed generator from {GENERATOR_PATH} ({len(filtered)}/{len(own_state)} layers)")
        else:
            print("Existing generator incompatible, starting fresh")

    dataset = CatDataset(DATA_DIR, img_size=IMG_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    print(f"Cat images: {len(dataset)}, Batches/epoch: {len(loader)}")

    criterion = nn.BCELoss()
    lr = args.lr
    beta1 = 0.5
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)
    real_label = 0.9
    fake_label = 0.1

    print("Training DCGAN (improved)...")
    for epoch in range(args.epochs):
        for i, real_imgs in enumerate(loader):
            current_batch = real_imgs.size(0)
            real_imgs = real_imgs.to(DEVICE)

            netD.zero_grad()
            label = torch.full((current_batch,), real_label, device=DEVICE)
            output = netD(add_noise(real_imgs, args.noise_sigma))
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(current_batch, LATENT_DIM, 1, 1, device=DEVICE)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(add_noise(fake.detach(), args.noise_sigma))
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake

            torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(add_noise(fake, args.noise_sigma * 0.5))
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()

            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            optimizerG.step()

            if i % 50 == 0:
                print(
                    f"[{epoch+1}/{args.epochs}][{i}/{len(loader)}] "
                    f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                    f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}"
                )
                _save_samples(netG, fixed_noise, epoch, i)

            if i % 200 == 0:
                _save_samples(netG, fixed_noise, epoch, i)

        torch.save(netG.state_dict(), GENERATOR_PATH)
        print(f"Epoch {epoch+1} done — generator saved.")

    print("Training complete!")


def _save_samples(netG, noise, epoch, batch_idx):
    samples_dir = MODEL_DIR / "gan_samples"
    samples_dir.mkdir(exist_ok=True)
    netG.eval()
    with torch.no_grad():
        fake = netG(noise).detach().cpu()
    netG.train()
    grid = _make_grid(fake, nrow=8)
    grid = grid.mul(0.5).add(0.5).clamp(0, 1)
    grid_pil = transforms.ToPILImage()(grid)
    path = samples_dir / f"epoch{epoch+1:03d}_batch{batch_idx:04d}.jpg"
    grid_pil.save(str(path), quality=92)


def _make_grid(tensor, nrow=8):
    n, c, h, w = tensor.shape
    nrow = min(nrow, n)
    ncol = (n + nrow - 1) // nrow
    grid = torch.zeros(c, h * ncol, w * nrow)
    for idx in range(n):
        r = idx // nrow
        c_idx = idx % nrow
        grid[:, r*h:(r+1)*h, c_idx*w:(c_idx+1)*w] = tensor[idx]
    return grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--noise-sigma", type=float, default=0.05,
                        help="Gaussian noise sigma for discriminator input (instance noise)")
    args = parser.parse_args()
    train(args)

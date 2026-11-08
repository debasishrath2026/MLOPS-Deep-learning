import hydra
import lightning as L
import timm
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class Classifier(L.LightningModule):
    def __init__(self, name: str, num_classes: int, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(name, pretrained=False, num_classes=num_classes)
        self.train_acc = MulticlassAccuracy(num_classes)
        self.val_acc = MulticlassAccuracy(num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.val_acc(logits, y)
        self.log("val/loss", F.cross_entropy(logits, y), prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


def make_loaders(cfg: DictConfig, seed: int):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_full = datasets.CIFAR10(cfg.dir, train=True, download=True, transform=train_tf)
    val_full = datasets.CIFAR10(cfg.dir, train=True, download=False, transform=val_tf)

    # Same split indices for both so validation images never get train augmentation.
    split = random_split(range(len(train_full)), [45000, 5000], generator=torch.Generator().manual_seed(seed))
    train_set = torch.utils.data.Subset(train_full, split[0].indices)
    val_set = torch.utils.data.Subset(val_full, split[1].indices)

    kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_set, shuffle=True, persistent_workers=cfg.num_workers > 0, **kwargs),
        DataLoader(val_set, shuffle=False, **kwargs),
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    L.seed_everything(cfg.seed, workers=True)

    train_loader, val_loader = make_loaders(cfg.data, cfg.seed)
    model = Classifier(**cfg.model)

    trainer = L.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        logger=TensorBoardLogger(cfg.log_dir, name="cifar10"),
        callbacks=[ModelCheckpoint(monitor="val/acc", mode="max", save_top_k=1, filename="best")],
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()

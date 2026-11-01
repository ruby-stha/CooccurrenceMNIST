"""Per-cell digit classifier (oracle) for evaluating generated 1x2 co-occurrence images.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torchvision.datasets import MNIST

PAIRS: tuple[tuple[int, int], ...] = ((1, 8), (3, 6), (5, 0), (7, 4), (9, 2))
PARTNER: dict[int, int] = {d: p for a, b in PAIRS for d, p in ((a, b), (b, a))}
CELL = 32
CELL_PAD = 2


class CellCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def random_shift(x: Tensor, max_shift: int = 2) -> Tensor:
    # Generated digits are rarely centered exactly, so train with small per-sample shifts.
    shifts = torch.randint(-max_shift, max_shift + 1, (x.shape[0], 2), device=x.device)
    theta = torch.zeros(x.shape[0], 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1
    theta[:, :, 2] = shifts * 2 / x.shape[-1]
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)


def load_mnist(root: Path, train: bool, device: torch.device) -> tuple[Tensor, Tensor]:
    ds = MNIST(str(root), train=train, download=False)
    return ds.data.unsqueeze(1).float().div(255).to(device), ds.targets.to(device)


@torch.no_grad()
def accuracy(model: nn.Module, x: Tensor, y: Tensor, batch_size: int = 4096) -> float:
    model.eval()
    correct = sum(
        (model(x[i:i + batch_size]).argmax(1) == y[i:i + batch_size]).sum().item()
        for i in range(0, len(x), batch_size)
    )
    return correct / len(x)


def train(mnist_root: Path, split_file: Path, device: torch.device, epochs: int, seed: int) -> CellCNN:
    """Train on the train-image digits and keep the epoch with the best val-digit accuracy."""
    torch.manual_seed(seed)
    x_all, y_all = load_mnist(mnist_root, True, device)
    split = np.load(split_file)
    train_idx = torch.from_numpy(split["train_idx"]).to(device)
    val_idx = torch.from_numpy(split["val_idx"]).to(device)
    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]

    model = CellCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch_size = 256
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, total_steps=epochs * -(-len(x_train) // batch_size)
    )
    best_acc, best_epoch, best_state = -1.0, 0, {}
    for epoch in range(epochs):
        model.train()
        for batch in torch.randperm(len(x_train), device=device).split(batch_size):
            loss = F.cross_entropy(model(random_shift(x_train[batch])), y_train[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
        val_acc = accuracy(model, x_val, y_val)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch + 1}/{epochs}  val digit accuracy {val_acc:.4f}", flush=True)
    model.load_state_dict(best_state)
    print(f"kept epoch {best_epoch} (val digit accuracy {best_acc:.4f})")
    return model


def load_classifier(path: Path, device: torch.device) -> CellCNN:
    model = CellCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval()


def split_cells(images: Tensor) -> Tensor:
    """[N, 1, 32 * rows, 32 * cols] -> [N * rows * cols, 1, 28, 28], cells in reading order.

    Each 32x32 cell is center-cropped to 28x28, which removes exactly the padding
    added around the MNIST digit.
    """
    n, _, h, w = images.shape
    rows, cols = h // CELL, w // CELL
    cells = images.reshape(n, 1, rows, CELL, cols, CELL).permute(0, 2, 4, 1, 3, 5)
    cells = cells.reshape(n * rows * cols, 1, CELL, CELL)
    return cells[..., CELL_PAD:CELL - CELL_PAD, CELL_PAD:CELL - CELL_PAD]


@torch.no_grad()
def predict_cells(model: nn.Module, images: Tensor, batch_size: int = 1024) -> Tensor:
    """Return predicted digits per cell, shape [N, cells per image]."""
    model.eval()
    device = next(model.parameters()).device
    n_cells = (images.shape[-2] // CELL) * (images.shape[-1] // CELL)
    preds = [
        model(split_cells(images[i:i + batch_size].to(device))).argmax(1).cpu()
        for i in range(0, len(images), batch_size)
    ]
    return torch.cat(preds).reshape(-1, n_cells)


def image_metrics(pred: Tensor, cond: Tensor) -> dict[str, float]:
    """Compare predicted cell digits with the requested ones.

    partner_leak_rate: per requested digit whose partner was not requested, how
        often that partner shows up in some cell, e.g. (1, 3) rendered as (1, 8).
    majority_pair_rate: share of images recognized as a majority pair; for
        minority conditions this means the model drifted to a majority pair.
    """
    pred, cond = pred.cpu(), cond.cpu()
    leaks = leak_total = majority = 0
    for p_row, c_row in zip(pred.tolist(), cond.tolist()):
        majority += len(p_row) == 2 and PARTNER[p_row[0]] == p_row[1]
        for digit in c_row:
            if PARTNER[digit] not in c_row:
                leak_total += 1
                leaks += PARTNER[digit] in p_row
    return {
        "n": len(cond),
        "cell_accuracy": (pred == cond).float().mean().item(),
        "image_accuracy": (pred == cond).all(1).float().mean().item(),
        "partner_leak_rate": leaks / leak_total if leak_total else float("nan"),
        "majority_pair_rate": majority / len(cond),
    }


def load_images(root: Path, rows: list[dict]) -> tuple[Tensor, Tensor]:
    images = np.stack([np.asarray(Image.open(root / r["path"])) for r in rows])
    tensor = torch.from_numpy(images).unsqueeze(1).float().div(255)
    cond = torch.tensor([[int(r["d1"]), int(r["d2"])] for r in rows])
    return tensor, cond


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("/home/all_data/cooccurrence_1x2"))
    parser.add_argument("--mnist-root", type=Path, default=Path("/home/all_data/mnist_raw"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    model = train(args.mnist_root, args.root / "mnist_train_val_split.npz", device, args.epochs, args.seed)
    model_path = args.root / "cell_classifier.pt"
    torch.save(model.state_dict(), model_path)
    print(f"saved {model_path}")
    x_test, y_test = load_mnist(args.mnist_root, False, device)
    print(f"MNIST test digit accuracy {accuracy(model, x_test, y_test):.4f}")

    with (args.root / "splits" / "test.csv").open() as f:
        rows = list(csv.DictReader(f))
    images, cond = load_images(args.root, rows)
    pred = predict_cells(model, images)
    groups = {"maj": [], "min": []}
    for i, row in enumerate(rows):
        groups[row["group"]].append(i)

    report = [{"group": g, **image_metrics(pred[idx], cond[idx])} for g, idx in groups.items()]
    report.append({"group": "all", **image_metrics(pred, cond)})
    report_path = args.root / "classifier_report_real_test.csv"
    with report_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    print(f"\nreal test images ({report_path}):")
    for r in report:
        print(
            f"  {r['group']:<4} n={r['n']:>5}  cell acc={r['cell_accuracy']:.4f}  "
            f"image acc={r['image_accuracy']:.4f}  partner leak={r['partner_leak_rate']:.4f}  "
            f"majority pair={r['majority_pair_rate']:.4f}"
        )


if __name__ == "__main__":
    main()

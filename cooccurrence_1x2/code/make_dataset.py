"""Build the 1x2 MNIST co-occurrence dataset.

Each image is a 32x64 strip of two different MNIST digits (left, right), each
padded from 28x28 to 32x32. The five majority pairs (1, 8), (3, 6), (5, 0),
(7, 4), (9, 2) appear in both orders; every other pair of different digits is a
minority pair (40 unordered, 80 ordered).

The test set has 250 images per ordered pair.
"""

import argparse
import csv
import os
import shutil
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import MNIST

PAIRS: tuple[tuple[int, int], ...] = ((1, 8), (3, 6), (5, 0), (7, 4), (9, 2))
PARTNER: dict[int, int] = {d: p for a, b in PAIRS for d, p in ((a, b), (b, a))}

TOTALS: dict[str, int] = {"train": 55000, "val": 5000}
VAL_DIGITS = 5000
RATIOS: dict[str, float] = {"balanced": 1 / 9, "0.95": 0.95, "0.98": 0.98, "0.99": 0.99, "0.995": 0.995}
TEST_PER_ORDERED_PAIR = 250
N_TEST_SUBSETS = 8
# 28x28 digits padded to 32x32 give a 32x64 image, which halves cleanly in U-Net/DCGAN stages.
CELL_PAD = 2

SPLITS = ("train", "val", "test")
MNIST_SOURCE: dict[str, str] = {"train": "train", "val": "train", "test": "test"}
FIELDS = ["idx", "path", "split", "group", "pair", "pool_order", "d1", "d2", "src1", "src2", "subset"]

_images: dict[str, np.ndarray] = {}


@dataclass(frozen=True)
class PoolSpec:
    split: str
    group: str
    pair: tuple[int, int] | None
    size: int


def pair_name(a: int, b: int) -> str:
    """Folder/column name of an unordered pair: majority pairs keep their PAIRS order."""
    if PARTNER[a] == b:
        return next(f"{x}_{y}" for x, y in PAIRS if {x, y} == {a, b})
    return f"{min(a, b)}_{max(a, b)}"


def minority_matchings() -> list[list[tuple[int, int]]]:
    circle = "ABCDEFGHI"
    seat_to_digit = {"Z": 1, "A": 8, "B": 3, "C": 5, "D": 7, "E": 9, "F": 2, "G": 4, "H": 0, "I": 6}

    def round_robin(r: int) -> list[tuple[int, int]]:
        seat_pairs = [("Z", circle[r])] + [(circle[(r + i) % 9], circle[(r - i) % 9]) for i in range(1, 5)]
        return [(seat_to_digit[a], seat_to_digit[b]) for a, b in seat_pairs]

    assert {frozenset(e) for e in round_robin(0)} == {frozenset(p) for p in PAIRS}
    matchings = [round_robin(r) for r in range(1, 9)]
    assert len({frozenset(e) for m in matchings for e in m}) == 40
    return matchings


MATCHINGS = minority_matchings()
MATCHING_OF: dict[frozenset[int], int] = {frozenset(e): i for i, m in enumerate(MATCHINGS) for e in m}


def split_counts(cr: float, total: int) -> tuple[int, int]:
    """Return (images per majority pair, minority images) for correlation strength cr."""
    minority = 5 * round((1 - cr) * total / 5)
    return (total - minority) // len(PAIRS), minority


def majority_stream(pair: tuple[int, int], rng: np.random.Generator) -> Iterator[tuple[int, int]]:
    a, b = pair
    while True:
        first = (a, b) if rng.random() < 0.5 else (b, a)
        yield first
        yield first[::-1]


def minority_stream(rng: np.random.Generator) -> Iterator[tuple[int, int]]:
    while True:
        flips = {key: rng.random() < 0.5 for key in MATCHING_OF}
        for second_pass in (False, True):
            for m in rng.permutation(len(MATCHINGS)):
                group = list(MATCHINGS[m])
                rng.shuffle(group)
                for a, b in group:
                    yield (b, a) if flips[frozenset((a, b))] != second_pass else (a, b)


class InstanceSampler:
    """Hand out MNIST indices per digit, reshuffling each time a digit runs out."""

    def __init__(self, targets: np.ndarray, allowed: np.ndarray, rng: np.random.Generator) -> None:
        self._rng = rng
        self._by_digit = {d: allowed[targets[allowed] == d] for d in range(10)}
        self._queues: dict[int, list[int]] = {d: [] for d in range(10)}

    def draw(self, digit: int) -> int:
        queue = self._queues[digit]
        if not queue:
            queue.extend(self._rng.permutation(self._by_digit[digit]).tolist())
        return queue.pop()


def train_val_indices(targets: np.ndarray, n_val: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split of MNIST train indices; largest-remainder rounding gives exactly n_val."""
    counts = np.bincount(targets, minlength=10)
    quota = n_val * counts / counts.sum()
    per_digit = np.floor(quota).astype(int)
    per_digit[np.argsort(per_digit - quota)[: n_val - per_digit.sum()]] += 1

    rng = np.random.default_rng([seed, 10_000])
    val_idx = np.sort(np.concatenate([
        rng.permutation(np.flatnonzero(targets == d))[:k] for d, k in enumerate(per_digit)
    ]))
    return np.setdiff1d(np.arange(len(targets)), val_idx), val_idx


def pool_specs() -> list[PoolSpec]:
    specs = []
    for split, total in TOTALS.items():
        max_majority = max(split_counts(cr, total)[0] for cr in RATIOS.values())
        max_minority = max(split_counts(cr, total)[1] for cr in RATIOS.values())
        specs += [PoolSpec(split, "maj", pair, max_majority) for pair in PAIRS]
        specs.append(PoolSpec(split, "min", None, max_minority))
    specs += [PoolSpec("test", "maj", pair, 2 * TEST_PER_ORDERED_PAIR) for pair in PAIRS]
    specs.append(PoolSpec("test", "min", None, 80 * TEST_PER_ORDERED_PAIR))
    return specs


def build_rows(
    specs: list[PoolSpec], targets: dict[str, np.ndarray], allowed: dict[str, np.ndarray], seed: int
) -> list[dict]:
    rows = []
    for pool_id, spec in enumerate(specs):
        rng = np.random.default_rng([seed, pool_id])
        sampler = InstanceSampler(targets[MNIST_SOURCE[spec.split]], allowed[spec.split], rng)
        stream = majority_stream(spec.pair, rng) if spec.group == "maj" else minority_stream(rng)
        for order in range(spec.size):
            d1, d2 = next(stream)
            idx = len(rows)
            pair = pair_name(d1, d2)
            rows.append({
                "idx": idx,
                "path": f"{spec.split}/{spec.group}/{pair}/img_{idx:06d}_{d1}_{d2}.png",
                "split": spec.split,
                "group": spec.group,
                "pair": pair,
                "pool_order": order,
                "d1": d1,
                "d2": d2,
                "src1": sampler.draw(d1),
                "src2": sampler.draw(d2),
                "subset": "",
            })
    return rows


def assign_test_subsets(rows: list[dict]) -> None:
    by_layout: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        if row["split"] == "test" and row["group"] == "min":
            by_layout.setdefault((row["d1"], row["d2"]), []).append(row)
    per_subset = TEST_PER_ORDERED_PAIR // N_TEST_SUBSETS
    for (d1, d2), members in by_layout.items():
        m = MATCHING_OF[frozenset((d1, d2))]
        extras = [m % N_TEST_SUBSETS, (m + 1) % N_TEST_SUBSETS]
        for i, row in enumerate(members):
            row["subset"] = i % N_TEST_SUBSETS if i < per_subset * N_TEST_SUBSETS else extras[i - per_subset * N_TEST_SUBSETS]


def _write_image(task: tuple[str, str, tuple[int, int]]) -> None:
    path, source, sources = task
    cells = np.pad(_images[source][list(sources)], ((0, 0), (CELL_PAD, CELL_PAD), (CELL_PAD, CELL_PAD)))
    Image.fromarray(np.hstack(cells), mode="L").save(path)


def write_images(rows: list[dict], root: Path, workers: int) -> None:
    for folder in {Path(r["path"]).parent for r in rows}:
        (root / folder).mkdir(parents=True, exist_ok=True)
    tasks = [(str(root / r["path"]), MNIST_SOURCE[r["split"]], (r["src1"], r["src2"])) for r in rows]
    with Pool(workers) as pool:
        for _ in pool.imap_unordered(_write_image, tasks, chunksize=512):
            pass


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def select_split(rows: list[dict], split: str, per_pair: int, minority: int) -> list[dict]:
    limits = {"maj": per_pair, "min": minority}
    return [r for r in rows if r["split"] == split and r["pool_order"] < limits[r["group"]]]


def summarize(name: str, rows: list[dict]) -> list[str]:
    for row in rows:
        d1, d2 = row["d1"], row["d2"]
        if d1 == d2 or (PARTNER[d1] == d2) != (row["group"] == "maj"):
            raise ValueError(f"{row['path']} breaks the digit rules of group {row['group']}")

    maj = [r for r in rows if r["group"] == "maj"]
    mnr = [r for r in rows if r["group"] == "min"]
    lines = [f"== {name}: {len(rows)} images ({len(maj)} majority, {len(mnr)} minority)"]

    with_digit = Counter(d for r in rows for d in (r["d1"], r["d2"]))
    with_partner = Counter(d for r in maj for d in (r["d1"], r["d2"]))
    p = [with_partner[d] / with_digit[d] for d in range(10)]
    lines.append(f"   P(partner|d) for d=0..9: {' '.join(f'{x:.4f}' for x in p)}")

    pair_counts = Counter(r["pair"] for r in maj)
    orient = Counter((r["d1"], r["d2"]) for r in maj)
    lines.append(
        "   majority per pair: " + ", ".join(
            f"{a}_{b}={pair_counts[f'{a}_{b}']} ({orient[(a, b)]}/{orient[(b, a)]})" for a, b in PAIRS
        )
    )
    if mnr:
        unordered = Counter(r["pair"] for r in mnr)
        ordered = Counter((r["d1"], r["d2"]) for r in mnr)
        per_digit = Counter(d for r in mnr for d in (r["d1"], r["d2"]))
        left = Counter(r["d1"] for r in mnr)
        lines.append(
            f"   minority: unordered pairs {len(unordered)}/40, ordered pairs {len(ordered)}/80, "
            f"per ordered pair {min(ordered.values())}-{max(ordered.values())}, "
            f"per digit {min(per_digit.values())}-{max(per_digit.values())}, "
            f"left share per digit {min(left[d] / per_digit[d] for d in range(10)):.3f}-"
            f"{max(left[d] / per_digit[d] for d in range(10)):.3f}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("/home/all_data/cooccurrence_1x2"))
    parser.add_argument("--mnist-root", type=Path, default=Path("/home/all_data/mnist_raw"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    parser.add_argument("--overwrite", action="store_true", help="delete existing train/, val/, test/ and splits/ first")
    args = parser.parse_args()

    for sub in (*SPLITS, "splits"):
        target = args.root / sub
        if target.exists():
            if not args.overwrite:
                raise SystemExit(f"{target} already exists; pass --overwrite to replace it")
            shutil.rmtree(target)

    targets = {}
    for source, is_train in (("train", True), ("test", False)):
        ds = MNIST(str(args.mnist_root), train=is_train, download=False)
        _images[source] = ds.data.numpy()
        targets[source] = ds.targets.numpy()
    train_idx, val_idx = train_val_indices(targets["train"], VAL_DIGITS, args.seed)
    np.savez(args.root / "mnist_train_val_split.npz", train_idx=train_idx, val_idx=val_idx)
    allowed = {"train": train_idx, "val": val_idx, "test": np.arange(len(targets["test"]))}

    rows = build_rows(pool_specs(), targets, allowed, args.seed)
    assign_test_subsets(rows)
    write_images(rows, args.root, args.workers)
    write_csv(args.root / "metadata.csv", rows)

    splits_dir = args.root / "splits"
    splits_dir.mkdir()
    splits = {
        f"{split}_{name}": select_split(rows, split, *split_counts(cr, total))
        for split, total in TOTALS.items()
        for name, cr in RATIOS.items()
    }
    test = [r for r in rows if r["split"] == "test"]
    splits["test"] = test
    splits["test_maj"] = [r for r in test if r["group"] == "maj"]
    for k in range(N_TEST_SUBSETS):
        splits[f"test_min_subset{k}"] = [r for r in test if r["subset"] == k]

    report = [f"MNIST train digits: {len(train_idx)} for train images, {len(val_idx)} for val images"]
    for name, split_rows in splits.items():
        write_csv(splits_dir / f"{name}.csv", split_rows)
        report += summarize(name, split_rows)
    (splits_dir / "summary.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()

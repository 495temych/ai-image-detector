import argparse
import random
import shutil
from pathlib import Path


def prepare(raw_dir: Path, processed_dir: Path, per_class: int,
            split: list[float], seed: int) -> None:
    assert abs(sum(split) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    random.seed(seed)

    class_names = ["real", "fake"]
    split_names = ["train", "val", "test"]

    for split_name in split_names:
        for cls in class_names:
            (processed_dir / split_name / cls).mkdir(parents=True, exist_ok=True)

    for cls in class_names:
        cls_dir = raw_dir / cls
        images = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
        if len(images) < per_class:
            raise ValueError(
                f"Only {len(images)} images found in {cls_dir}, need {per_class}"
            )
        sampled = random.sample(images, per_class)

        n = len(sampled)
        n_train = int(n * split[0])
        n_val = int(n * split[1])

        assignments = (
            [("train", img) for img in sampled[:n_train]]
            + [("val", img) for img in sampled[n_train : n_train + n_val]]
            + [("test", img) for img in sampled[n_train + n_val :]]
        )

        for split_name, img_path in assignments:
            dest = processed_dir / split_name / cls / img_path.name
            shutil.copy2(img_path, dest)

        counts = {s: sum(1 for s2, _ in assignments if s2 == s) for s in split_names}
        print(f"  {cls}: train={counts['train']} val={counts['val']} test={counts['test']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset splits")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--split", default="0.70/0.15/0.15")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split = [float(x) for x in args.split.split("/")]
    prepare(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        per_class=args.per_class,
        split=split,
        seed=args.seed,
    )
    print("Done.")


if __name__ == "__main__":
    main()

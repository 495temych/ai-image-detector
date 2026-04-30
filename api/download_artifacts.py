"""One-time script to pull model artifacts and sample images from S3/DagHub."""
import os
from pathlib import Path
import boto3
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

KEY_ID = os.environ["DAGSHUB_KEY_ID"]
SECRET = os.environ["DAGSHUB_S3"]
ENDPOINT = os.environ["S3_ENDPOINT"]
BUCKET = os.environ["S3_BUCKET"]
PREFIX = os.environ.get("S3_PREFIX", "models/efficientnet_v1")

API_DIR = Path(__file__).parent

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=KEY_ID,
    aws_secret_access_key=SECRET,
)


def download(s3_key: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        print(f"  skip (exists): {local_path.name}")
        return
    print(f"  downloading: {s3_key} → {local_path}")
    s3.download_file(BUCKET, s3_key, str(local_path))


def download_folder(s3_prefix: str, local_dir: Path, limit: int = 10) -> None:
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            if count >= limit:
                return
            key = obj["Key"]
            filename = Path(key).name
            if not filename:
                continue
            download(key, local_dir / filename)
            count += 1


if __name__ == "__main__":
    print("Pulling model artifacts...")
    download(f"{PREFIX}/model.onnx", API_DIR / "model.onnx")
    download(f"{PREFIX}/model.onnx.data", API_DIR / "model.onnx.data")
    download(f"{PREFIX}/best_weights.pt", API_DIR / "best_weights.pt")

    print("Pulling sample test images...")
    download_folder("data/v1_subset_5000/test/fake/", API_DIR / "samples/fake/", limit=10)
    download_folder("data/v1_subset_5000/test/real/", API_DIR / "samples/real/", limit=10)

    print("Done.")

"""
dataset/download.py
-------------------
Handles dataset acquisition from all four supported sources:

  1. "torchvision" — built-in datasets (MNIST, CIFAR-10, etc.)
  2. "local"       — a folder already on disk (ImageFolder format)
  3. "url"         — direct link to a .zip or .tar.gz file
  4. "github"      — a GitHub folder URL, release asset, or full repo URL

The main entry point is download_dataset(config, logger) which reads
config.DATASET["source"] and dispatches to the correct strategy.
All strategies save raw data under config.DATA_DIR.

Beginners: Think of this file as a smart downloader that figures out
WHERE your data is and fetches it correctly regardless of the source.
"""

import logging
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

from config import DATASET, DATA_DIR
from utils  import Timer


# Supported torchvision dataset names and their classes
_TORCHVISION_DATASETS = {
    "mnist"       : "MNIST",
    "cifar10"     : "CIFAR10",
    "cifar100"    : "CIFAR100",
    "fashionmnist": "FashionMNIST",
    "svhn"        : "SVHN",
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def download_dataset(logger: logging.Logger = None) -> Path:
    """
    Download or locate the dataset based on DATASET["source"] in config.py.

    Parameters
    ----------
    logger : logging.Logger, optional

    Returns
    -------
    Path  The directory where raw dataset files now live.

    Raises
    ------
    ValueError  If the source type is unknown or required fields are missing.
    """
    source = DATASET["source"].lower()
    log    = logger.info if logger else print

    log(f"Dataset source: '{source}'")

    if source == "torchvision":
        return _download_torchvision(logger)
    elif source == "local":
        return _use_local(logger)
    elif source == "url":
        return _download_url(DATASET["url"], logger)
    elif source == "github":
        return _download_github(DATASET["url"], logger)
    else:
        raise ValueError(
            f"Unknown dataset source '{source}'. "
            f"Choose from: 'torchvision', 'local', 'url', 'github'."
        )


# ---------------------------------------------------------------------------
# Strategy 1: torchvision built-in datasets
# ---------------------------------------------------------------------------

def _download_torchvision(logger: logging.Logger = None) -> Path:
    """
    Download a torchvision built-in dataset (MNIST, CIFAR-10, etc.).

    torchvision handles caching — if the dataset is already on disk it
    skips the download automatically.

    Returns
    -------
    Path  DATA_DIR (torchvision creates its own subfolder structure inside)
    """
    import torchvision.datasets as tvd

    log  = logger.info if logger else print
    name = DATASET["name"].lower().replace("-", "")

    if name not in _TORCHVISION_DATASETS:
        raise ValueError(
            f"Unknown torchvision dataset '{DATASET['name']}'. "
            f"Supported: {list(_TORCHVISION_DATASETS.values())}"
        )

    cls_name = _TORCHVISION_DATASETS[name]
    cls      = getattr(tvd, cls_name)

    with Timer(f"Downloading {cls_name} (train)", logger=logger):
        log(f"Fetching {cls_name} train split → {DATA_DIR}")
        # SVHN uses split='train' instead of train=True
        if cls_name == "SVHN":
            cls(root=str(DATA_DIR), split="train", download=True)
        else:
            cls(root=str(DATA_DIR), train=True,  download=True)

    with Timer(f"Downloading {cls_name} (test)", logger=logger):
        log(f"Fetching {cls_name} test split  → {DATA_DIR}")
        if cls_name == "SVHN":
            cls(root=str(DATA_DIR), split="test", download=True)
        else:
            cls(root=str(DATA_DIR), train=False, download=True)

    log(f"{cls_name} ready at: {DATA_DIR}")
    return DATA_DIR


# ---------------------------------------------------------------------------
# Strategy 2: local folder
# ---------------------------------------------------------------------------

def _use_local(logger: logging.Logger = None) -> Path:
    """
    Validate and return the path to a local dataset folder.

    Expected structure (ImageFolder format):
        dataset_root/
            class_a/image1.jpg
            class_a/image2.jpg
            class_b/image1.jpg
            ...

    Returns
    -------
    Path  The validated local dataset root.

    Raises
    ------
    ValueError  If local_path is not set or the directory does not exist.
    """
    log = logger.info if logger else print

    local_path = DATASET.get("local_path")
    if not local_path:
        raise ValueError(
            "DATASET['local_path'] must be set when source='local'."
        )

    path = Path(local_path).resolve()
    if not path.exists():
        raise ValueError(f"Local dataset path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Local dataset path is not a directory: {path}")

    # Count class subdirectories
    classes = [d for d in path.iterdir() if d.is_dir()]
    log(f"Local dataset found: {path}")
    log(f"  Class folders detected: {len(classes)}")
    for c in sorted(classes)[:10]:   # show first 10 only
        count = len(list(c.glob("*")))
        log(f"    {c.name}: {count} files")
    if len(classes) > 10:
        log(f"    ... and {len(classes) - 10} more classes")

    return path


# ---------------------------------------------------------------------------
# Strategy 3: direct URL (.zip or .tar.gz)
# ---------------------------------------------------------------------------

def _download_url(url: str, logger: logging.Logger = None) -> Path:
    """
    Download a dataset from a direct URL pointing to a .zip or .tar.gz file.

    The archive is extracted into DATA_DIR. If the file already exists on
    disk (same filename), the download is skipped.

    Parameters
    ----------
    url    : str  Direct link to a .zip or .tar.gz archive.
    logger : logging.Logger, optional

    Returns
    -------
    Path  The directory where the extracted dataset lives.
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "The 'requests' library is required for URL downloads. "
            "Install it with: pip install requests"
        )

    log = logger.info if logger else print

    if not url:
        raise ValueError("DATASET['url'] must be set when source='url'.")

    filename  = url.split("/")[-1].split("?")[0]   # strip query params
    dest_file = DATA_DIR / filename

    # Skip download if file already exists
    if dest_file.exists():
        log(f"Archive already exists, skipping download: {dest_file}")
    else:
        with Timer(f"Downloading {filename}", logger=logger):
            log(f"Downloading: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(dest_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)

            log(f"Downloaded {downloaded / 1e6:.1f} MB → {dest_file}")

    # Extract the archive
    extract_dir = DATA_DIR / dest_file.stem.replace(".tar", "")
    if not extract_dir.exists():
        with Timer(f"Extracting {filename}", logger=logger):
            _extract_archive(dest_file, DATA_DIR, logger)
    else:
        log(f"Already extracted: {extract_dir}")

    return extract_dir


# ---------------------------------------------------------------------------
# Strategy 4: GitHub URL
# ---------------------------------------------------------------------------

def _download_github(url: str, logger: logging.Logger = None) -> Path:
    """
    Download a dataset from a GitHub URL. Supports three sub-cases:

    Case A — Release asset (.zip/.tar.gz link):
        e.g. https://github.com/user/repo/releases/download/v1/data.zip
        → treated exactly like a direct URL download

    Case B — GitHub folder (tree URL):
        e.g. https://github.com/user/repo/tree/main/data/
        → uses GitHub Contents API to list files, downloads each one

    Case C — Full repo URL:
        e.g. https://github.com/user/repo
        → shallow-clones the repo (--depth=1), copies the data folder

    Parameters
    ----------
    url    : str  Any GitHub URL.
    logger : logging.Logger, optional

    Returns
    -------
    Path  Directory containing the downloaded dataset.
    """
    log = logger.info if logger else print

    if not url:
        raise ValueError("DATASET['url'] must be set when source='github'.")

    log(f"GitHub URL detected: {url}")

    # Case A: release asset — contains a file extension in the last segment
    last_segment = url.split("/")[-1].split("?")[0]
    if any(last_segment.endswith(ext) for ext in [".zip", ".tar.gz", ".tar"]):
        log("Identified as: GitHub release asset → using direct download")
        return _download_url(url, logger)

    # Case B: folder URL (contains /tree/ in path)
    if "/tree/" in url:
        log("Identified as: GitHub folder URL → using Contents API")
        return _download_github_folder(url, logger)

    # Case C: full repo URL
    log("Identified as: GitHub repository → shallow clone")
    return _clone_github_repo(url, logger)


def _download_github_folder(url: str, logger: logging.Logger = None) -> Path:
    """
    Download individual files from a GitHub folder using the Contents API.

    Parses a URL like:
        https://github.com/user/repo/tree/main/path/to/data
    into API calls to:
        https://api.github.com/repos/user/repo/contents/path/to/data?ref=main
    """
    try:
        import requests
    except ImportError:
        raise ImportError("Install 'requests': pip install requests")

    log = logger.info if logger else print

    # Parse URL: github.com/USER/REPO/tree/BRANCH/PATH
    pattern = r"github\.com/([^/]+)/([^/]+)/tree/([^/]+)/?(.*)"
    match   = re.search(pattern, url)
    if not match:
        raise ValueError(f"Cannot parse GitHub folder URL: {url}")

    owner, repo, branch, folder_path = match.groups()
    api_url = (
        f"https://api.github.com/repos/{owner}/{repo}"
        f"/contents/{folder_path}?ref={branch}"
    )

    log(f"Fetching file list from GitHub API: {api_url}")
    response = requests.get(api_url)
    response.raise_for_status()
    files = response.json()

    if not isinstance(files, list):
        raise ValueError(f"GitHub API returned unexpected response for: {url}")

    dest_dir = DATA_DIR / f"{repo}_{folder_path.replace('/', '_')}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    with Timer(f"Downloading {len(files)} files from GitHub", logger=logger):
        for item in files:
            if item["type"] == "file":
                file_url  = item["download_url"]
                file_dest = dest_dir / item["name"]
                if not file_dest.exists():
                    r = requests.get(file_url)
                    r.raise_for_status()
                    file_dest.write_bytes(r.content)
                    log(f"  Downloaded: {item['name']}")
                else:
                    log(f"  Already exists: {item['name']}")

    log(f"GitHub folder downloaded to: {dest_dir}")
    return dest_dir


def _clone_github_repo(url: str, logger: logging.Logger = None) -> Path:
    """
    Shallow-clone a GitHub repository (fastest: only gets the latest commit).
    """
    log    = logger.info if logger else print
    # Extract repo name from URL for the destination folder name
    repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
    dest_dir  = DATA_DIR / repo_name

    if dest_dir.exists():
        log(f"Repo already cloned: {dest_dir}")
        return dest_dir

    with Timer(f"Cloning {repo_name}", logger=logger):
        log(f"Shallow cloning: {url}")
        result = subprocess.run(
            ["git", "clone", "--depth=1", url, str(dest_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git clone failed:\n{result.stderr}"
            )

    log(f"Repository cloned to: {dest_dir}")
    return dest_dir


# ---------------------------------------------------------------------------
# Archive extraction helper
# ---------------------------------------------------------------------------

def _extract_archive(archive_path: Path, dest_dir: Path, logger=None) -> None:
    """
    Extract a .zip or .tar.gz archive to dest_dir.

    Parameters
    ----------
    archive_path : Path  Path to the archive file.
    dest_dir     : Path  Directory to extract into.
    logger       : logging.Logger, optional
    """
    log = logger.info if logger else print
    log(f"Extracting: {archive_path.name}")

    suffix = "".join(archive_path.suffixes)   # e.g. ".tar.gz" or ".zip"

    if suffix in (".tar.gz", ".tgz", ".tar"):
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(path=dest_dir)
    elif suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(path=dest_dir)
    else:
        raise ValueError(
            f"Unsupported archive format '{suffix}'. "
            f"Supported: .zip, .tar.gz, .tgz"
        )

    log(f"Extracted to: {dest_dir}")
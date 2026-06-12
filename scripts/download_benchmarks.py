#!/usr/bin/env python
"""Download benchmark datasets (HumanEval, MBPP) to data/tasks/.

Requires: pip install requests
Uses proxy from config/settings.yaml automatically.
"""

import gzip
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("download")

DATASETS = {
    "HumanEval": {
        "url": "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
        "dest": "HumanEval.jsonl.gz",
        "final": "HumanEval.jsonl",
        "gzipped": True,
    },
    "MBPP": {
        "url": "https://github.com/google-research/google-research/raw/master/mbpp/mbpp.jsonl",
        "dest": "mbpp.jsonl",
        "final": "mbpp.jsonl",
        "gzipped": False,
    },
}

DATA_DIR = Path("data/tasks")


def get_proxy() -> dict | None:
    try:
        import yaml

        config_path = Path("config/settings.yaml")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            proxy = config.get("proxy", {})
            http_proxy = proxy.get("http", "")
            https_proxy = proxy.get("https", http_proxy)
            if http_proxy:
                return {"http": http_proxy, "https": https_proxy}
    except Exception:
        pass
    return None


def download_file(url: str, dest: Path, proxies: dict | None = None) -> bool:
    logger.info("Downloading %s -> %s", url, dest)

    for attempt, proxy in enumerate([proxies, None]):
        label = "proxy" if proxy else "direct"
        if attempt > 0:
            logger.info("Retrying with %s connection...", label)
        try:
            resp = requests.get(url, proxies=proxy, stream=True, timeout=120)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = min(100, int(downloaded / total * 100))
                        sys.stdout.write(f"\r  {pct}% ")
                        sys.stdout.flush()
            sys.stdout.write("\r  Done.    \n")
            return True
        except Exception as e:
            if attempt == 0 and proxy:
                logger.warning("Proxy download failed (%s), trying direct...", str(e)[:80])
            else:
                logger.error("Download failed: %s", e)

    return False


def main():
    proxies = get_proxy()
    if proxies:
        logger.info("Using proxy: %s", proxies.get("http", proxies.get("https", "none")))
    else:
        logger.info("No proxy configured, using direct connection")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, info in DATASETS.items():
        dest = DATA_DIR / info["dest"]
        final = DATA_DIR / info["final"]
        logger.info("--- %s ---", name)

        if final.exists():
            logger.info("%s already exists at %s, skipping", name, final)
            continue

        if not download_file(info["url"], dest, proxies):
            logger.error("Failed to download %s", name)
            continue

        if info["gzipped"] and dest.suffix == ".gz":
            logger.info("Decompressing %s -> %s", dest, final)
            with gzip.open(dest, "rb") as f_in:
                with open(final, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            dest.unlink()
            logger.info("Decompressed: %s (%d bytes)", final, final.stat().st_size)
        else:
            logger.info("Saved: %s (%d bytes)", final, final.stat().st_size)

    logger.info("Done. Files in %s:", DATA_DIR)
    for f in sorted(DATA_DIR.iterdir()):
        logger.info("  %s  (%d bytes)", f.name, f.stat().st_size)


if __name__ == "__main__":
    main()

"""
DownloadManager: reads sites.csv and dispatches download requests to the
appropriate site-specific downloader for each enabled site.

To add a new site:
  1. Add a row to downloader/sites.csv (set enabled=true).
  2. Create a downloader class in downloader/<type>_downloader.py that
     inherits BaseDownloader and implements download().
  3. Register the type in DOWNLOADER_REGISTRY below.
"""

import csv
import logging
from pathlib import Path

from downloader.mavat_downloader import MavatDownloader

logger = logging.getLogger(__name__)

# Maps the 'type' column in sites.csv to the downloader class
DOWNLOADER_REGISTRY: dict[str, type] = {
    "mavat": MavatDownloader,
}

SITES_CSV = Path(__file__).parent / "sites.csv"


class DownloadManager:
    def __init__(self, dest_dir: Path | None = None):
        self.dest_dir = dest_dir or Path("data/raw")
        self._downloaders = self._load_downloaders()

    def _load_downloaders(self) -> list:
        downloaders = []
        with open(SITES_CSV, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("enabled", "").strip().lower() != "true":
                    continue
                site_type = row.get("type", "").strip()
                cls = DOWNLOADER_REGISTRY.get(site_type)
                if cls is None:
                    logger.warning("Unknown downloader type %r (site %r) — skipping", site_type, row.get("name"))
                    continue
                downloaders.append((row["name"], cls()))
        return downloaders

    def download(self, plan_name: str) -> tuple[list[Path], list[str]]:
        """
        Download documents for the given plan name from all enabled sites.
        Returns (list of downloaded file paths, list of log messages).
        """
        all_files: list[Path] = []
        all_log: list[str] = []
        for site_name, downloader in self._downloaders:
            logger.info("Downloading from site=%s plan_name=%r", site_name, plan_name)
            try:
                files = downloader.download(plan_name, self.dest_dir)
                all_files.extend(files)
                if hasattr(downloader, "log"):
                    all_log.extend(downloader.log)
                logger.info("Site %s returned %d file(s)", site_name, len(files))
            except Exception as exc:
                msg = f"שגיאה ב-{site_name}: {exc}"
                logger.error("Site %s download failed: %s", site_name, exc)
                all_log.append(msg)
        return all_files, all_log

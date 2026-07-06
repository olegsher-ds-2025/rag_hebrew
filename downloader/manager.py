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
        self._downloader_classes = self._load_downloader_classes()

    def _load_downloader_classes(self) -> list:
        """Read sites.csv once; downloader INSTANCES are created per download()
        call, so concurrent requests never share mutable log/session state."""
        classes = []
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
                classes.append((row["name"], cls))
        return classes

    def download(self, plan_name: str) -> tuple[list[Path], list[str], list[str]]:
        """
        Download documents for the given plan name from all enabled sites.

        Returns (downloaded file paths, log messages, metadata text chunks).
        The metadata chunks are plan-info summaries (status, area, dates) that
        can be indexed even when the PDFs themselves are not downloadable.
        """
        all_files: list[Path] = []
        all_log: list[str] = []
        all_metadata: list[str] = []
        for site_name, cls in self._downloader_classes:
            downloader = cls()
            logger.info("Downloading from site=%s plan_name=%r", site_name, plan_name)
            try:
                files = downloader.download(plan_name, self.dest_dir)
                all_files.extend(files)
                if hasattr(downloader, "log"):
                    all_log.extend(downloader.log)
                if hasattr(downloader, "metadata"):
                    all_metadata.extend(downloader.metadata)
                logger.info("Site %s returned %d file(s)", site_name, len(files))
            except Exception as exc:
                msg = f"שגיאה ב-{site_name}: {exc}"
                logger.error("Site %s download failed: %s", site_name, exc)
                all_log.append(msg)
        return all_files, all_log, all_metadata

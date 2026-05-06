from abc import ABC, abstractmethod
from pathlib import Path


class BaseDownloader(ABC):
    """
    Base class for site-specific document downloaders.
    Subclasses implement download() for their respective site.
    """

    @abstractmethod
    def download(self, gush: str, helka: str, dest_dir: Path) -> list[Path]:
        """
        Download all documents matching the given gush/helka from this site.

        Args:
            gush:     Block number (גוש)
            helka:    Parcel number (חלקה)
            dest_dir: Directory to save downloaded files into

        Returns:
            List of paths of successfully downloaded files.
        """

from abc import ABC, abstractmethod
from pathlib import Path


class BaseDownloader(ABC):
    """
    Base class for site-specific document downloaders.
    Subclasses implement download() for their respective site.
    """

    @abstractmethod
    def download(self, plan_name: str, dest_dir: Path) -> list[Path]:
        """
        Download all documents matching the given plan name (שם תכנית) from this site.

        Args:
            plan_name: Plan name to search for (שם תכנית), e.g. "נוף הפארק - יובלים גנים"
            dest_dir:  Directory to save downloaded files into

        Returns:
            List of paths of successfully downloaded files.
        """

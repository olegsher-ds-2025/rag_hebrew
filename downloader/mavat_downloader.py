"""
Downloader for mavat.iplan.gov.il (Israel Planning Authority).

Flow:
  1. Query the public ArcGIS REST service to find plan IDs that cover the given
     gush/helka.
  2. For each plan, call the mavat document-list API to get all attached PDF docs.
  3. Download each PDF into dest_dir, skipping files that already exist.
"""

import requests
import logging
from pathlib import Path
from urllib.parse import urljoin

from downloader.base_downloader import BaseDownloader

logger = logging.getLogger(__name__)

# ArcGIS REST service – layer 1 contains approved/deposited plans indexed by parcel
PLANS_QUERY_URL = (
    "https://ags.iplan.gov.il/arcgisiplan/rest/services/"
    "PlanningPublic/Xplan/MapServer/1/query"
)

# mavat API – returns document metadata for a given plan number
MAVAT_DOCS_URL = "https://mavat.iplan.gov.il/SV4/1/{plan_number}"

# mavat API endpoint that returns the JSON document list for a plan
MAVAT_API_DOCS = "https://mavat.iplan.gov.il/api/documents/{plan_number}"

# Direct PDF download URL pattern served by the mavat document viewer
MAVAT_PDF_URL = "https://mavat.iplan.gov.il/SV4/1/{plan_number}/{doc_id}"

REQUEST_TIMEOUT = 30  # seconds


class MavatDownloader(BaseDownloader):
    """Download planning documents from mavat.iplan.gov.il by גוש/חלקה."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; HebrewRAG/1.0)"
        })

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download(self, gush: str, helka: str, dest_dir: Path) -> list[Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)

        plan_numbers = self._search_plans(gush, helka)
        if not plan_numbers:
            logger.warning("mavat: no plans found for gush=%s helka=%s", gush, helka)
            return []

        downloaded: list[Path] = []
        for plan_num in plan_numbers:
            logger.info("mavat: fetching documents for plan %s", plan_num)
            docs = self._get_document_list(plan_num)
            for doc_id, filename in docs:
                path = self._download_pdf(plan_num, doc_id, filename, dest_dir)
                if path:
                    downloaded.append(path)

        return downloaded

    # ------------------------------------------------------------------
    # Step 1: find plan numbers via ArcGIS REST
    # ------------------------------------------------------------------

    def _search_plans(self, gush: str, helka: str) -> list[str]:
        params = {
            "where": f"GUSH_NUM={gush} AND PARCEL_NUM={helka}",
            "outFields": "PL_NUMBER",
            "returnDistinctValues": "true",
            "returnGeometry": "false",
            "f": "json",
        }
        try:
            resp = self.session.get(PLANS_QUERY_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("mavat: ArcGIS query failed: %s", exc)
            return []

        features = data.get("features", [])
        plan_numbers = []
        for feat in features:
            attrs = feat.get("attributes", {})
            pl = attrs.get("PL_NUMBER") or attrs.get("pl_number")
            if pl:
                plan_numbers.append(str(pl).strip())

        logger.info("mavat: found %d plan(s) for gush=%s helka=%s", len(plan_numbers), gush, helka)
        return plan_numbers

    # ------------------------------------------------------------------
    # Step 2: get document list for a plan
    # ------------------------------------------------------------------

    def _get_document_list(self, plan_number: str) -> list[tuple[str, str]]:
        """Return list of (doc_id, filename) tuples for the given plan."""
        url = MAVAT_API_DOCS.format(plan_number=plan_number)
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(
                "mavat: document list API failed for plan %s (%s); "
                "falling back to direct plan-page scrape",
                plan_number, exc
            )
            return self._scrape_plan_page(plan_number)

        # The API returns a list of document objects; field names vary –
        # try several common key names.
        docs = []
        items = data if isinstance(data, list) else data.get("documents", data.get("docs", []))
        for item in items:
            doc_id = (
                item.get("DOC_ID") or item.get("docId") or
                item.get("id") or item.get("ID") or ""
            )
            name = (
                item.get("DOC_NAME") or item.get("docName") or
                item.get("name") or item.get("NAME") or f"doc_{doc_id}"
            )
            if doc_id:
                docs.append((str(doc_id), str(name)))

        return docs

    def _scrape_plan_page(self, plan_number: str) -> list[tuple[str, str]]:
        """Fallback: scrape the mavat plan viewer page for PDF hrefs."""
        url = MAVAT_DOCS_URL.format(plan_number=plan_number)
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except Exception as exc:
            logger.error("mavat: plan page scrape failed for plan %s: %s", plan_number, exc)
            return []

        import re
        # Look for links that look like document IDs in the page HTML
        doc_ids = re.findall(r'/SV4/1/' + re.escape(plan_number) + r'/(\d+)', resp.text)
        return [(did, f"plan_{plan_number}_doc_{did}.pdf") for did in set(doc_ids)]

    # ------------------------------------------------------------------
    # Step 3: download a single PDF
    # ------------------------------------------------------------------

    def _download_pdf(
        self, plan_number: str, doc_id: str, filename: str, dest_dir: Path
    ) -> Path | None:
        # Ensure the filename ends with .pdf
        safe_name = filename if filename.lower().endswith(".pdf") else filename + ".pdf"
        # Sanitise for filesystem
        safe_name = safe_name.replace("/", "_").replace("\\", "_")
        dest = dest_dir / safe_name

        if dest.exists():
            logger.info("mavat: skipping %s (already downloaded)", safe_name)
            return dest

        url = MAVAT_PDF_URL.format(plan_number=plan_number, doc_id=doc_id)
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)
            logger.info("mavat: downloaded %s -> %s", url, dest)
            return dest
        except Exception as exc:
            logger.error("mavat: failed to download %s: %s", url, exc)
            return None

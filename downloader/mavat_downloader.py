"""
Downloader for mavat.iplan.gov.il (Israel Planning Authority).

Flow:
  1. Query the public ArcGIS REST service to find plan IDs matching the given
     plan name (שם תכנית) using a LIKE search on the pl_name field.
  2. For each plan, call the mavat document-list API to get all attached PDF docs.
  3. Download each PDF into dest_dir, skipping files that already exist.

Progress messages are accumulated in self.log (list[str]) during each download() call.
"""

import ssl
import requests
import logging
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

from downloader.base_downloader import BaseDownloader

logger = logging.getLogger(__name__)

# ArcGIS REST service – layer 1 contains approved/deposited plans indexed by plan name
PLANS_QUERY_URL = (
    "https://ags.iplan.gov.il/arcgisiplan/rest/services/"
    "PlanningPublic/Xplan/MapServer/1/query"
)

# mavat REST API base (may be under maintenance)
MAVAT_REST_BASE = "https://mavat.iplan.gov.il/rest/api"

REQUEST_TIMEOUT = 60  # seconds; the ArcGIS planning layer is slow (~20-35s)


class _LegacySSLAdapter(HTTPAdapter):
    """HTTPAdapter that lowers TLS cipher security level to connect to older government servers."""

    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl_context"] = ctx
        super().init_poolmanager(*args, **kwargs)


class MavatDownloader(BaseDownloader):
    """Download planning documents from mavat.iplan.gov.il by שם תכנית (plan name)."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; HebrewRAG/1.0)"
        })
        # Both gov hosts (ags.iplan.gov.il ArcGIS and mavat.iplan.gov.il) use old
        # TLS ciphers that fail the handshake under strict OpenSSL configs (e.g.
        # inside the container: SSLV3_ALERT_HANDSHAKE_FAILURE). The legacy adapter
        # lowers SECLEVEL so the handshake succeeds, so mount it for all https.
        # It adds some latency to ArcGIS — REQUEST_TIMEOUT is sized to absorb it.
        self.session.mount("https://", _LegacySSLAdapter())
        self.log: list[str] = []

    def _emit(self, msg: str) -> None:
        logger.info("mavat: %s", msg)
        self.log.append(msg)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download(self, plan_name: str, dest_dir: Path) -> list[Path]:
        self.log = []
        dest_dir.mkdir(parents=True, exist_ok=True)

        self._emit(f"מחפש תכניות עבור: {plan_name}")
        plans = self._search_plans(plan_name)
        if not plans:
            self._emit("לא נמצאו תכניות מתאימות")
            return []

        self._emit(f"נמצאו {len(plans)} תכנית/ות:")
        for p in plans:
            self._emit(f"  • {p['pl_number']} — {p['pl_name']}  ({p['pl_url']})")

        downloaded: list[Path] = []
        for plan in plans:
            plan_num = plan["pl_number"]
            self._emit(f"מאחזר מסמכים לתכנית {plan_num}")
            docs = self._get_document_list(plan_num, plan["pl_url"])
            if not docs:
                self._emit(f"  לא נמצאו מסמכים להורדה — ניתן לצפות בתכנית ב: {plan['pl_url']}")
                continue
            for doc_id, filename in docs:
                path = self._download_pdf(plan_num, doc_id, filename, dest_dir)
                if path:
                    self._emit(f"  ✓ הורד: {path.name}")
                    downloaded.append(path)
                else:
                    self._emit(f"  ✗ כשל בהורדת: {filename}")

        return downloaded

    # ------------------------------------------------------------------
    # Step 1: find plan numbers via ArcGIS REST
    # ------------------------------------------------------------------

    def _search_plans(self, plan_name: str) -> list[dict]:
        """
        Return list of dicts with pl_number, pl_name, pl_url.

        The query string may be either a plan number (מספר תכנית, e.g.
        "504-0100552") or a free-text plan name (שם תכנית). A single LIKE-on-both
        query covers either case, so the caller does not need to know which was
        entered.
        """
        safe = plan_name.replace("'", "''")
        params = {
            "where": f"pl_number LIKE '%{safe}%' OR pl_name LIKE '%{safe}%'",
            "outFields": "pl_number,pl_name,pl_url",
            "returnDistinctValues": "true",
            "returnGeometry": "false",
            "f": "json",
        }
        try:
            resp = self.session.get(PLANS_QUERY_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            self._emit(f"שגיאה בחיפוש ArcGIS: {exc}")
            return []

        results = []
        for feat in data.get("features", []):
            attrs = feat.get("attributes", {})
            pl_num = attrs.get("pl_number") or attrs.get("PL_NUMBER")
            if pl_num:
                results.append({
                    "pl_number": str(pl_num).strip(),
                    "pl_name": attrs.get("pl_name") or attrs.get("PL_NAME") or "",
                    "pl_url": attrs.get("pl_url") or attrs.get("PL_URL") or "",
                })
        return results

    # ------------------------------------------------------------------
    # Step 2: get document list for a plan
    # ------------------------------------------------------------------

    def _get_document_list(self, plan_number: str, pl_url: str) -> list[tuple[str, str]]:
        """Return list of (doc_id, filename) tuples for the given plan."""
        url = f"{MAVAT_REST_BASE}/documents/{plan_number}"
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            # When the mavat REST API is down it redirects to maintenance.gov.il
            # (HTTP 200 + HTML). Detect this explicitly so the log is unambiguous.
            if "maintenance.gov.il" in resp.url or b"maintenance.gov.il" in resp.content[:1024]:
                self._emit('  ⚠ ממשק המסמכים של מבא"ת בתחזוקה (maintenance.gov.il) — מנסה דרך דף התכנית')
                return self._scrape_plan_page(pl_url, plan_number)
            if not resp.ok or resp.content[:64].lstrip().startswith(b"<"):
                raise ValueError(f"API returned non-JSON (status {resp.status_code})")
            data = resp.json()
        except Exception as exc:
            self._emit(f"  ממשק המסמכים אינו זמין ({exc}) — מנסה דרך דף התכנית")
            return self._scrape_plan_page(pl_url, plan_number)

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

    def _scrape_plan_page(self, pl_url: str, plan_number: str) -> list[tuple[str, str]]:
        """Fallback: scrape the mavat plan viewer page for PDF hrefs."""
        if not pl_url:
            self._emit("  אין קישור לדף התכנית — לא ניתן לאחזר מסמכים")
            return []
        import re
        try:
            resp = self.session.get(pl_url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except Exception as exc:
            self._emit(f"  כשל בגישה לדף התכנית: {exc}")
            return []

        doc_ids = re.findall(r'/SV4/\d+/\d+/(\d+)', resp.text)
        if not doc_ids:
            # The plan viewer is a JavaScript SPA: the static HTML carries no
            # document links, so there is nothing to scrape. Tell the user where
            # to grab the files manually instead of silently returning empty.
            self._emit(
                "  דף התכנית נטען כיישום JavaScript (SPA) — רשימת המסמכים אינה זמינה "
                f"בקוד הסטטי. ניתן לצפות ולהוריד ידנית ב: {pl_url}"
            )
        return [(did, f"plan_{plan_number}_doc_{did}.pdf") for did in set(doc_ids)]

    # ------------------------------------------------------------------
    # Step 3: download a single PDF
    # ------------------------------------------------------------------

    def _download_pdf(
        self, plan_number: str, doc_id: str, filename: str, dest_dir: Path
    ) -> Path | None:
        safe_name = filename if filename.lower().endswith(".pdf") else filename + ".pdf"
        safe_name = safe_name.replace("/", "_").replace("\\", "_")
        dest = dest_dir / safe_name

        if dest.exists():
            self._emit(f"  דילוג על {safe_name} (כבר קיים)")
            return dest

        url = f"{MAVAT_REST_BASE}/Attacments/?eid={doc_id}&fn={safe_name}&edn=temp-default&pn={plan_number}"
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)
            return dest
        except Exception as exc:
            logger.error("mavat: failed to download %s: %s", url, exc)
            return None

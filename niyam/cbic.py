"""Client for the CBIC Tax Information Portal's public circular API.

The portal (https://taxinformation.cbic.gov.in/) is an Angular app backed by a JSON API.
Two endpoints matter here:

- ``GET /api/cbic-circular-msts/fetchAllCircularsByTaxId/{taxId}`` lists every circular.
- ``GET /api/cbic-circular-msts/download/{id}/ENG`` returns ``{"fileName": ..., "data": <base64 PDF>}``.
"""

from __future__ import annotations

import base64
import ssl
import unicodedata
from dataclasses import dataclass

import httpx
import truststore

BASE_URL = "https://taxinformation.cbic.gov.in/api/cbic-circular-msts"
GST_TAX_ID = 1000001  # the portal's id for GST
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "Mozilla/5.0 (compatible; niyam/0.1; GST circular research project)",
    "Referer": "https://taxinformation.cbic.gov.in/",
}


def make_client(timeout: float = 120.0) -> httpx.Client:
    # The portal serves an incomplete certificate chain. Verify through the OS trust store
    # (which can fetch the missing intermediate) instead of turning verification off.
    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return httpx.Client(verify=ctx, headers=HEADERS, timeout=httpx.Timeout(timeout, connect=20))


@dataclass(frozen=True)
class Circular:
    id: int
    number: str  # e.g. "164/20/2021-GST"
    date: str  # ISO date, e.g. "2021-10-06"
    subject: str
    category: str  # e.g. "Circulars CGST"
    file_name: str
    has_hindi: bool

    @classmethod
    def from_api(cls, row: dict) -> Circular:
        date = (row.get("circularDt") or "")[:10]
        number = " ".join((row.get("circularNo") or "").split())
        if not number or number.lower() == "corrigendum":
            number = f"Corrigendum dated {date}"  # several corrigenda share the bare label "Corrigendum"
        return cls(
            id=row["id"],
            number=number,
            date=date,
            # NFKC folds the typographic ligatures CBIC uses ("ﬁ" -> "fi") so keyword search matches.
            subject=" ".join(unicodedata.normalize("NFKC", row.get("circularName") or "").split()),
            category=row.get("circularCategory") or "",
            file_name=row.get("docFileName") or "",
            has_hindi=bool(row.get("docFilePathHi")),
        )


def list_circulars(client: httpx.Client, tax_id: int = GST_TAX_ID) -> list[dict]:
    response = client.get(f"{BASE_URL}/fetchAllCircularsByTaxId/{tax_id}")
    response.raise_for_status()
    return response.json()


def download_pdf(client: httpx.Client, circular_id: int, language: str = "ENG") -> bytes:
    response = client.get(f"{BASE_URL}/download/{circular_id}/{language}")
    response.raise_for_status()
    pdf = base64.b64decode(response.json()["data"])
    if not pdf.startswith(b"%PDF"):
        raise ValueError(f"circular {circular_id}: response is not a PDF")
    return pdf

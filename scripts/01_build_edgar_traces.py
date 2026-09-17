#!/usr/bin/env python3
"""Build SEC EDGAR structured-JSON traces for MKTG-3389 Phase 1."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

UA = "DigitalOcean Community Research vbaranwal@digitalocean.com"
HEADERS = {"User-Agent": UA, "Accept-Encoding": "identity"}
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "traces" / "edgar_json"
OUT.mkdir(parents=True, exist_ok=True)

# Diverse large filers by SIC — CIKs without leading zeros for submissions path use zero-pad 10
SEED_TICKERS = [
    # (cik_int, ticker, reason)
    (320193, "AAPL", "tech"),
    (789019, "MSFT", "tech"),
    (1652044, "GOOGL", "tech"),
    (1018724, "AMZN", "retail"),
    (1045810, "NVDA", "semiconductor"),
    (1326801, "META", "tech"),
    (884522, "ORCL", "tech"),
    (21344, "KO", "consumer"),
    (34088, "XOM", "energy"),
    (4962, "AXP", "finance"),
    (19617, "JPM", "finance"),
    (51143, "IBM", "tech"),
    (63908, "MCD", "consumer"),
    (200406, "JNJ", "healthcare"),
    (310158, "MRK", "healthcare"),
    (80424, "PG", "consumer"),
    (1403161, "V", "finance"),
    (1067983, "BRK-B", "finance"),
    (858877, "COST", "retail"),
    (354950, "HD", "retail"),
]

SCHEMAS = {
    "company_profile": {
        "cik", "company_name", "ticker", "sic_code", "sic_description",
        "state_of_incorporation", "fiscal_year_end", "address", "former_names",
    },
    "filing_summary": {
        "accession_number", "form_type", "filing_date", "period_of_report",
        "documents", "is_amended",
    },
    "officers_and_signatories": {
        "filer_cik", "signatories", "principal_executive_offices",
    },
    "financial_highlights": {
        "entity", "period", "currency", "metrics",
    },
}


def get_json(url: str):
    time.sleep(0.12)  # <10 req/s
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def get_bytes(url: str) -> bytes:
    time.sleep(0.12)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def html_to_text(raw: bytes) -> str:
    """Minimal tag strip — enough for extraction prompts, not a full HTML parser."""
    import re
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        text = raw.decode("latin-1", errors="replace")
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&amp;|&lt;|&gt;|&quot;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_filing_body(cik: int, accession: str, primary_doc: str, max_chars: int = 2200) -> str | None:
    """Pull primary filing document text from SEC Archives (filing body, not metadata).

    max_chars kept modest so prompts fit the 2K request-phase regime with
    output headroom (prompt + 320 completion < 2048). Dense XBRL/HTML text
    tokenizes far denser than ~4 chars/token, so 4500 chars overshoots 2K.
    """
    if not accession or not primary_doc:
        return None
    acc_nodash = accession.replace("-", "")
    # Archives path uses CIK without leading-zero pad requirement, but zero-pad works
    url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{acc_nodash}/{primary_doc}"
    )
    try:
        raw = get_bytes(url)
    except Exception as e:
        print(f"  body fetch fail {url}: {e}")
        return None
    text = html_to_text(raw)
    if len(text) < 200:
        return None
    return text[:max_chars]


def cik10(cik: int) -> str:
    return f"{cik:010d}"


def build_company_profile(sub: dict, ticker: str) -> dict:
    addr = {}
    # addresses may be under 'addresses'
    addresses = sub.get("addresses") or {}
    business = addresses.get("business") or addresses.get("mailing") or {}
    if business:
        addr = {
            "street": business.get("street1") or "",
            "city": business.get("city") or "",
            "state": business.get("stateOrCountry") or "",
            "zip": business.get("zipCode") or "",
        }
    former = [
        {"name": f.get("name"), "date_changed": f.get("from")}
        for f in (sub.get("formerNames") or [])
    ]
    return {
        "cik": str(sub.get("cik") or sub.get("entityType") or ""),
        "company_name": sub.get("name"),
        "ticker": ticker,
        "sic_code": str(sub.get("sic") or ""),
        "sic_description": sub.get("sicDescription") or "",
        "state_of_incorporation": sub.get("stateOfIncorporation") or "",
        "fiscal_year_end": sub.get("fiscalYearEnd") or "",
        "address": addr,
        "former_names": former,
    }


def build_filing_summary(sub: dict) -> dict | None:
    filings = (sub.get("filings") or {}).get("recent") or {}
    forms = filings.get("form") or []
    if not forms:
        return None
    # Prefer 10-K then 10-Q then 8-K
    pick = None
    for want in ("10-K", "10-Q", "8-K"):
        for i, form in enumerate(forms):
            if form == want:
                pick = i
                break
        if pick is not None:
            break
    if pick is None:
        pick = 0
    acc = filings["accessionNumber"][pick]
    docs = []
    # primary document only from recent arrays
    primary = filings.get("primaryDocument", [None] * (pick + 1))[pick]
    if primary:
        docs.append({
            "sequence": 1,
            "description": filings.get("primaryDocDescription", [""])[pick] if filings.get("primaryDocDescription") else "",
            "type": filings["form"][pick],
            "filename": primary,
        })
    return {
        "accession_number": acc,
        "form_type": filings["form"][pick],
        "filing_date": filings["filingDate"][pick],
        "period_of_report": (filings.get("reportDate") or [""])[pick] if filings.get("reportDate") else "",
        "documents": docs,
        "is_amended": "A" in (filings["form"][pick] or ""),
    }


def build_officers(sub: dict) -> dict:
    # submissions JSON does not always include signatories; leave structured empty-capable
    addresses = sub.get("addresses") or {}
    business = addresses.get("business") or {}
    return {
        "filer_cik": cik10(int(sub["cik"])) if str(sub.get("cik", "")).isdigit() else str(sub.get("cik")),
        "signatories": [],  # filled from SGML when available; empty allowed with note
        "principal_executive_offices": {
            "street": business.get("street1") or "",
            "city": business.get("city") or "",
            "state": business.get("stateOrCountry") or "",
            "zip": business.get("zipCode") or "",
        },
    }


def build_financials(cik: int, company_name: str) -> dict | None:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10(cik)}.json"
    try:
        facts = get_json(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    usgaap = (facts.get("facts") or {}).get("us-gaap") or {}
    metrics = []
    for tag in ("Assets", "Liabilities", "StockholdersEquity", "Revenues", "NetIncomeLoss"):
        node = usgaap.get(tag)
        if not node:
            continue
        units = node.get("units") or {}
        # prefer USD
        series = units.get("USD") or next(iter(units.values()), [])
        if not series:
            continue
        # latest by end date
        latest = sorted(series, key=lambda x: x.get("end") or "")[-1]
        metrics.append({
            "tag": tag,
            "label": (node.get("label") or tag),
            "value": latest.get("val"),
            "unit": "USD",
            "decimals": latest.get("decimals"),
        })
        if len(metrics) >= 4:
            break
    if not metrics:
        return None
    return {
        "entity": company_name,
        "period": metrics[0].get("value") and (sorted((usgaap.get(metrics[0]["tag"], {}).get("units", {}).get("USD") or []), key=lambda x: x.get("end") or "")[-1].get("end") if False else ""),
        "currency": "USD",
        "metrics": metrics,
    }


def main():
    accession_log = []
    records = []
    for cik, ticker, reason in SEED_TICKERS:
        print(f"fetch {ticker} CIK{cik10(cik)}", flush=True)
        try:
            sub = get_json(f"https://data.sec.gov/submissions/CIK{cik10(cik)}.json")
        except Exception as e:
            print(f"  FAIL submissions: {e}")
            continue
        profile = build_company_profile(sub, ticker)
        filing = build_filing_summary(sub)
        officers = build_officers(sub)
        # fix cik in profile from submissions
        profile["cik"] = cik10(int(sub["cik"]))
        try:
            financials = build_financials(int(sub["cik"]), sub.get("name") or ticker)
        except Exception as e:
            print(f"  financials err: {e}")
            financials = None

        # Build prompt text: submissions metadata PLUS primary filing body text.
        # Without the body, prompts top out near ~227 tokens and only test field
        # rewriting, not document extraction.
        prose = (
            f"Company: {sub.get('name')}\n"
            f"CIK: {profile['cik']}\n"
            f"Ticker: {ticker}\n"
            f"SIC: {profile['sic_code']} {profile['sic_description']}\n"
            f"Incorporated: {profile['state_of_incorporation']}\n"
            f"Fiscal year end: {profile['fiscal_year_end']}\n"
            f"Business address: {profile['address']}\n"
            f"Former names: {profile['former_names']}\n"
        )
        body = None
        if filing:
            prose += (
                f"Recent filing: {filing['form_type']} accession {filing['accession_number']} "
                f"filed {filing['filing_date']} period {filing['period_of_report']}\n"
            )
            accession_log.append(filing["accession_number"])
            primary = (filing.get("documents") or [{}])[0].get("filename")
            body = fetch_filing_body(int(sub["cik"]), filing["accession_number"], primary)
            if body:
                prose += f"\nFILING BODY (truncated):\n{body}\n"
            else:
                print(f"  WARN no filing body for {ticker}")

        if financials and financials.get("metrics"):
            prose += "\nREPORTED FINANCIAL FACTS:\n"
            for m in financials["metrics"]:
                prose += f"- {m.get('label') or m.get('tag')}: {m.get('value')} {m.get('unit')}\n"

        for schema_name, expected in [
            ("company_profile", profile),
            ("filing_summary", filing),
            ("officers_and_signatories", officers),
            ("financial_highlights", financials),
        ]:
            if expected is None:
                continue
            records.append({
                "domain": "structured-json",
                "schema": schema_name,
                "ticker": ticker,
                "cik": profile["cik"],
                "sic_bucket": reason,
                "has_filing_body": bool(body),
                "prompt": (
                    f"Extract a JSON object matching the {schema_name} schema from the text below. "
                    f"Return only valid JSON.\n\nSCHEMA keys: {sorted(SCHEMAS[schema_name])}\n\nTEXT:\n{prose}"
                ),
                "expected": expected,
            })

    (OUT / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (OUT / "accession_numbers.txt").write_text("\n".join(accession_log) + "\n")
    (OUT / "schemas.json").write_text(json.dumps({k: sorted(v) for k, v in SCHEMAS.items()}, indent=2))
    n_with_body = sum(1 for r in records if r.get("has_filing_body"))
    manifest = {
        "n_records": len(records),
        "n_accessions": len(accession_log),
        "n_with_filing_body": n_with_body,
        "accessions": accession_log,
        "user_agent": UA,
        "source": "data.sec.gov submissions + companyfacts + Archives primary document text",
        "note": "Prompts include truncated primary filing body so the domain tests document extraction, not metadata rewrite.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("WROTE", OUT, "records", len(records), "accessions", len(accession_log), "with_body", n_with_body)


if __name__ == "__main__":
    main()

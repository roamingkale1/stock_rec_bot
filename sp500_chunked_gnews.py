#!/usr/bin/env python3
"""
S&P 500 Google News RSS → CSV (chunked, with per-ticker flush + resume)

- Reads tickers from a CSV with 'Symbol' column (e.g., SP500_2025.csv).
- Scrapes Google News RSS for each ticker from --since to --until (default: 2006-01-01..today).
- Splits into N-month chunks (default 3 = quarter) to bypass per-query caps.
- Prints progress per chunk and per ticker.
- **Flushes results to CSV immediately after each ticker** (so progress isn't lost).
- Optional --resume via a simple JSON state file (default: <output>.state.json).
- De-dup across runs by (title,url).

Install:
  pip install requests python-dateutil
"""

import argparse
import csv
import io
import sys
import time
import json
import os
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

import requests
from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SP500ChunkedGNews/1.1)"}

# ---------------------------- Utils / I/O -------------------------------------

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def read_tickers_from_csv(path: str) -> list[str]:
    """Read tickers from CSV; use 'Symbol' column if present, else first column."""
    with open(path, "rb") as raw:
        txt = raw.read().decode("utf-8-sig", errors="replace")
    f = io.StringIO(txt)
    rows = list(csv.reader(f))
    if not rows:
        raise ValueError("Symbols file is empty.")
    header = [h.strip() for h in rows[0]]
    sym_idx = 0
    for i, h in enumerate(header):
        if h.lower() == "symbol":
            sym_idx = i
            break
    tickers: list[str] = []
    for r in rows[1:]:
        if not r or sym_idx >= len(r):
            continue
        sym = (r[sym_idx] or "").strip()
        if not sym or sym.startswith("#"):
            continue
        tickers.append(sym.upper())
    if not tickers:
        raise ValueError("No tickers found (check the 'Symbol' column).")
    return tickers

def normalize_url(link: str) -> str:
    """Unwrap Google News redirects to the original publisher URL when present."""
    try:
        p = urllib.parse.urlparse(link)
        if "news.google.com" in p.netloc:
            qs = urllib.parse.parse_qs(p.query)
            if "url" in qs and qs["url"]:
                return urllib.parse.unquote(qs["url"][0])
    except Exception:
        pass
    return link

def ensure_output_header(path: str):
    """Create/overwrite or ensure header exists once."""
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ticker","date","title","url"])
        w.writeheader()

def load_seen_from_csv(path: str) -> set:
    """Load existing (title,url) pairs to support de-dup across runs."""
    seen = set()
    if not os.path.exists(path):
        return seen
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = (row.get("title","").strip(), row.get("url","").strip())
                if t[0] and t[1]:
                    seen.add(t)
    except Exception:
        pass
    return seen

def append_rows(path: str, rows: List[Dict[str, str]]):
    """Append rows to CSV."""
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ticker","date","title","url"])
        w.writerows(rows)

# ---------------------------- State (resume) ----------------------------------

def load_state(path: str) -> Dict:
    if not os.path.exists(path):
        return {"completed": [], "last_run": None}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"completed": [], "last_run": None}

def save_state(path: str, completed: List[str]):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"completed": completed, "last_run": datetime.utcnow().isoformat()+"Z"}, f, indent=2)
    except Exception as e:
        eprint(f"[warn] could not save state: {e}")

# ---------------------------- Google News RSS ---------------------------------

def fetch_rss(query: str, retries: int = 5, backoff: float = 0.7) -> bytes:
    """Fetch RSS bytes with simple retries and backoff on transient errors."""
    q = urllib.parse.quote_plus(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff * (2 ** i))
                continue
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_err = e
            time.sleep(backoff * (2 ** i))
    raise last_err if last_err else RuntimeError("Fetch failed")

def parse_items(xml_bytes: bytes):
    """Parse <item> entries into [{'title','url','date'}] with UTC datetimes."""
    root = ET.fromstring(xml_bytes)
    items = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link  = (item.findtext("link") or "").strip()
        pub   = item.findtext("pubDate") or ""
        if not (title and link and pub):
            continue
        try:
            dt = dateparser.parse(pub)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
        except Exception:
            continue
        items.append({"title": title, "url": normalize_url(link), "date": dt})
    return items

# ---------------------------- Date chunking -----------------------------------

def chunk_ranges(since_dt: datetime, until_dt: datetime, months: int) -> List[Tuple[datetime, datetime]]:
    """Yield (start_inclusive, end_exclusive) chunks of size `months` (aligned to month starts)."""
    if since_dt.tzinfo is None: since_dt = since_dt.replace(tzinfo=timezone.utc)
    if until_dt.tzinfo is None: until_dt = until_dt.replace(tzinfo=timezone.utc)
    cur = since_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    ranges = []
    while cur <= until_dt:
        nxt = cur + relativedelta(months=months)
        start = max(cur, since_dt)
        end_excl = min(nxt, until_dt + relativedelta(days=1))  # 'before:' is exclusive
        ranges.append((start, end_excl))
        cur = nxt
    return ranges

def iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()

# ------------------------------- Main -----------------------------------------

def main():
    ap = argparse.ArgumentParser(description="S&P 500 Google News RSS → CSV (per-ticker flush + resume)")
    ap.add_argument("--symbols-file", required=True, help="Path to SP500_2025.csv (has 'Symbol' column)")
    ap.add_argument("-o", "--output", default="headlines.csv", help="Output CSV path")
    ap.add_argument("--since", default="2006-01-01", help="Start date YYYY-MM-DD (default 2006-01-01)")
    ap.add_argument("--until", help="End date YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--chunk-months", type=int, default=3, help="Chunk size in months (3=quarter, 1=month, 6=half-year)")
    ap.add_argument("--sleep", type=float, default=0.6, help="Seconds to sleep between chunk requests")
    ap.add_argument("--max-per-chunk", type=int, default=1000, help="Safety cap per chunk (rarely hit)")
    ap.add_argument("--append", action="store_true", help="Append to output; de-dup using existing rows")
    ap.add_argument("--ticker-limit", type=int, default=0, help="Process only first N tickers (0=all)")
    ap.add_argument("--year-progress", action="store_true", help="Print yearly summary lines per ticker")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output")
    ap.add_argument("--state-file", help="Path to resume state JSON (default: <output>.state.json)")
    ap.add_argument("--resume", action="store_true", help="Skip tickers already marked completed in state file")
    args = ap.parse_args()

    # Dates
    try:
        since_dt = dateparser.parse(args.since).astimezone(timezone.utc)
    except Exception:
        print("Invalid --since; use YYYY-MM-DD", file=sys.stderr); sys.exit(2)
    if args.until:
        try:
            until_dt = dateparser.parse(args.until).astimezone(timezone.utc)
        except Exception:
            print("Invalid --until; use YYYY-MM-DD", file=sys.stderr); sys.exit(2)
    else:
        until_dt = datetime.now(timezone.utc)

    # Tickers
    try:
        tickers = read_tickers_from_csv(args.symbols_file)
    except Exception as e:
        print(f"[error] reading symbols: {e}", file=sys.stderr); sys.exit(2)
    if args.ticker_limit and args.ticker_limit > 0:
        tickers = tickers[:args.ticker_limit]

    # Output header setup
    if args.append:
        ensure_output_header(args.output)
    else:
        # fresh file
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["ticker","date","title","url"])
            w.writeheader()

    # Seen set & baseline totals
    seen = load_seen_from_csv(args.output) if args.append else set()
    overall_total = len(seen)  # approximate count for progress

    # State (resume)
    state_path = args.state_file or (args.output + ".state.json")
    state = load_state(state_path)
    completed = set(state.get("completed", [])) if args.resume else set()

    # Process tickers
    for t_idx, tk in enumerate(tickers, 1):
        if tk in completed:
            if not args.quiet:
                eprint(f"[{t_idx}/{len(tickers)}] {tk}: already completed (resume)")
            continue

        ranges = chunk_ranges(since_dt, until_dt, max(1, args.chunk_months))
        if not args.quiet:
            eprint(f"[{t_idx}/{len(tickers)}] {tk}: {len(ranges)} chunks "
                   f"{ranges[0][0].date()}→{(ranges[-1][1]-relativedelta(days=1)).date()}")

        ticker_rows: List[Dict[str,str]] = []
        ticker_total = 0
        cur_year = None
        year_count = 0

        for i, (start, end_excl) in enumerate(ranges, 1):
            q = f'{tk} after:{start.date()} before:{end_excl.date()}'
            try:
                xml = fetch_rss(q)
                items = parse_items(xml)
            except Exception as e:
                if not args.quiet:
                    eprint(f"[warn] {tk} {start.date()}→{(end_excl-relativedelta(days=1)).date()}: {e}")
                items = []

            items = [it for it in items if start <= it["date"] < end_excl]
            items = sorted(items, key=lambda x: x["date"], reverse=True)[:args.max_per_chunk]

            new_items = 0
            for it in items:
                key = (it["title"], it["url"])
                if key in seen:
                    continue
                seen.add(key)
                row = {"ticker": tk.upper(), "date": iso(it["date"]), "title": it["title"], "url": it["url"]}
                ticker_rows.append(row)
                new_items += 1
                ticker_total += 1
                overall_total += 1

                if args.year_progress:
                    y = it["date"].year
                    if cur_year is None:
                        cur_year = y
                        year_count = 0
                    if y != cur_year:
                        eprint(f"[{tk}] YEAR {cur_year}: {year_count} (so far)")
                        cur_year = y
                        year_count = 0
                    year_count += 1

            if not args.quiet:
                pct = (i / len(ranges)) * 100.0
                window = f"{start.date()}→{(end_excl - relativedelta(days=1)).date()}"
                eprint(f"[{tk}] {i}/{len(ranges)} ({pct:5.1f}%) {window} | +{new_items} | ticker={ticker_total} | overall={overall_total}")

            time.sleep(args.sleep)

        if args.year_progress and cur_year is not None:
            eprint(f"[{tk}] YEAR {cur_year}: {year_count} (final for year)")

        # ---- Per-ticker FLUSH ----
        append_rows(args.output, ticker_rows)
        if not args.quiet:
            eprint(f"[{tk}] written {len(ticker_rows)} new rows to {args.output} (ticker total {ticker_total})")

        # Mark ticker completed in state and save
        completed.add(tk)
        save_state(state_path, sorted(completed))

    print(f"Done. Progress saved to {args.output} (state: {state_path})")

if __name__ == "__main__":
    main()

"""
NSE Stock Analyzer — Database Seeder v2
Run once on your PC to build nse_data.db, then upload it to GitHub.

Usage:
    python setup_db.py --top50        # 50 most popular stocks (~5-10 min)
    python setup_db.py --all          # all ~500 Nifty stocks (~60-90 min)
    python setup_db.py TCS INFY SBIN  # specific symbols only
    python setup_db.py                # same as --top50 (safe default)

The nse_data.db file is created in the same folder as this script.
Upload it to your GitHub backend repo alongside server.py.
"""

import sys, os, time, datetime, sqlite3, pathlib
import yfinance as yf
import pandas as pd

DB_PATH = pathlib.Path(__file__).parent / "nse_data.db"

# ── Top 50 most-searched NSE symbols ─────────────────────────────────────────
TOP50 = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC","SBIN",
    "BHARTIARTL","KOTAKBANK","LT","AXISBANK","ASIANPAINT","HCLTECH","BAJFINANCE",
    "TITAN","MARUTI","SUNPHARMA","WIPRO","POWERGRID","NTPC","ONGC","ULTRACEMCO",
    "BAJAJFINSV","NESTLEIND","TECHM","HDFCLIFE","ADANIENT","JSWSTEEL","TATASTEEL",
    "COALINDIA","DRREDDY","CIPLA","DIVISLAB","GRASIM","BPCL","HEROMOTOCO",
    "BRITANNIA","EICHERMOT","TATAMOTORS","INDUSINDBK","SBILIFE","MM","HINDALCO",
    "VEDL","TATACONSUM","APOLLOHOSP","PIDILITIND","DABUR","BERGEPAINT",
]

# ── Full Nifty 500 (cleaned, deduplicated) ────────────────────────────────────
ALL_SYMBOLS = list(dict.fromkeys([
    # Nifty 50
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC","SBIN",
    "BHARTIARTL","KOTAKBANK","LT","AXISBANK","ASIANPAINT","HCLTECH","BAJFINANCE",
    "TITAN","MARUTI","SUNPHARMA","WIPRO","POWERGRID","NTPC","ONGC","ULTRACEMCO",
    "BAJAJFINSV","NESTLEIND","TECHM","HDFCLIFE","ADANIENT","JSWSTEEL","TATASTEEL",
    "COALINDIA","DRREDDY","CIPLA","DIVISLAB","GRASIM","BPCL","HEROMOTOCO",
    "BRITANNIA","EICHERMOT","TATAMOTORS","INDUSINDBK","SBILIFE","MM","HINDALCO",
    "VEDL","TATACONSUM","APOLLOHOSP","PIDILITIND","DABUR","BERGEPAINT",
    # Nifty Next 50
    "ADANIPORTS","ADANIGREEN","AMBUJACEM","AUROPHARMA","BANDHANBNK","BANKBARODA",
    "BEL","BHEL","BIOCON","BOSCHLTD","CANBK","CHOLAFIN","COLPAL","CONCOR",
    "COROMANDEL","CUMMINSIND","DEEPAKNTR","DLF","ESCORTS","EXIDEIND",
    "FEDERALBNK","FORTIS","GAIL","GODREJCP","GODREJPROP","HAL","HAVELLS",
    "HINDPETRO","IDFCFIRSTB","IGL","INDHOTEL","INDUSTOWER","IRCTC",
    "JKCEMENT","JUBLFOOD","KANSAINER","LTF","LUPIN","MARICO","METROPOLIS",
    "MFSL","MOTHERSON","MPHASIS","MRF","MUTHOOTFIN","NAUKRI","NMDC",
    "OBEROIRLTY","OIL","PAGEIND","PEL","PERSISTENT","PETRONET","PFIZER",
    "PFC","PNB","POLYCAB","RAIN","RAMCOCEM","RECLTD","SAIL","SCHAEFFLER",
    "SRF","SUNDRMFAST","SUPREMEIND","SUZLON","SYNGENE","TATACHEM","TATACOMM",
    "TATAELXSI","TATAPOWER","TRENT","TRIDENT","TVSMOTORS","UBL","UNIONBANK",
    "UPL","VOLTAS","ZEEL","ZOMATO","DELHIVERY","NYKAA","POLICYBZR",
    # Additional Nifty 500
    "ANGELONE","APTUS","BAJAJ-AUTO","BALKRISIND","BATAINDIA","CAMS","CAMPUS",
    "CANFINHOME","CDSL","CGPOWER","CLEAN","CROMPTON","CUB","CESC","DATAPATTNS",
    "DOMS","DBREALTY","EMUDHRA","GLAXO","GMRINFRA","GRANULES","HONAUT","HUDCO",
    "ISEC","LATENTVIEW","LALPATHLAB","LTIM","MEDPLUS","NAZARA","NAVINFLUOR",
    "PRAJIND","SANOFI","SANSERA","SAPPHIRE","STARHEALTH","CENTURYTEX",
    "CRAFTSMAN","CCL","CHALET","DEEPAKFERT","DELTACORP","DIXONS","ELGIEQUIP",
    "ESTER","FINEORG","GLENMARK","GRINDWELL","HSCL","IPCALAB","JYOTHYLAB",
    "KAJARIACER","KPITTECH","LAURUSLABS","LEMONTREE","LUXIND","MAHLOG",
    "MAHSEAMLES","MCDOWELL-N","MINDTREE","NATIONALUM","NBCC","NFL","NOCIL",
    "OFSS","ORIENTELEC","ORIENTCEM","PAYTM","PIIND","POLYMED","PRSMJOHNSN",
    "RAJRATAN","RELAXO","SAFARI","SAREGAMA","SHRIRAMFIN","SIEMENS","SOBHA",
    "SOLARA","SPANDANA","STAR","STOVEKRAFT","SUBEXLTD","SUMICHEM","SUNTVNETWORK",
    "SUVENPHAR","TANLA","THYROCARE","TTKPRESTIG","UNITDSPR","VAKRANGEE",
    "VARROC","VGUARD","VSTIND","WHIRLPOOL","WOCKPHARMA","ZENSARTECH","ZYDUSLIFE",
    "ADANITRANS","ATGL","HINDCOPPER","IOC","IRFC","RVNL","SJVN","NHPC",
    "PRESTIGE","PHOENIXLTD","NIACL","MAXHEALTH","KFINTECH","JSWENERGY",
    "INOXWIND","GPIL","GEPIL","FIVESTAR","FAIRCHEMOR","EQUITASBNK","EMAMILTD",
    "DELHIVERY","DCMSHRIRAM","DBCORP","COSMOFIRST","CAMPUS","CEATLTD","BRIGADE",
    "ABFRL","AAVAS","ABBE","ROUTE","RADICO","QUESS","POONAWALLA",
]))


def get_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    return conn


def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT NOT NULL, date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL NOT NULL, volume INTEGER,
            PRIMARY KEY (symbol, date)
        );
        CREATE TABLE IF NOT EXISTS fundamentals (
            symbol TEXT PRIMARY KEY, fetched_date TEXT,
            pe REAL, forward_pe REAL, revenue_growth REAL,
            roe REAL, debt_equity REAL, insider_pct REAL, inst_pct REAL,
            sector TEXT, industry TEXT, high52 REAL, low52 REAL,
            mkt_price REAL, prev_close REAL, day_open REAL,
            day_high REAL, day_low REAL, avg_volume INTEGER
        );
        CREATE TABLE IF NOT EXISTS history_meta (
            symbol TEXT PRIMARY KEY, oldest_date TEXT, newest_date TEXT,
            last_price_sync TEXT, last_fund_sync TEXT, last_vol_sync TEXT
        );
        CREATE TABLE IF NOT EXISTS prices_live (
            symbol TEXT NOT NULL, ts INTEGER NOT NULL,
            price REAL NOT NULL, volume INTEGER,
            PRIMARY KEY (symbol, ts)
        );
        CREATE TABLE IF NOT EXISTS volume_hourly (
            symbol TEXT NOT NULL, hour TEXT NOT NULL, volume INTEGER,
            PRIMARY KEY (symbol, hour)
        );
        CREATE INDEX IF NOT EXISTS idx_prices_sym_date ON prices(symbol, date);
    """)
    conn.commit()


def yf_fetch(func, sym="", retries=4, base_delay=5):
    """Fetch with exponential retry. Returns None on total failure."""
    for attempt in range(retries):
        try:
            result = func()
            if isinstance(result, pd.DataFrame) and result.empty and attempt < retries-1:
                raise ValueError("empty result")
            return result
        except Exception as e:
            wait = base_delay * (2 ** attempt)
            print(f"    [retry {attempt+1}/{retries}] {sym}: {type(e).__name__} — waiting {wait}s")
            time.sleep(wait)
    return None


def seed_prices(sym, conn, today, start_date):
    """Download 10-year daily OHLCV and store in DB. Returns row count."""
    # Check if already done today
    row = conn.execute(
        "SELECT newest_date FROM history_meta WHERE symbol=?", (sym,)
    ).fetchone()
    if row and row[0] == today:
        return -1  # already current, skip

    # If we have some data, only fetch what's missing
    existing_newest = row[0] if row else None
    fetch_from = existing_newest if existing_newest else start_date
    fetch_to   = str(datetime.date.today() + datetime.timedelta(days=1))

    df = yf_fetch(
        lambda: yf.Ticker(f"{sym}.NS").history(
            start=fetch_from, end=fetch_to, interval="1d", auto_adjust=True
        ),
        sym=sym
    )

    if df is None or df.empty:
        return 0

    rows = []
    for idx, r in df.iterrows():
        c = float(r.get("Close", 0) or 0)
        if c <= 0:
            continue
        rows.append((
            sym, str(idx.date()),
            round(float(r.get("Open",   0) or 0), 2),
            round(float(r.get("High",   0) or 0), 2),
            round(float(r.get("Low",    0) or 0), 2),
            round(c, 2),
            int(r.get("Volume", 0) or 0),
        ))

    if not rows:
        return 0

    conn.executemany(
        "INSERT OR REPLACE INTO prices(symbol,date,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?)",
        rows
    )
    oldest = conn.execute(
        "SELECT MIN(date) FROM prices WHERE symbol=?", (sym,)
    ).fetchone()[0] or rows[0][1]

    conn.execute("""
        INSERT OR REPLACE INTO history_meta(symbol, oldest_date, newest_date, last_price_sync)
        VALUES (?, ?, ?, ?)
    """, (sym, oldest, rows[-1][1], today))
    conn.commit()
    return len(rows)


def seed_fundamentals(sym, conn, today):
    """Fetch and store fundamentals. Skips if already done today."""
    row = conn.execute(
        "SELECT fetched_date FROM fundamentals WHERE symbol=?", (sym,)
    ).fetchone()
    if row and row[0] == today:
        return False

    info = yf_fetch(
        lambda: yf.Ticker(f"{sym}.NS").info or {},
        sym=sym, retries=3, base_delay=4
    ) or {}

    def safe(k, d=None):
        v = info.get(k)
        return v if (v is not None and v != "N/A" and v != 0) else d

    # Revenue growth via financials table
    rev_growth = None
    try:
        fin = yf_fetch(
            lambda: yf.Ticker(f"{sym}.NS").financials,
            sym=sym, retries=2, base_delay=4
        )
        if fin is not None and not fin.empty:
            for k in ["Total Revenue", "Revenue", "TotalRevenue"]:
                if k in fin.index:
                    rv = fin.loc[k].dropna().sort_index()
                    if len(rv) >= 2:
                        rev_growth = round(
                            float((rv.iloc[-1] - rv.iloc[0]) / abs(rv.iloc[0]) * 100), 1
                        )
                    break
    except Exception:
        pass

    conn.execute("""
        INSERT OR REPLACE INTO fundamentals(
            symbol, fetched_date, pe, forward_pe, revenue_growth,
            roe, debt_equity, insider_pct, inst_pct,
            sector, industry, high52, low52,
            mkt_price, prev_close, day_open, day_high, day_low, avg_volume
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        sym, today,
        safe("trailingPE"), safe("forwardPE"), rev_growth,
        safe("returnOnEquity"), safe("debtToEquity"),
        safe("heldPercentInsiders"), safe("heldPercentInstitutions"),
        safe("sector", "N/A"), safe("industry", "N/A"),
        safe("fiftyTwoWeekHigh"), safe("fiftyTwoWeekLow"),
        safe("regularMarketPrice") or safe("currentPrice"),
        safe("previousClose"), safe("open"),
        safe("dayHigh"), safe("dayLow"), safe("averageVolume"),
    ))
    # Update last_fund_sync in meta
    conn.execute("""
        INSERT INTO history_meta(symbol, oldest_date, newest_date, last_fund_sync)
        VALUES(?,
            COALESCE((SELECT oldest_date FROM history_meta WHERE symbol=?), ''),
            COALESCE((SELECT newest_date FROM history_meta WHERE symbol=?), ''),
            ?)
        ON CONFLICT(symbol) DO UPDATE SET last_fund_sync = excluded.last_fund_sync
    """, (sym, sym, sym, today))
    conn.commit()
    return True


def main():
    args = sys.argv[1:]

    if "--all" in args:
        symbols = ALL_SYMBOLS
        mode = "all"
    elif "--top50" in args or not args:
        symbols = TOP50
        mode = "top50"
    else:
        symbols = [s.upper().strip() for s in args if not s.startswith("--")]
        mode = "custom"

    today      = str(datetime.date.today())
    start_date = str(datetime.date.today() - datetime.timedelta(days=365 * 10 + 5))

    print("=" * 60)
    print(f"  NSE Stock Analyzer — Database Seeder")
    print(f"  Mode     : {mode} ({len(symbols)} symbols)")
    print(f"  Period   : {start_date}  to  {today}  (10 years)")
    print(f"  Database : {DB_PATH}")
    print("=" * 60)
    print()

    conn = get_db()
    init_db(conn)

    ok = skipped = failed = 0
    total_rows = 0
    t0 = time.time()

    for i, sym in enumerate(symbols):
        print(f"[{i+1:3d}/{len(symbols)}] {sym:<20}", end="", flush=True)

        # ── Prices ───────────────────────────────────────────────
        try:
            n = seed_prices(sym, conn, today, start_date)
            if n == -1:
                print(f"  prices: already current  ", end="")
                skipped += 1
            elif n == 0:
                print(f"  prices: no data          ", end="")
                failed += 1
            else:
                print(f"  prices: {n:5d} rows stored", end="")
                total_rows += n
                ok += 1
        except Exception as e:
            print(f"  prices: ERROR {e}         ", end="")
            failed += 1

        # ── Fundamentals ─────────────────────────────────────────
        try:
            did = seed_fundamentals(sym, conn, today)
            print("  fundamentals: OK" if did else "  fundamentals: cached", flush=True)
        except Exception as e:
            print(f"  fundamentals: SKIP ({type(e).__name__})", flush=True)

        # Rate-limit pause every 10 symbols
        if (i + 1) % 10 == 0 and (i + 1) < len(symbols):
            elapsed  = time.time() - t0
            remaining = len(symbols) - (i + 1)
            rate     = (i + 1) / elapsed if elapsed > 0 else 1
            eta_min  = remaining / rate / 60
            db_mb    = round(DB_PATH.stat().st_size / 1024 / 1024, 1) if DB_PATH.exists() else 0
            print(f"\n  --- Progress: {i+1}/{len(symbols)} | OK:{ok} Skip:{skipped} Fail:{failed} | DB:{db_mb} MB | ETA:{eta_min:.1f} min ---\n")
            time.sleep(1)  # brief pause to avoid rate limiting

    conn.close()

    elapsed_min = (time.time() - t0) / 60
    db_mb = round(DB_PATH.stat().st_size / 1024 / 1024, 1) if DB_PATH.exists() else 0

    print()
    print("=" * 60)
    print(f"  Done in {elapsed_min:.1f} minutes")
    print(f"  Stored : {ok} symbols, {total_rows:,} price rows")
    print(f"  Skipped: {skipped} (already up to date)")
    print(f"  Failed : {failed}")
    print(f"  DB size: {db_mb} MB  →  {DB_PATH}")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Upload nse_data.db to your GitHub backend repo")
    print("  2. Render will auto-redeploy and use this database")
    print("  3. Run this script again anytime to refresh the data")
    print()

    # Quick DB health check
    conn2 = get_db()
    total = conn2.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    syms  = conn2.execute("SELECT COUNT(DISTINCT symbol) FROM prices").fetchone()[0]
    conn2.close()
    print(f"  DB health: {total:,} total price rows across {syms} symbols")


if __name__ == "__main__":
    main()

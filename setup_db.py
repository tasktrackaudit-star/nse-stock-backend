"""
NSE Stock Analyzer — Database Seeder
Run this ONCE to pre-download 10 years of data for all Nifty 500 stocks.

Usage:
    python setup_db.py              # seeds all 500 stocks (takes ~30-60 min)
    python setup_db.py TCS INFY     # seeds specific symbols only
    python setup_db.py --top50      # seeds the 50 most popular symbols

After seeding, the nse_data.db file (~300-600 MB) is created next to server.py.
The server will then serve all data from this local DB with near-instant response.

You can re-run at any time to top-up the database with the latest prices.
"""

import sys, os, time, datetime, sqlite3, pathlib, threading
import yfinance as yf
import pandas as pd

DB_PATH  = pathlib.Path(__file__).parent / "nse_data.db"
DB_LOCK  = threading.Lock()

# Top 50 most-used symbols (if --top50 flag)
TOP50 = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC","SBIN",
    "BHARTIARTL","KOTAKBANK","LT","AXISBANK","ASIANPAINT","HCLTECH","BAJFINANCE",
    "TITAN","MARUTI","SUNPHARMA","WIPRO","POWERGRID","NTPC","ONGC","ULTRACEMCO",
    "BAJAJFINSV","NESTLEIND","TECHM","HDFCLIFE","ADANIENT","JSWSTEEL","TATASTEEL",
    "COALINDIA","DRREDDY","CIPLA","DIVISLAB","GRASIM","BPCL","HEROMOTOCO",
    "BRITANNIA","EICHERMOT","TATAMOTORS","INDUSINDBK","SBILIFE","M&M","HINDALCO",
    "VEDL","TATACONSUM","APOLLOHOSP","PIDILITIND","DABUR","BERGEPAINT"
]

# Full Nifty 500 list (abbreviated -- add full list from NSE website if needed)
NIFTY500_SYMBOLS = TOP50 + [
    "ADANIPORTS","ADANIGREEN","ADANITRANS","ADANIGAS","ADANIPOWER","ATGL",
    "AMBUJACEM","AUROPHARMA","BANDHANBNK","BANKBARODA","BANKINDIA","BEL",
    "BHEL","BIOCON","BOSCHLTD","CANBK","CHOLAFIN","COLPAL","CONCOR",
    "COROMANDEL","CUMMINSIND","DEEPAKNTR","DLF","ESCORTS","EXIDEIND",
    "FEDERALBNK","FORTIS","GAIL","GLAXO","GMRINFRA","GODREJCP","GODREJPROP",
    "GRANULES","HAL","HAVELLS","HINDPETRO","HONAUT","HUDCO","IDFCFIRSTB",
    "IGL","INDHOTEL","INDIGOPNTS","INDUSTOWER","IRCTC","ISEC","JKCEMENT",
    "JUBLFOOD","KANSAINER","LTF","LALPATHLAB","LTIM","LUPIN","MARICO",
    "MCDOWELL-N","METROPOLIS","MFSL","MOTHERSON","MPHASIS","MRF","MUTHOOTFIN",
    "NAUKRI","NAVINFLUOR","NMDC","OBEROIRLTY","OIL","PAGEIND","PEL",
    "PERSISTENT","PETRONET","PFIZER","PFC","PNB","POLYCAB","PRAJIND",
    "RAIN","RAMCOCEM","RECLTD","SAIL","SANOFI","SCHAEFFLER","SRF",
    "STAR","SUNDRMFAST","SUPREMEIND","SUZLON","SYNGENE","TATACHEM",
    "TATACOMM","TATAELXSI","TATAINVEST","TATAPOWER","TRENT","TRIDENT",
    "TTKPRESTIG","TVSMOTORS","UBL","UNIONBANK","UNITDSPR","UPL","VAKRANGEE",
    "VOLTAS","WHIRLPOOL","ZEEL","ZOMATO","PAYTM","NYKAA","DELHIVERY",
    "POLICYBZR","CARTRADE","NAZARA","EASEMYTRIP","SIGACHI","LATENTVIEW",
    "EMUDHRA","MEDPLUS","SANSERA","SAPPHIRE","STARHEALTH","GLENMARK",
    "IPCALAB","ALKEM","ABBOTINDIA","APLAPOLLO","ANGELONE","APTUS",
    "BAJAJ-AUTO","BALKRISIND","BATAINDIA","BBTC","CAMS","CAMPUS",
    "CANFINHOME","CCL","CDSL","CENTURYTEX","CGPOWER","CHALET","CLEAN",
    "CRAFTSMAN","CROMPTON","CUB","CESC","DATAPATTNS","DOMS","DBREALTY",
]

def get_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30)
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
        CREATE INDEX IF NOT EXISTS idx_prices_sym_date ON prices(symbol, date);
    """)
    conn.commit()

def fetch_with_retry(func, sym, retries=5, base_delay=5):
    for attempt in range(retries):
        try:
            result = func()
            if isinstance(result, pd.DataFrame) and result.empty and attempt < retries-1:
                raise ValueError("empty")
            return result
        except Exception as e:
            wait = base_delay * (2 ** attempt)
            print(f"    [retry {attempt+1}/{retries}] {sym}: {e} — waiting {wait}s")
            time.sleep(wait)
    return None

def seed_symbol(sym, conn, today, start_date):
    # Check if already done today
    row = conn.execute("SELECT newest_date FROM history_meta WHERE symbol=?", (sym,)).fetchone()
    if row and row[0] == today:
        print(f"  ✓ {sym} already up to date")
        return True

    existing_start = row[0] if row else None
    fetch_start    = existing_start or start_date
    fetch_end      = str(datetime.date.today() + datetime.timedelta(days=1))

    print(f"  → {sym}: fetching {fetch_start} to {fetch_end}...")

    df = fetch_with_retry(
        lambda: yf.Ticker(f"{sym}.NS").history(start=fetch_start, end=fetch_end, interval="1d", auto_adjust=True),
        sym
    )

    if df is None or df.empty:
        print(f"  ✗ {sym}: no data returned")
        return False

    rows = []
    for idx, r in df.iterrows():
        c = float(r.get("Close", 0) or 0)
        if c <= 0: continue
        rows.append((sym, str(idx.date()),
                     round(float(r.get("Open",   0) or 0), 2),
                     round(float(r.get("High",   0) or 0), 2),
                     round(float(r.get("Low",    0) or 0), 2),
                     round(c, 2), int(r.get("Volume", 0) or 0)))

    if not rows:
        print(f"  ✗ {sym}: all rows were null")
        return False

    with DB_LOCK:
        conn.executemany(
            "INSERT OR REPLACE INTO prices(symbol,date,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?)",
            rows
        )
        conn.execute("""
            INSERT OR REPLACE INTO history_meta(symbol,oldest_date,newest_date,last_price_sync)
            VALUES(?,?,?,?)
        """, (sym, rows[0][1], rows[-1][1], today))
        conn.commit()

    print(f"  ✓ {sym}: {len(rows)} rows ({rows[0][1]} to {rows[-1][1]})")
    return True

def seed_fundamentals(sym, conn, today):
    row = conn.execute("SELECT fetched_date FROM fundamentals WHERE symbol=?", (sym,)).fetchone()
    if row and row[0] == today:
        return  # already done today

    info = fetch_with_retry(lambda: yf.Ticker(f"{sym}.NS").info or {}, sym, retries=3, base_delay=4) or {}

    def safe(k, d=None):
        v = info.get(k)
        return v if (v is not None and v != "N/A" and v != 0) else d

    rev_growth = None
    try:
        fin = fetch_with_retry(lambda: yf.Ticker(f"{sym}.NS").financials, sym, retries=2, base_delay=4)
        if fin is not None and not fin.empty:
            for k in ["Total Revenue", "Revenue", "TotalRevenue"]:
                if k in fin.index:
                    rv = fin.loc[k].dropna().sort_index()
                    if len(rv) >= 2:
                        rev_growth = round(float((rv.iloc[-1]-rv.iloc[0])/abs(rv.iloc[0])*100), 1)
                    break
    except Exception: pass

    with DB_LOCK:
        conn.execute("""INSERT OR REPLACE INTO fundamentals(
            symbol,fetched_date,pe,forward_pe,revenue_growth,roe,debt_equity,
            insider_pct,inst_pct,sector,industry,high52,low52,
            mkt_price,prev_close,day_open,day_high,day_low,avg_volume)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sym, today, safe("trailingPE"), safe("forwardPE"), rev_growth,
             safe("returnOnEquity"), safe("debtToEquity"),
             safe("heldPercentInsiders"), safe("heldPercentInstitutions"),
             safe("sector","N/A"), safe("industry","N/A"),
             safe("fiftyTwoWeekHigh"), safe("fiftyTwoWeekLow"),
             safe("regularMarketPrice") or safe("currentPrice"),
             safe("previousClose"), safe("open"),
             safe("dayHigh"), safe("dayLow"), safe("averageVolume")))
        conn.execute("""
            INSERT INTO history_meta(symbol,oldest_date,newest_date,last_fund_sync)
            VALUES(?,COALESCE((SELECT oldest_date FROM history_meta WHERE symbol=?),''),
                     COALESCE((SELECT newest_date FROM history_meta WHERE symbol=?),''),?)
            ON CONFLICT(symbol) DO UPDATE SET last_fund_sync=excluded.last_fund_sync
        """, (sym, sym, sym, today))
        conn.commit()


def main():
    args = sys.argv[1:]

    if "--top50" in args:
        symbols = TOP50
        print(f"Seeding top {len(symbols)} symbols...")
    elif args:
        symbols = [s.upper() for s in args if not s.startswith("--")]
        print(f"Seeding {len(symbols)} specified symbols...")
    else:
        symbols = list(dict.fromkeys(NIFTY500_SYMBOLS))  # deduplicated
        print(f"Seeding all {len(symbols)} Nifty 500 symbols...")

    today      = str(datetime.date.today())
    start_date = str(datetime.date.today() - datetime.timedelta(days=365*10+5))

    print(f"Database : {DB_PATH}")
    print(f"Period   : {start_date}  to  {today}  (10 years)")
    print(f"Symbols  : {len(symbols)}")
    print("="*60)

    conn = get_db()
    init_db(conn)

    ok = fail = 0
    t0 = time.time()

    for i, sym in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] {sym}")
        try:
            if seed_symbol(sym, conn, today, start_date):
                seed_fundamentals(sym, conn, today)
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            fail += 1

        # Rate-limit: brief pause every 10 symbols to avoid Yahoo Finance throttle
        if (i+1) % 10 == 0:
            elapsed = time.time() - t0
            remaining = len(symbols) - (i+1)
            rate = (i+1) / elapsed * 60  # per minute
            eta  = remaining / (rate/60) if rate > 0 else 0
            db_mb = round(DB_PATH.stat().st_size/1024/1024, 1) if DB_PATH.exists() else 0
            print(f"\n  Progress: {i+1}/{len(symbols)} | OK:{ok} Fail:{fail} | DB:{db_mb}MB | ETA:{eta/60:.1f}min\n")
            time.sleep(2)

    conn.close()
    db_mb = round(DB_PATH.stat().st_size/1024/1024, 1) if DB_PATH.exists() else 0
    elapsed_min = (time.time()-t0)/60

    print("="*60)
    print(f"  Done in {elapsed_min:.1f} minutes")
    print(f"  Success: {ok}  |  Failed: {fail}")
    print(f"  Database size: {db_mb} MB  at  {DB_PATH}")
    print("="*60)
    print("\nYou can now start server.py — all data will load from the local database.")

if __name__ == "__main__":
    main()

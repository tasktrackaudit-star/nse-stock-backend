"""
NSE Stock Analyzer — Server v7
Fixes applied:
  1. ThreadingHTTPServer — handles multiple requests concurrently
  2. BrokenPipeError / connection-reset errors caught globally — no more crashes
  3. ensure_history runs in BACKGROUND thread — requests never block waiting for yfinance
  4. In-memory cache (_mem_cache) — second request for same symbol is instant (no DB query)
  5. Render ephemeral-filesystem fix — DB copied to /tmp on startup
  6. yf_fetch timeout per attempt capped at 25s — avoids indefinite hangs
  7. All exceptions in handlers are caught including BrokenPipe
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
import json, traceback, warnings, datetime, os, math, sqlite3, threading, time, pathlib, socket

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PORT = int(os.environ.get("PORT", 7777))

# ── DB path ───────────────────────────────────────────────────────────────────
_here   = pathlib.Path(__file__).parent
DB_PATH = _here / "nse_data.db"

# On Render: copy bundled DB to /tmp (repo is read-only at runtime)
_render = os.environ.get("RENDER", "")
if _render:
    import shutil
    _tmp_db = pathlib.Path("/tmp/nse_data.db")
    if DB_PATH.exists() and (not _tmp_db.exists() or _tmp_db.stat().st_size < DB_PATH.stat().st_size):
        print(f"[DB] Copying bundled DB to /tmp ({round(DB_PATH.stat().st_size/1024/1024,1)} MB)...")
        shutil.copy2(str(DB_PATH), str(_tmp_db))
        print("[DB] Copy complete.")
    DB_PATH = _tmp_db

# ── yfinance ──────────────────────────────────────────────────────────────────
try:
    import yfinance as yf
except ImportError:
    print("[ERROR] yfinance not installed. Run: pip install yfinance")
    raise SystemExit(1)

import numpy as np
import pandas as pd

# ── PyTorch LSTM ──────────────────────────────────────────────────────────────
TORCH_OK = False
try:
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler
    TORCH_OK = True
    print(f"[OK] PyTorch {torch.__version__} (LSTM ready)")
except ImportError:
    print("[WARN] PyTorch not installed")
except Exception as e:
    print(f"[WARN] PyTorch error: {e}")

# ── Prophet ───────────────────────────────────────────────────────────────────
PROPHET_OK = False
try:
    from prophet import Prophet
    PROPHET_OK = True
    print("[OK] Prophet loaded")
except Exception:
    try:
        from fbprophet import Prophet
        PROPHET_OK = True
    except Exception as e:
        print(f"[WARN] Prophet not installed ({e})")


# ════════════════════════════════════════════════════════════════════════════
# IN-MEMORY CACHE  — makes repeated requests instant
# ════════════════════════════════════════════════════════════════════════════
_mem_cache = {}          # key -> {"data": ..., "ts": unix_time}
_mem_lock  = threading.Lock()
MEM_TTL    = 300         # seconds before cache entry expires (5 min)

def cache_get(key):
    with _mem_lock:
        entry = _mem_cache.get(key)
        if entry and (time.time() - entry["ts"]) < MEM_TTL:
            return entry["data"]
    return None

def cache_set(key, data):
    with _mem_lock:
        _mem_cache[key] = {"data": data, "ts": time.time()}

def cache_invalidate(sym):
    with _mem_lock:
        keys = [k for k in _mem_cache if k.startswith(sym + ":")]
        for k in keys:
            del _mem_cache[k]


# ════════════════════════════════════════════════════════════════════════════
# DATABASE
# ════════════════════════════════════════════════════════════════════════════
DB_LOCK = threading.Lock()

def get_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-32000")
    return conn

def init_db():
    with DB_LOCK:
        conn = get_db()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS prices (
                symbol TEXT NOT NULL, date TEXT NOT NULL,
                open REAL, high REAL, low REAL, close REAL NOT NULL, volume INTEGER,
                PRIMARY KEY (symbol, date)
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
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT UNIQUE NOT NULL,
                email       TEXT UNIQUE NOT NULL,
                password    TEXT NOT NULL,
                role        TEXT NOT NULL DEFAULT 'user',
                status      TEXT NOT NULL DEFAULT 'pending',
                created_at  TEXT NOT NULL,
                approved_at TEXT,
                reset_token TEXT,
                reset_expiry TEXT
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_prices_sym_date ON prices(symbol, date);
            CREATE INDEX IF NOT EXISTS idx_live_sym ON prices_live(symbol, ts);
        """)
        import hashlib as _hl
        admin_pw = _hl.sha256(b"admin123").hexdigest()
        now = str(datetime.datetime.utcnow())
        conn.execute("""INSERT OR IGNORE INTO users(username,email,password,role,status,created_at,approved_at)
            VALUES('admin','admin@nse.local',?,'admin','approved',?,?)""", (admin_pw, now, now))
        conn.commit()
        conn.close()
    print(f"[DB] Ready: {DB_PATH}  ({round(DB_PATH.stat().st_size/1024/1024,1) if DB_PATH.exists() else 0} MB)")

init_db()

# ════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ════════════════════════════════════════════════════════════════
import hashlib as _hl, secrets as _sec

def _hash_pw(pw): return _hl.sha256(pw.encode()).hexdigest()
def _make_token(): return _sec.token_hex(32)

def _read_body(handler):
    n = int(handler.headers.get("Content-Length", 0))
    return json.loads(handler.rfile.read(n)) if n else {}

def _get_session(handler):
    auth = handler.headers.get("Authorization","")
    if not auth.startswith("Bearer "): return None
    token = auth[7:]
    conn  = get_db()
    now   = str(datetime.datetime.utcnow())
    row   = conn.execute(
        "SELECT u.id,u.username,u.email,u.role,u.status FROM sessions s "
        "JOIN users u ON u.id=s.user_id WHERE s.token=? AND s.expires_at>?",
        (token, now)
    ).fetchone()
    conn.close()
    return {"id":row[0],"username":row[1],"email":row[2],"role":row[3],"status":row[4]} if row else None

def _require_auth(handler, admin=False):
    user = _get_session(handler)
    if not user:
        handler.send_json({"ok":False,"error":"Not authenticated"}, 401); return None
    if user["status"] != "approved":
        handler.send_json({"ok":False,"error":"Account pending admin approval"}, 403); return None
    if admin and user["role"] != "admin":
        handler.send_json({"ok":False,"error":"Admin only"}, 403); return None
    return user


# ════════════════════════════════════════════════════════════════════════════
# RETRY FETCH — capped per-attempt timeout, exponential back-off
# ════════════════════════════════════════════════════════════════════════════
def yf_fetch(func, retries=4, base_delay=3):
    """
    Call func() up to `retries` times.
    Each attempt is run in a thread with a 25-second hard timeout so a hung
    yfinance call can never block the server indefinitely.
    Returns None on total failure — callers fall back to cached DB data.
    """
    for attempt in range(retries):
        result_holder = [None]
        error_holder  = [None]

        def _run():
            try:
                result_holder[0] = func()
            except Exception as e:
                error_holder[0] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=25)   # 25s hard cap per attempt

        if t.is_alive():
            print(f"  [TIMEOUT] attempt {attempt+1}/{retries} — yfinance call exceeded 25s")
            wait = base_delay * (2 ** attempt)
            time.sleep(min(wait, 30))
            continue

        if error_holder[0] is not None:
            wait = base_delay * (2 ** attempt)
            print(f"  [RETRY {attempt+1}/{retries}] {type(error_holder[0]).__name__}: {error_holder[0]} — wait {wait}s")
            time.sleep(wait)
            continue

        result = result_holder[0]
        if isinstance(result, pd.DataFrame) and result.empty and attempt < retries-1:
            wait = base_delay * (2 ** attempt)
            print(f"  [RETRY {attempt+1}/{retries}] empty DataFrame — wait {wait}s")
            time.sleep(wait)
            continue

        return result

    print(f"  [FAIL] all {retries} attempts exhausted")
    return None


# ════════════════════════════════════════════════════════════════════════════
# BACKGROUND SEED TRACKER  — prevents duplicate concurrent seeds
# ════════════════════════════════════════════════════════════════════════════
_seeding_now  = set()   # symbols currently being seeded in background
_seed_lock    = threading.Lock()
_active_syms  = set()
_active_lock  = threading.Lock()

def register_symbol(sym):
    with _active_lock:
        _active_syms.add(sym)

def _seed_in_background(sym):
    """Kick off a background thread to seed 10yr history if not already running."""
    with _seed_lock:
        if sym in _seeding_now:
            return   # already seeding
        _seeding_now.add(sym)

    def _do():
        try:
            _seed_history(sym)
        finally:
            with _seed_lock:
                _seeding_now.discard(sym)

    threading.Thread(target=_do, daemon=True, name=f"seed-{sym}").start()


def _seed_history(sym):
    """Download and store 10yr daily OHLCV. Runs in background thread."""
    conn  = get_db()
    today = str(datetime.date.today())
    yesterday = str(datetime.date.today() - datetime.timedelta(days=1))

    row = conn.execute(
        "SELECT oldest_date, newest_date FROM history_meta WHERE symbol=?", (sym,)
    ).fetchone()
    row_count = conn.execute(
        "SELECT COUNT(*) FROM prices WHERE symbol=?", (sym,)
    ).fetchone()[0]

    # Already fresh — nothing to do
    if row and row_count >= 200 and row[1] >= yesterday:
        conn.close()
        return

    ten_yr_start = str(datetime.date.today() - datetime.timedelta(days=365*10+5))
    fetch_start  = row[1] if (row and row_count >= 200) else ten_yr_start
    fetch_end    = str(datetime.date.today() + datetime.timedelta(days=1))

    action = "incremental" if row and row_count >= 200 else "full 10yr"
    print(f"  [SEED] {sym}: {action} seed from {fetch_start}...")

    df = yf_fetch(
        lambda: yf.Ticker(f"{sym}.NS").history(
            start=fetch_start, end=fetch_end, interval="1d", auto_adjust=True
        )
    )

    if df is None or df.empty:
        print(f"  [SEED] {sym}: no data returned — will retry next request")
        conn.close()
        return

    rows = []
    for idx, r in df.iterrows():
        c = float(r.get("Close", 0) or 0)
        if c <= 0: continue
        rows.append((
            sym, str(idx.date()),
            round(float(r.get("Open",   0) or 0), 2),
            round(float(r.get("High",   0) or 0), 2),
            round(float(r.get("Low",    0) or 0), 2),
            round(c, 2), int(r.get("Volume", 0) or 0),
        ))

    if not rows:
        conn.close()
        return

    with DB_LOCK:
        conn.executemany(
            "INSERT OR REPLACE INTO prices(symbol,date,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?)",
            rows
        )
        oldest = conn.execute("SELECT MIN(date) FROM prices WHERE symbol=?", (sym,)).fetchone()[0] or rows[0][1]
        newest = rows[-1][1]
        conn.execute("""
            INSERT OR REPLACE INTO history_meta(symbol, oldest_date, newest_date, last_price_sync)
            VALUES (?, ?, ?, ?)
        """, (sym, oldest, newest, today))
        conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM prices WHERE symbol=?", (sym,)).fetchone()[0]
    print(f"  [SEED] {sym}: done — {total} total rows ({oldest} to {newest})")
    conn.close()

    # Invalidate mem cache so next request reads the fresh DB data
    cache_invalidate(sym)


# ════════════════════════════════════════════════════════════════════════════
# SCHEDULED SYNC
# ════════════════════════════════════════════════════════════════════════════
def sync_live_prices():
    with _active_lock:
        syms = list(_active_syms)
    if not syms:
        return
    print(f"  [SYNC] Live prices for {len(syms)} symbol(s)...")
    for sym in syms:
        try:
            df = yf_fetch(
                lambda s=sym: yf.Ticker(f"{s}.NS").history(period="2d", interval="1m", auto_adjust=True),
                retries=2, base_delay=3
            )
            if df is None or df.empty:
                continue

            ddf = yf_fetch(
                lambda s=sym: yf.Ticker(f"{s}.NS").history(period="2d", interval="1d", auto_adjust=True),
                retries=2, base_delay=2
            )
            today = str(datetime.date.today())
            conn  = get_db()
            live_rows = [
                (sym, int(idx.timestamp()), round(float(r.get("Close",0) or 0), 2), int(r.get("Volume",0) or 0))
                for idx, r in df.iterrows() if float(r.get("Close",0) or 0) > 0
            ]
            with DB_LOCK:
                if live_rows:
                    conn.executemany(
                        "INSERT OR REPLACE INTO prices_live(symbol,ts,price,volume) VALUES(?,?,?,?)",
                        live_rows
                    )
                if ddf is not None and not ddf.empty:
                    lr = ddf.iloc[-1]
                    conn.execute("""
                        INSERT OR REPLACE INTO prices(symbol,date,open,high,low,close,volume)
                        VALUES(?,?,?,?,?,?,?)
                    """, (
                        sym, today,
                        round(float(lr.get("Open",  0) or 0), 2),
                        round(float(lr.get("High",  0) or 0), 2),
                        round(float(lr.get("Low",   0) or 0), 2),
                        round(float(lr.get("Close", 0) or 0), 2),
                        int(lr.get("Volume", 0) or 0),
                    ))
                    conn.execute("""
                        INSERT INTO history_meta(symbol,oldest_date,newest_date,last_price_sync)
                        VALUES(?,COALESCE((SELECT oldest_date FROM history_meta WHERE symbol=?),'2016-01-01'),?,?)
                        ON CONFLICT(symbol) DO UPDATE SET newest_date=excluded.newest_date,
                                                          last_price_sync=excluded.last_price_sync
                    """, (sym, sym, today, today))
                conn.commit()
            conn.close()
            cache_invalidate(sym)   # fresh data → evict stale cache
        except Exception as e:
            print(f"  [SYNC] {sym}: {e}")

    # Prune live data older than 7 days
    try:
        cutoff = int((datetime.datetime.now() - datetime.timedelta(days=7)).timestamp())
        conn = get_db()
        with DB_LOCK:
            conn.execute("DELETE FROM prices_live WHERE ts < ?", (cutoff,))
            conn.commit()
        conn.close()
    except Exception:
        pass


def sync_fundamentals(sym, force=False):
    conn  = get_db()
    today = str(datetime.date.today())
    row   = conn.execute("SELECT fetched_date FROM fundamentals WHERE symbol=?", (sym,)).fetchone()
    if row and row[0] == today and not force:
        conn.close()
        return

    print(f"  [FUND] {sym}...")
    try:
        info = yf_fetch(lambda s=sym: yf.Ticker(f"{s}.NS").info or {}, retries=3, base_delay=4) or {}
        def safe(k, d=None):
            v = info.get(k)
            return v if (v is not None and v != "N/A" and v != 0) else d

        rev_growth = None
        try:
            fin = yf_fetch(lambda s=sym: yf.Ticker(f"{s}.NS").financials, retries=2, base_delay=4)
            if fin is not None and not fin.empty:
                for k in ["Total Revenue", "Revenue", "TotalRevenue"]:
                    if k in fin.index:
                        rv = fin.loc[k].dropna().sort_index()
                        if len(rv) >= 2:
                            rev_growth = round(float((rv.iloc[-1]-rv.iloc[0])/abs(rv.iloc[0])*100), 1)
                        break
        except Exception:
            pass

        with DB_LOCK:
            conn.execute("""
                INSERT OR REPLACE INTO fundamentals(
                    symbol,fetched_date,pe,forward_pe,revenue_growth,roe,debt_equity,
                    insider_pct,inst_pct,sector,industry,high52,low52,
                    mkt_price,prev_close,day_open,day_high,day_low,avg_volume)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                sym, today, safe("trailingPE"), safe("forwardPE"), rev_growth,
                safe("returnOnEquity"), safe("debtToEquity"),
                safe("heldPercentInsiders"), safe("heldPercentInstitutions"),
                safe("sector","N/A"), safe("industry","N/A"),
                safe("fiftyTwoWeekHigh"), safe("fiftyTwoWeekLow"),
                safe("regularMarketPrice") or safe("currentPrice"),
                safe("previousClose"), safe("open"),
                safe("dayHigh"), safe("dayLow"), safe("averageVolume"),
            ))
            conn.execute("""
                INSERT INTO history_meta(symbol,oldest_date,newest_date,last_fund_sync)
                VALUES(?,COALESCE((SELECT oldest_date FROM history_meta WHERE symbol=?),''),
                         COALESCE((SELECT newest_date FROM history_meta WHERE symbol=?),''),?)
                ON CONFLICT(symbol) DO UPDATE SET last_fund_sync=excluded.last_fund_sync
            """, (sym, sym, sym, today))
            conn.commit()
        conn.close()
        cache_invalidate(sym)
        print(f"  [FUND] {sym}: cached.")
    except Exception as e:
        print(f"  [FUND] {sym}: error — {e}")
        try: conn.close()
        except: pass


def _scheduler():
    last_vol   = 0
    last_fund_date = ""
    while True:
        time.sleep(300)
        try: sync_live_prices()
        except Exception as e: print(f"  [SCHED] price: {e}")

        now_ts = time.time()
        if now_ts - last_vol >= 3600:
            last_vol = now_ts
            try:
                with _active_lock:
                    syms = list(_active_syms)
                for sym in syms:
                    df = yf_fetch(
                        lambda s=sym: yf.Ticker(f"{s}.NS").history(period="1d", interval="60m", auto_adjust=True),
                        retries=2, base_delay=3
                    )
                    if df is None or df.empty: continue
                    conn = get_db()
                    rows = [(sym, idx.strftime("%Y-%m-%d %H:00"), int(r.get("Volume",0) or 0))
                            for idx, r in df.iterrows()]
                    with DB_LOCK:
                        conn.executemany("INSERT OR REPLACE INTO volume_hourly(symbol,hour,volume) VALUES(?,?,?)", rows)
                        conn.commit()
                    conn.close()
            except Exception as e:
                print(f"  [SCHED] vol: {e}")

        now_ist    = datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)
        today_str  = str(now_ist.date())
        if now_ist.hour >= 8 and last_fund_date != today_str:
            with _active_lock:
                syms = list(_active_syms)
            for s in syms:
                try: sync_fundamentals(s)
                except Exception as e: print(f"  [SCHED] fund {s}: {e}")
            last_fund_date = today_str

threading.Thread(target=_scheduler, daemon=True, name="Scheduler").start()
print("[OK] Scheduler started")


# ════════════════════════════════════════════════════════════════════════════
# DATA ACCESS  — mem cache → DB → live fallback
# ════════════════════════════════════════════════════════════════════════════
def _db_rows_to_response(sym, rows_db, fund):
    rows   = [{"date":r[0],"open":r[1] or 0,"high":r[2] or 0,"low":r[3] or 0,"close":r[4],"volume":r[5] or 0}
              for r in rows_db]
    closes = [r["close"] for r in rows]
    last   = closes[-1] if closes else 0
    prev   = closes[-2] if len(closes) > 1 else last
    info   = {
        "regularMarketPrice":   round(float(fund[0] or last), 2)               if fund else round(last, 2),
        "regularMarketOpen":    round(float(fund[1] or rows[-1]["open"]),  2)   if fund else 0,
        "regularMarketDayHigh": round(float(fund[2] or rows[-1]["high"]),  2)   if fund else 0,
        "regularMarketDayLow":  round(float(fund[3] or rows[-1]["low"]),   2)   if fund else 0,
        "regularMarketVolume":  int(fund[4] or rows[-1]["volume"])               if fund else 0,
        "fiftyTwoWeekHigh":     round(float(fund[5] or max(r["high"] for r in rows[-252:])), 2) if fund else 0,
        "fiftyTwoWeekLow":      round(float(fund[6] or min(r["low"]  for r in rows[-252:] if r["low"]>0)), 2) if fund else 0,
        "chartPreviousClose":   round(float(fund[7] or prev), 2)               if fund else round(prev, 2),
    }
    return {"ok": True, "rows": rows, "info": info}


def get_stock_data(sym, start, end):
    register_symbol(sym)
    cache_key = f"{sym}:{start}:{end}"

    # 1. Memory cache hit → instant
    cached = cache_get(cache_key)
    if cached:
        return cached

    conn = get_db()
    rows_db = conn.execute(
        "SELECT date,open,high,low,close,volume FROM prices "
        "WHERE symbol=? AND date>=? AND date<=? ORDER BY date ASC",
        (sym, start, end)
    ).fetchall()
    fund = conn.execute(
        "SELECT mkt_price,day_open,day_high,day_low,avg_volume,high52,low52,prev_close "
        "FROM fundamentals WHERE symbol=?", (sym,)
    ).fetchone()
    conn.close()

    # 2. DB has data → return it, kick background seed to fill gaps
    if rows_db:
        response = _db_rows_to_response(sym, rows_db, fund)
        cache_set(cache_key, response)
        # Kick background top-up if newest is stale
        _seed_in_background(sym)
        return response

    # 3. No DB data → live fetch NOW (first time for this symbol)
    print(f"  [LIVE] {sym}: no DB data, fetching live...")
    response = _live_fallback(sym, start, end)
    if response.get("ok"):
        cache_set(cache_key, response)
        # Also kick background for full 10yr seed
        _seed_in_background(sym)
    return response


def _live_fallback(sym, start, end):
    """Direct yfinance fetch. Stores result into DB for next time."""
    days     = (datetime.date.fromisoformat(end) - datetime.date.fromisoformat(start)).days
    interval = "1wk" if days > 730 else "1d"

    df = yf_fetch(lambda: yf.Ticker(f"{sym}.NS").history(
        start=start, end=end, interval=interval, auto_adjust=True))
    if df is None or df.empty:
        return {"ok": False, "error": f"No data available for {sym}. Please try again in a moment."}

    info = {}
    try:
        fi   = yf.Ticker(f"{sym}.NS").fast_info
        info = {
            "regularMarketPrice":   round(float(fi.last_price            or 0), 2),
            "regularMarketOpen":    round(float(fi.open                  or 0), 2),
            "regularMarketDayHigh": round(float(fi.day_high              or 0), 2),
            "regularMarketDayLow":  round(float(fi.day_low               or 0), 2),
            "regularMarketVolume":  int(fi.three_month_average_volume     or 0),
            "fiftyTwoWeekHigh":     round(float(fi.year_high             or 0), 2),
            "fiftyTwoWeekLow":      round(float(fi.year_low              or 0), 2),
            "chartPreviousClose":   round(float(fi.previous_close        or 0), 2),
        }
    except Exception:
        pass

    rows = []
    for idx, r in df.iterrows():
        c = float(r.get("Close", 0) or 0)
        if c <= 0: continue
        rows.append({"date": str(idx.date()),
                     "open":   round(float(r.get("Open",   0) or 0), 2),
                     "high":   round(float(r.get("High",   0) or 0), 2),
                     "low":    round(float(r.get("Low",    0) or 0), 2),
                     "close":  round(c, 2),
                     "volume": int(r.get("Volume", 0) or 0)})

    # Persist to DB immediately so next request is from cache
    if rows:
        conn  = get_db()
        today = str(datetime.date.today())
        with DB_LOCK:
            conn.executemany(
                "INSERT OR REPLACE INTO prices(symbol,date,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?)",
                [(sym,r["date"],r["open"],r["high"],r["low"],r["close"],r["volume"]) for r in rows]
            )
            conn.execute("""
                INSERT INTO history_meta(symbol,oldest_date,newest_date,last_price_sync)
                VALUES(?,?,?,?)
                ON CONFLICT(symbol) DO UPDATE SET newest_date=excluded.newest_date,
                                                   last_price_sync=excluded.last_price_sync
            """, (sym, rows[0]["date"], rows[-1]["date"], today))
            conn.commit()
        conn.close()

    return {"ok": True, "rows": rows, "info": info}


def fetch_training_data(sym):
    register_symbol(sym)
    end   = datetime.date.today() + datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=730)
    conn  = get_db()
    rows_db = conn.execute(
        "SELECT date,open,high,low,close,volume FROM prices "
        "WHERE symbol=? AND date>=? AND date<=? ORDER BY date ASC",
        (sym, str(start), str(end))
    ).fetchall()
    conn.close()

    if len(rows_db) >= 100:
        df = pd.DataFrame(rows_db, columns=["Date","Open","High","Low","Close","Volume"])
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date").dropna(subset=["Close"])

    df = yf_fetch(lambda: yf.Ticker(f"{sym}.NS").history(
        start=str(start), end=str(end), interval="1d", auto_adjust=True))
    return df.dropna(subset=["Close"]) if df is not None and not df.empty else pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ════════════════════════════════════════════════════════════════════════════
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))

def add_features(df):
    df = df.copy()
    df["MA7"]      = df["Close"].rolling(7).mean()
    df["MA21"]     = df["Close"].rolling(21).mean()
    df["RSI"]      = compute_rsi(df["Close"], 14)
    df["Return"]   = df["Close"].pct_change()
    df["HL_ratio"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)
    df["Vol_norm"] = (df["Volume"] - df["Volume"].mean()) / (df["Volume"].std() + 1e-9)
    return df.dropna()

def compute_indicators(sym):
    register_symbol(sym)
    cache_key = f"{sym}:indicators"
    cached = cache_get(cache_key)
    if cached:
        return cached

    today = str(datetime.date.today())

    # Ensure fundamentals (in background if possible)
    conn = get_db()
    fd   = conn.execute("SELECT fetched_date FROM fundamentals WHERE symbol=?", (sym,)).fetchone()
    conn.close()
    if not fd or fd[0] != today:
        sync_fundamentals(sym)   # runs synchronously but with per-attempt 25s cap

    end   = today
    start = str(datetime.date.today() - datetime.timedelta(days=730))
    conn  = get_db()
    rows_db = conn.execute(
        "SELECT date,open,high,low,close,volume FROM prices "
        "WHERE symbol=? AND date>=? AND date<=? ORDER BY date ASC",
        (sym, start, end)
    ).fetchall()
    fund = conn.execute(
        "SELECT pe,forward_pe,revenue_growth,roe,debt_equity,"
        "insider_pct,inst_pct,sector,industry,high52,low52 "
        "FROM fundamentals WHERE symbol=?", (sym,)
    ).fetchone()
    conn.close()

    if not rows_db:
        # No price data — fetch live
        df = yf_fetch(lambda: yf.Ticker(f"{sym}.NS").history(
            start=start, end=str(datetime.date.today()+datetime.timedelta(days=1)),
            interval="1d", auto_adjust=True))
        if df is None or df.empty:
            raise ValueError(f"No price data for {sym}.NS")
        info = yf_fetch(lambda: yf.Ticker(f"{sym}.NS").info or {}, retries=2, base_delay=3) or {}
        fin  = yf_fetch(lambda: yf.Ticker(f"{sym}.NS").financials, retries=2, base_delay=3)
        result = _build_indicators_from_live(sym, df.dropna(subset=["Close"]), info, fin)
        cache_set(cache_key, result)
        return result

    df = pd.DataFrame(rows_db, columns=["date","Open","High","Low","Close","Volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    closes = df["Close"]
    last   = float(closes.iloc[-1])
    result = {}

    ma50  = float(closes.rolling(50).mean().iloc[-1])  if len(closes) >= 50  else None
    ma200 = float(closes.rolling(200).mean().iloc[-1]) if len(closes) >= 200 else None

    for key, val, n in [("ma50",ma50,50), ("ma200",ma200,200)]:
        if val is None:
            result[key] = {"value":"N/A","signal":"N/A","note":f"Need {n}+ days of data"}
        else:
            pct = round((last-val)/val*100, 2)
            sig = "BUY" if last>val and pct>2 else ("AVOID" if last<val else "CAUTION")
            result[key] = {"value":f"₹{val:.2f}","pct_above":pct,"signal":sig,
                           "note":f"Price {'above' if last>val else 'below'} MA{n} by {abs(pct):.1f}%"}

    rsi_val = float(compute_rsi(closes,14).iloc[-1])
    result["rsi"] = {"value":round(rsi_val,1),
        "signal":"BUY" if rsi_val<30 else ("AVOID" if rsi_val>70 else "CAUTION"),
        "note":"Oversold" if rsi_val<30 else ("Overbought" if rsi_val>70 else "Neutral zone (30-70)")}

    if fund:
        pe_r,fpe_r,rev_g,roe_r,de_r,ins,inst,sec,ind,h52,l52 = fund
    else:
        pe_r=fpe_r=rev_g=roe_r=de_r=ins=inst=None; sec=ind="N/A"; h52=l52=None

    pe_use = pe_r or fpe_r
    if pe_use is None:
        result["pe"] = {"value":"N/A","signal":"N/A","note":"P/E not available"}
    else:
        pe = round(float(pe_use),1)
        result["pe"] = {"value":f"{pe}x",
            "signal":"AVOID" if pe<0 else ("BUY" if pe<15 else ("CAUTION" if pe<30 else "AVOID")),
            "note": f"P/E {pe}x — {'undervalued' if pe<15 else 'fairly valued' if pe<30 else 'overvalued'}"}

    if rev_g is None:
        result["revenue_growth"] = {"value":"N/A","signal":"N/A","note":"Revenue data unavailable"}
    else:
        rg = round(float(rev_g),1)
        result["revenue_growth"] = {"value":f"{rg}%",
            "signal":"BUY" if rg>=20 else ("CAUTION" if rg>=10 else "AVOID"),
            "note":f"{'Strong' if rg>=20 else 'Moderate' if rg>=10 else 'Weak'} revenue growth ({rg}%)"}

    if roe_r is None:
        result["roe"] = {"value":"N/A","signal":"N/A","note":"ROE not available"}
    else:
        roe = round(float(roe_r)*100,1)
        result["roe"] = {"value":f"{roe}%",
            "signal":"BUY" if roe>=15 else ("CAUTION" if roe>=8 else "AVOID"),
            "note":f"ROE {roe}% — {'excellent' if roe>=20 else 'healthy' if roe>=15 else 'moderate' if roe>=8 else 'low'}"}

    if de_r is None:
        result["debt_equity"] = {"value":"N/A","signal":"N/A","note":"D/E not available"}
    else:
        de = round(float(de_r)/100,2)
        result["debt_equity"] = {"value":f"{de}x",
            "signal":"BUY" if de<0.5 else ("CAUTION" if de<1.0 else "AVOID"),
            "note":f"Debt/Equity {de}x — {'low' if de<0.5 else 'moderate' if de<1.0 else 'high'}"}

    if ins is None:
        result["promoter"] = {"value":"N/A","signal":"N/A","note":"Holding data not available"}
    else:
        ph = round(float(ins)*100,1); ih = round(float(inst)*100,1) if inst else 0.0
        result["promoter"] = {"value":f"{ph}%","institutional":f"{ih}%",
            "signal":"BUY" if ph>=50 else ("CAUTION" if ph>=30 else "AVOID"),
            "note":f"Promoter holding {ph}%"}

    high52 = float(h52) if h52 else float(closes.tail(252).max())
    low52  = float(l52) if l52 else float(closes.tail(252).min())
    pft = round((last-high52)/high52*100,1); pfl = round((last-low52)/low52*100,1)
    result["sector_trend"] = {"value":sec or "N/A","industry":ind or None,
        "pct_from_high":pft,"pct_from_low":pfl,
        "signal":"BUY" if pft>-10 else ("AVOID" if pfl<20 else "CAUTION"),
        "note":"Near 52W high" if pft>-10 else ("Near 52W low" if pfl<20 else "Mid-range")}

    weights   = {"ma50":1,"ma200":2,"rsi":2,"pe":2,"revenue_growth":2,"roe":2,"debt_equity":2,"promoter":1,"sector_trend":1}
    score_map = {"BUY":1,"CAUTION":0,"AVOID":-1,"N/A":None}
    tw = ts_ = 0
    for k,w in weights.items():
        if k not in result: continue
        s = score_map.get(result[k].get("signal","N/A"))
        if s is None: continue
        ts_ += s*w; tw += w
    sc = round(ts_/tw*100,1) if tw else 0
    result["overall"] = {"score":sc,"last_price":round(last,2),"high52":round(high52,2),"low52":round(low52,2),
        "signal":"BUY" if sc>=15 else ("CAUTION" if sc>=-15 else "AVOID"),
        "label":"Strong Buy" if sc>=50 else ("Moderate Buy" if sc>=15 else
               ("Neutral / Hold" if sc>=-15 else ("Moderate Caution" if sc>=-50 else "Strong Caution")))}

    cache_set(cache_key, result)
    return result


def _build_indicators_from_live(sym, df, info, fin):
    closes = df["Close"]; last = float(closes.iloc[-1]); result = {}
    def safe(k,d=None):
        v=info.get(k); return v if (v is not None and v!="N/A" and v!=0) else d
    ma50  = float(closes.rolling(50).mean().iloc[-1])  if len(closes)>=50  else None
    ma200 = float(closes.rolling(200).mean().iloc[-1]) if len(closes)>=200 else None
    for key,val,n in [("ma50",ma50,50),("ma200",ma200,200)]:
        if val is None: result[key]={"value":"N/A","signal":"N/A","note":f"Need {n}+ days"}
        else:
            pct=round((last-val)/val*100,2); sig="BUY" if last>val and pct>2 else ("AVOID" if last<val else "CAUTION")
            result[key]={"value":f"₹{val:.2f}","pct_above":pct,"signal":sig,
                         "note":f"Price {'above' if last>val else 'below'} MA{n} by {abs(pct):.1f}%"}
    rsi=float(compute_rsi(closes,14).iloc[-1])
    result["rsi"]={"value":round(rsi,1),"signal":"BUY" if rsi<30 else ("AVOID" if rsi>70 else "CAUTION"),
                   "note":"Oversold" if rsi<30 else ("Overbought" if rsi>70 else "Neutral")}
    pe_use=safe("trailingPE") or safe("forwardPE")
    if pe_use is None: result["pe"]={"value":"N/A","signal":"N/A","note":"P/E not available"}
    else:
        pe=round(float(pe_use),1)
        result["pe"]={"value":f"{pe}x","signal":"AVOID" if pe<0 else ("BUY" if pe<15 else ("CAUTION" if pe<30 else "AVOID")),"note":f"P/E {pe}x"}
    rev_g=None
    try:
        if fin is not None and not fin.empty:
            for k in ["Total Revenue","Revenue","TotalRevenue"]:
                if k in fin.index:
                    rv=fin.loc[k].dropna().sort_index()
                    if len(rv)>=2: rev_g=round(float((rv.iloc[-1]-rv.iloc[0])/abs(rv.iloc[0])*100),1)
                    break
    except: pass
    if rev_g is None: result["revenue_growth"]={"value":"N/A","signal":"N/A","note":"Revenue unavailable"}
    else: result["revenue_growth"]={"value":f"{rev_g}%","signal":"BUY" if rev_g>=20 else ("CAUTION" if rev_g>=10 else "AVOID"),"note":f"{rev_g}% growth"}
    for key,sk,mult,t0,t1 in [("roe","returnOnEquity",100,15,8),("debt_equity","debtToEquity",0.01,0.5,1.0)]:
        raw=safe(sk)
        if raw is None: result[key]={"value":"N/A","signal":"N/A","note":f"{key} not available"}
        else:
            v=round(float(raw)*mult,1 if mult==100 else 2)
            sg="BUY" if (v>=t0 if mult==100 else v<t0) else ("CAUTION" if (v>=t1 if mult==100 else v<t1) else "AVOID")
            result[key]={"value":f"{v}%" if mult==100 else f"{v}x","signal":sg,"note":f"{key} = {v}"}
    ins=safe("heldPercentInsiders"); inst=safe("heldPercentInstitutions")
    if ins is None: result["promoter"]={"value":"N/A","signal":"N/A","note":"Holding data N/A"}
    else:
        ph=round(float(ins)*100,1); ih=round(float(inst)*100,1) if inst else 0.0
        result["promoter"]={"value":f"{ph}%","institutional":f"{ih}%","signal":"BUY" if ph>=50 else ("CAUTION" if ph>=30 else "AVOID"),"note":f"Promoter {ph}%"}
    h52=safe("fiftyTwoWeekHigh") or float(closes.tail(252).max())
    l52=safe("fiftyTwoWeekLow")  or float(closes.tail(252).min())
    pft=round((last-h52)/h52*100,1); pfl=round((last-l52)/l52*100,1)
    result["sector_trend"]={"value":safe("sector","N/A"),"industry":safe("industry",None),
        "pct_from_high":pft,"pct_from_low":pfl,
        "signal":"BUY" if pft>-10 else ("AVOID" if pfl<20 else "CAUTION"),
        "note":"Near 52W high" if pft>-10 else ("Near 52W low" if pfl<20 else "Mid-range")}
    weights={"ma50":1,"ma200":2,"rsi":2,"pe":2,"revenue_growth":2,"roe":2,"debt_equity":2,"promoter":1,"sector_trend":1}
    sm={"BUY":1,"CAUTION":0,"AVOID":-1,"N/A":None}; tw=ts_=0
    for k,w in weights.items():
        if k not in result: continue
        s=sm.get(result[k].get("signal","N/A"))
        if s is None: continue
        ts_+=s*w; tw+=w
    sc=round(ts_/tw*100,1) if tw else 0
    result["overall"]={"score":sc,"last_price":round(last,2),"high52":round(h52,2),"low52":round(l52,2),
        "signal":"BUY" if sc>=15 else ("CAUTION" if sc>=-15 else "AVOID"),
        "label":"Strong Buy" if sc>=50 else ("Moderate Buy" if sc>=15 else ("Neutral / Hold" if sc>=-15 else ("Moderate Caution" if sc>=-50 else "Strong Caution")))}
    return result


# ════════════════════════════════════════════════════════════════════════════
# LSTM + PROPHET  (unchanged logic)
# ════════════════════════════════════════════════════════════════════════════
SEQ_LEN   = 30
FEAT_COLS = ["Close","Open","High","Low","MA7","MA21","RSI","Return","HL_ratio","Vol_norm"]

def next_trading_days(last_date, n):
    dates, d = [], last_date
    while len(dates) < n:
        d += datetime.timedelta(days=1)
        if d.weekday() < 5: dates.append(d)
    return dates

if TORCH_OK:
    class LSTMNet(nn.Module):
        def __init__(self,n_features,hidden=64,layers=2,dropout=0.2):
            super().__init__()
            self.lstm=nn.LSTM(n_features,hidden,num_layers=layers,batch_first=True,dropout=dropout)
            self.head=nn.Sequential(nn.Linear(hidden,32),nn.ReLU(),nn.Dropout(dropout),nn.Linear(32,1))
        def forward(self,x):
            out,_=self.lstm(x); return self.head(out[:,-1,:]).squeeze(-1)

def forecast_lstm(sym,days):
    if not TORCH_OK: raise RuntimeError("PyTorch not installed.")
    df=fetch_training_data(sym); df=add_features(df)
    feats=[c for c in FEAT_COLS if c in df.columns]; n_f=len(feats)
    if len(df)<SEQ_LEN+20: raise ValueError(f"Not enough history ({len(df)} rows).")
    data=df[feats].values.astype(np.float32); scaler=MinMaxScaler(); scaled=scaler.fit_transform(data).astype(np.float32)
    Xs,ys=[],[]
    for i in range(SEQ_LEN,len(scaled)): Xs.append(scaled[i-SEQ_LEN:i]); ys.append(scaled[i,0])
    X=torch.tensor(np.array(Xs),dtype=torch.float32); y=torch.tensor(np.array(ys),dtype=torch.float32)
    split=int(len(X)*0.85); X_tr,X_val=X[:split],X[split:]; y_tr,y_val=y[:split],y[split:]
    loader=DataLoader(TensorDataset(X_tr,y_tr),batch_size=16,shuffle=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=LSTMNet(n_f).to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
    loss_fn=nn.HuberLoss(); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5,factor=0.5)
    best_val,best_state,pat=float("inf"),None,0
    for ep in range(80):
        model.train()
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(); loss_fn(model(xb),yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval()
        with torch.no_grad(): vl=loss_fn(model(X_val.to(device)),y_val.to(device)).item()
        sched.step(vl)
        if vl<best_val-1e-5: best_val,best_state,pat=vl,{k:v.clone() for k,v in model.state_dict().items()},0
        else:
            pat+=1
            if pat>=12: break
    model.load_state_dict(best_state); model.eval()
    def inv(arr):
        d=np.zeros((len(arr),n_f),dtype=np.float32); d[:,0]=arr; return scaler.inverse_transform(d)[:,0]
    with torch.no_grad(): vp=model(X_val.to(device)).cpu().numpy()
    va=inv(y_val.numpy()); vf=inv(vp)
    mae=float(np.mean(np.abs(va-vf))); rmse=float(np.sqrt(np.mean((va-vf)**2))); rs=float(np.std(va-vf))
    lp=float(df["Close"].iloc[-1]); conf=max(0,min(100,round(100-(rmse/lp*100),1)))
    window=scaled[-SEQ_LEN:].copy(); preds=[]
    with torch.no_grad():
        for _ in range(days):
            xi=torch.tensor(window[np.newaxis],dtype=torch.float32).to(device); ps=model(xi).item(); preds.append(ps)
            nr=window[-1].copy(); nr[0]=ps; window=np.vstack([window[1:],nr])
    prices=inv(np.array(preds,dtype=np.float32))
    ld=df.index[-1].date() if hasattr(df.index[-1],"date") else datetime.date.fromisoformat(str(df.index[-1])[:10])
    fds=next_trading_days(ld,days)
    fc=[{"date":str(d),"price":round(float(p),2),"lower":round(float(p)-rs*(1+i*0.06),2),"upper":round(float(p)+rs*(1+i*0.06),2)} for i,(d,p) in enumerate(zip(fds,prices))]
    return {"model":"LSTM (PyTorch)","confidence":conf,"mae":round(mae,2),"rmse":round(rmse,2),
            "last_actual_date":str(ld),"last_actual_price":round(lp,2),"forecast":fc}

def forecast_prophet(sym,days):
    if not PROPHET_OK: raise RuntimeError("Prophet not installed.")
    df=fetch_training_data(sym); df=add_features(df)
    ts=df.index.tz_localize(None) if hasattr(df.index,"tzinfo") and df.index.tzinfo else df.index
    pdf=pd.DataFrame({"ds":ts,"y":df["Close"].values}); pdf["vol_norm"]=df["Vol_norm"].values
    m=Prophet(daily_seasonality=False,weekly_seasonality=True,yearly_seasonality=True,
              changepoint_prior_scale=0.05,seasonality_prior_scale=10.0,interval_width=0.80)
    m.add_regressor("vol_norm")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); m.fit(pdf)
    ld=df.index[-1].date() if hasattr(df.index[-1],"date") else datetime.date.fromisoformat(str(df.index[-1])[:10])
    fds=next_trading_days(ld,days)
    fdf=pd.DataFrame({"ds":fds,"vol_norm":0.0}); fc=m.predict(fdf)
    p60=pdf.tail(60).copy(); ifc=m.predict(p60[["ds","vol_norm"]])
    mae=float(np.mean(np.abs(p60["y"].values-ifc["yhat"].values))); rmse=float(np.sqrt(np.mean((p60["y"].values-ifc["yhat"].values)**2)))
    lp=float(df["Close"].iloc[-1]); conf=max(0,min(100,round(100-(rmse/lp*100),1)))
    forecast=[{"date":str(r["ds"].date()),"price":round(float(r["yhat"]),2),"lower":round(float(r["yhat_lower"]),2),"upper":round(float(r["yhat_upper"]),2)} for _,r in fc.iterrows()]
    return {"model":"Prophet","confidence":conf,"mae":round(mae,2),"rmse":round(rmse,2),
            "last_actual_date":str(ld),"last_actual_price":round(lp,2),"forecast":forecast}

def forecast_ensemble(sym,days):
    results,errors=[],[]
    if TORCH_OK:
        try: results.append(forecast_lstm(sym,days))
        except Exception as e: errors.append(f"LSTM: {e}")
    if PROPHET_OK:
        try: results.append(forecast_prophet(sym,days))
        except Exception as e: errors.append(f"Prophet: {e}")
    if not results: raise RuntimeError("No models available.\n"+"\n".join(errors))
    if len(results)==1: r=results[0]; r["model"]+=" (solo)"; return r
    n=min(len(r["forecast"]) for r in results)
    bl=[{"date":results[0]["forecast"][i]["date"],
         "price":round(sum(r["forecast"][i]["price"] for r in results)/len(results),2),
         "lower":round(sum(r["forecast"][i].get("lower",r["forecast"][i]["price"]) for r in results)/len(results),2),
         "upper":round(sum(r["forecast"][i].get("upper",r["forecast"][i]["price"]) for r in results)/len(results),2)} for i in range(n)]
    return {"model":"Ensemble (LSTM + Prophet)",
            "confidence":round(sum(r["confidence"] for r in results)/len(results),1),
            "mae":round(sum(r["mae"] for r in results)/len(results),2),
            "rmse":round(sum(r["rmse"] for r in results)/len(results),2),
            "last_actual_date":results[0]["last_actual_date"],
            "last_actual_price":results[0]["last_actual_price"],"forecast":bl}


# ════════════════════════════════════════════════════════════════════════════
# THREADING HTTP SERVER  — handles concurrent requests
# ════════════════════════════════════════════════════════════════════════════
class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Each request runs in its own thread. BrokenPipe never crashes the server."""
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        """Override default handler — silently ignore BrokenPipe and ConnectionReset."""
        exc_type = sys.exc_info()[0] if __import__('sys').exc_info()[0] else None
        import sys as _sys
        etype = _sys.exc_info()[0]
        if etype in (BrokenPipeError, ConnectionResetError):
            return   # client closed connection — totally normal, ignore
        # For anything else, print a short summary (not the full traceback)
        print(f"  [HTTP ERROR] {client_address}: {_sys.exc_info()[1]}")


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # Only log non-trivial requests
        if args and "200" in str(args[0]):
            print(f"  -> {self.path.split('?')[0]}  {args[0]}")

    def send_json(self, data, status=200):
        try:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type",                 "application/json")
            self.send_header("Access-Control-Allow-Origin",  "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass   # client disconnected — not an error

    def do_OPTIONS(self):
        try:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin",  "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        try:
            self._handle_get()
        except (BrokenPipeError, ConnectionResetError):
            pass   # client closed tab/connection — ignore
        except Exception:
            try: self.send_json({"ok":False,"error":traceback.format_exc()}, 500)
            except: pass

    def _handle_get(self):
        parsed = urlparse(self.path)
        qs     = parse_qs(parsed.query)

        if parsed.path == "/ping":
            db_mb = round(DB_PATH.stat().st_size/1024/1024, 2) if DB_PATH.exists() else 0
            self.send_json({"ok":True,"lstm":TORCH_OK,"prophet":PROPHET_OK,"db_mb":db_mb})
            return

        if parsed.path == "/db_status":
            conn = get_db()
            rows = conn.execute(
                "SELECT symbol,oldest_date,newest_date,last_price_sync,last_fund_sync "
                "FROM history_meta ORDER BY symbol"
            ).fetchall()
            cnt  = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
            conn.close()
            self.send_json({"ok":True,"total_rows":cnt,
                            "symbols":[{"sym":r[0],"oldest":r[1],"newest":r[2],
                                        "last_price":r[3],"last_fund":r[4]} for r in rows]})
            return

        if parsed.path == "/history":
            sym   = qs.get("sym",   [""])[0].upper().strip()
            start = qs.get("start", [""])[0].strip()
            end   = qs.get("end",   [""])[0].strip()
            if not sym or not start or not end:
                self.send_json({"ok":False,"error":"Missing sym/start/end"}, 400); return
            self.send_json(get_stock_data(sym, start, end))
            return

        if parsed.path == "/indicators":
            sym = qs.get("sym", [""])[0].upper().strip()
            if not sym:
                self.send_json({"ok":False,"error":"Missing sym"}, 400); return
            data = compute_indicators(sym)
            data["ok"] = True
            self.send_json(data)
            return

        if parsed.path == "/forecast":
            sym   = qs.get("sym",   [""])[0].upper().strip()
            days  = max(7, min(30, int(qs.get("days",  ["14"])[0])))
            model = qs.get("model", ["ensemble"])[0].lower()
            if not sym:
                self.send_json({"ok":False,"error":"Missing sym"}, 400); return
            r = (forecast_lstm(sym,days)     if model == "lstm"    else
                 forecast_prophet(sym,days)   if model == "prophet" else
                 forecast_ensemble(sym,days))
            r["ok"] = True
            self.send_json(r)
            return

        # ── /symbols — dynamic Nifty list from DB ──────────────────────────
        if parsed.path == "/symbols":
            conn = get_db()
            syms = conn.execute(
                "SELECT DISTINCT symbol FROM history_meta ORDER BY symbol"
            ).fetchall()
            conn.close()
            result = [{"s": r[0], "n": r[0]} for r in syms]
            self.send_json({"ok": True, "symbols": result})
            return

        # ── /peer?sym=TCS — sector peers ─────────────────────────────────
        if parsed.path == "/peer":
            sym = qs.get("sym", [""])[0].upper().strip()
            if not sym:
                self.send_json({"ok": False, "error": "Missing sym"}, 400); return
            # Get sector from fundamentals, then fetch all peers in same sector
            conn = get_db()
            row  = conn.execute("SELECT sector FROM fundamentals WHERE symbol=?", (sym,)).fetchone()
            conn.close()
            if not row or not row[0] or row[0] == "N/A":
                self.send_json({"ok": False, "error": "Sector not available for this symbol"}, 400); return
            sector = row[0]
            # Gather peers — symbols in same sector with fundamentals
            conn   = get_db()
            peers  = conn.execute(
                "SELECT symbol,pe,forward_pe,roe,debt_equity,revenue_growth,high52,low52,mkt_price "
                "FROM fundamentals WHERE sector=? AND symbol!=? AND mkt_price IS NOT NULL ORDER BY symbol LIMIT 10",
                (sector, sym)
            ).fetchall()
            conn.close()
            peer_list = []
            for p in peers:
                s,pe,fpe,roe,de,rg,h52,l52,mp = p
                peer_list.append({
                    "sym": s, "pe": round(float(pe),1) if pe else None,
                    "roe": round(float(roe)*100,1) if roe else None,
                    "de": round(float(de)/100,2) if de else None,
                    "rev_growth": round(float(rg),1) if rg else None,
                    "price": round(float(mp),2) if mp else None,
                })
            self.send_json({"ok": True, "sector": sector, "peers": peer_list})
            return

        # ════════════════════════════════════════════════════════════════
        # AUTH ENDPOINTS  (POST only)
        # ════════════════════════════════════════════════════════════════
        if self.command == "POST":
            self._handle_post(parsed.path)
            return

        self.send_json({"ok":False,"error":"Not found"}, 404)

    def do_POST(self):
        try:
            self._handle_post(urlparse(self.path).path)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            try: self.send_json({"ok":False,"error":traceback.format_exc()}, 500)
            except: pass

    def _handle_post(self, path):
        # ── /auth/register ───────────────────────────────────────────────
        if path == "/auth/register":
            body = _read_body(self)
            username = (body.get("username","")).strip()
            email    = (body.get("email","")).strip().lower()
            password = (body.get("password","")).strip()
            if not username or not email or not password:
                self.send_json({"ok":False,"error":"All fields required"}, 400); return
            if len(password) < 6:
                self.send_json({"ok":False,"error":"Password must be at least 6 characters"}, 400); return
            conn = get_db()
            now  = str(datetime.datetime.utcnow())
            try:
                conn.execute(
                    "INSERT INTO users(username,email,password,role,status,created_at) VALUES(?,?,?,?,?,?)",
                    (username, email, _hash_pw(password), "user", "pending", now)
                )
                conn.commit()
                conn.close()
                print(f"  [AUTH] New registration: {username} <{email}>  — awaiting admin approval")
                self.send_json({"ok":True,"msg":"Registration successful. Awaiting admin approval."})
            except Exception as e:
                conn.close()
                if "UNIQUE" in str(e):
                    self.send_json({"ok":False,"error":"Username or email already registered"}, 409)
                else:
                    self.send_json({"ok":False,"error":str(e)}, 500)
            return

        # ── /auth/login ──────────────────────────────────────────────────
        if path == "/auth/login":
            body     = _read_body(self)
            username = (body.get("username","")).strip()
            password = (body.get("password","")).strip()
            conn     = get_db()
            user     = conn.execute(
                "SELECT id,username,email,role,status FROM users WHERE username=? AND password=?",
                (username, _hash_pw(password))
            ).fetchone()
            if not user:
                conn.close()
                self.send_json({"ok":False,"error":"Invalid username or password"}, 401); return
            uid,uname,uemail,urole,ustatus = user
            if ustatus == "pending":
                conn.close()
                self.send_json({"ok":False,"error":"Your account is pending admin approval. Please wait."}); return
            if ustatus == "rejected":
                conn.close()
                self.send_json({"ok":False,"error":"Your account has been rejected. Contact admin."}); return
            token    = _make_token()
            now      = datetime.datetime.utcnow()
            expires  = str(now + datetime.timedelta(days=7))
            now_str  = str(now)
            conn.execute("INSERT INTO sessions(token,user_id,created_at,expires_at) VALUES(?,?,?,?)",
                         (token, uid, now_str, expires))
            conn.commit()
            conn.close()
            self.send_json({"ok":True,"token":token,"username":uname,"email":uemail,"role":urole})
            return

        # ── /auth/logout ─────────────────────────────────────────────────
        if path == "/auth/logout":
            auth = self.headers.get("Authorization","")
            if auth.startswith("Bearer "):
                token = auth[7:]
                conn  = get_db()
                with DB_LOCK:
                    conn.execute("DELETE FROM sessions WHERE token=?", (token,))
                    conn.commit()
                conn.close()
            self.send_json({"ok":True})
            return

        # ── /auth/me ─────────────────────────────────────────────────────
        if path == "/auth/me":
            user = _get_session(self)
            if not user:
                self.send_json({"ok":False,"error":"Not authenticated"}, 401); return
            self.send_json({"ok":True,"user":user})
            return

        # ── /auth/forgot ─────────────────────────────────────────────────
        if path == "/auth/forgot":
            body  = _read_body(self)
            email = (body.get("email","")).strip().lower()
            conn  = get_db()
            user  = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
            if user:
                token   = _make_token()[:8].upper()   # short code for manual entry
                expiry  = str(datetime.datetime.utcnow() + datetime.timedelta(hours=1))
                conn.execute("UPDATE users SET reset_token=?,reset_expiry=? WHERE email=?",
                             (token, expiry, email))
                conn.commit()
                # In production: send email. For now, log it.
                print(f"  [AUTH] Password reset token for {email}: {token}  (expires {expiry})")
            conn.close()
            # Always respond OK (don't reveal if email exists)
            self.send_json({"ok":True,"msg":"If that email is registered, a reset code has been sent. Check server logs for now."})
            return

        # ── /auth/reset ──────────────────────────────────────────────────
        if path == "/auth/reset":
            body     = _read_body(self)
            email    = (body.get("email","")).strip().lower()
            token    = (body.get("token","")).strip().upper()
            password = (body.get("password","")).strip()
            if len(password) < 6:
                self.send_json({"ok":False,"error":"Password must be at least 6 characters"}, 400); return
            conn = get_db()
            now  = str(datetime.datetime.utcnow())
            user = conn.execute(
                "SELECT id FROM users WHERE email=? AND reset_token=? AND reset_expiry>?",
                (email, token, now)
            ).fetchone()
            if not user:
                conn.close()
                self.send_json({"ok":False,"error":"Invalid or expired reset code"}, 400); return
            conn.execute("UPDATE users SET password=?,reset_token=NULL,reset_expiry=NULL WHERE id=?",
                         (_hash_pw(password), user[0]))
            conn.commit()
            conn.close()
            self.send_json({"ok":True,"msg":"Password reset successful. You can now log in."})
            return

        # ── /admin/users ─────────────────────────────────────────────────
        if path == "/admin/users":
            user = _require_auth(self, admin=True)
            if not user: return
            conn  = get_db()
            users = conn.execute(
                "SELECT id,username,email,role,status,created_at,approved_at FROM users ORDER BY created_at DESC"
            ).fetchall()
            conn.close()
            self.send_json({"ok":True,"users":[
                {"id":r[0],"username":r[1],"email":r[2],"role":r[3],
                 "status":r[4],"created_at":r[5],"approved_at":r[6]} for r in users]})
            return

        # ── /admin/approve ────────────────────────────────────────────────
        if path == "/admin/approve":
            user = _require_auth(self, admin=True)
            if not user: return
            body   = _read_body(self)
            uid    = body.get("id")
            action = body.get("action","approve")   # approve | reject
            status = "approved" if action == "approve" else "rejected"
            now    = str(datetime.datetime.utcnow())
            conn   = get_db()
            conn.execute("UPDATE users SET status=?,approved_at=? WHERE id=?", (status, now, uid))
            conn.commit()
            uname = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone()
            conn.close()
            print(f"  [ADMIN] User {uname[0] if uname else uid} {status}")
            self.send_json({"ok":True,"status":status})
            return

        # ── /admin/pending_count ──────────────────────────────────────────
        if path == "/admin/pending_count":
            user = _require_auth(self, admin=True)
            if not user: return
            conn = get_db()
            cnt  = conn.execute("SELECT COUNT(*) FROM users WHERE status='pending'").fetchone()[0]
            conn.close()
            self.send_json({"ok":True,"count":cnt})
            return

        self.send_json({"ok":False,"error":"Not found"}, 404)


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    db_mb = round(DB_PATH.stat().st_size/1024/1024, 1) if DB_PATH.exists() else 0
    print("=" * 62)
    print("  NSE Stock Analyzer — Server v7")
    print(f"  Listening on 0.0.0.0:{PORT}")
    print(f"  DB: {DB_PATH}  ({db_mb} MB)")
    print(f"  LSTM    : {'OK' if TORCH_OK    else 'not installed'}")
    print(f"  Prophet : {'OK' if PROPHET_OK  else 'not installed'}")
    print(f"  Threads : multi (BrokenPipe safe)")
    print("=" * 62)
    try:
        server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

import os, sys, json, uuid, math, traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from google.cloud import bigquery

# ---------- utils ----------
def log(x: str): 
    print(x, flush=True)

def z(dt: datetime) -> str:
    if dt.tzinfo is None: 
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00","Z")

def normalize(xs: List[float]) -> List[float]:
    if not xs: return xs
    a = np.array(xs, dtype=float)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if math.isclose(lo, hi): return [0.0 for _ in a]
    return list((a - lo) / (hi - lo))

# ---------- UW flow ----------
def fetch_uw_flow(api_key: str, base: str, path: str, start: datetime, end: datetime, min_prem: int) -> List[Dict[str, Any]]:
    url = base.rstrip("/") + "/" + path.lstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    params = {
        "start": start.date().isoformat(), 
        "end": end.date().isoformat(),
        "min_premium": int(min_prem), 
        "sort": "premium_desc", 
        "limit": 5000
    }
    log(f"uw GET {url} {params}")
    try:
        r = requests.get(url, headers=headers, params=params, timeout=25)
        if r.status_code == 401:
            raise RuntimeError("UW 401 Unauthorized — check UW_API_KEY/plan.")
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            for k in ("results","data","items","flow"):
                if k in data and isinstance(data[k], list): 
                    return data[k]
        if isinstance(data, list): return data
        return []
    except Exception as e:
        log(f"uw error: {e}")
        return []

def aggregate_flow(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[
            "ticker","flow_total_premium","flow_call_prem","flow_put_prem",
            "flow_call_put_ratio","flow_sweeps","flow_aggr_buys","flow_score"
        ])
    recs = []
    for r in rows:
        t = (r.get("ticker") or r.get("symbol") or r.get("underlying") or "").upper()
        if not t: continue
        prem = float(r.get("premium") or r.get("usd_premium") or 0)
        side = str(r.get("side") or r.get("option_type") or "").lower()
        is_call = 1 if "call" in side else 0
        is_put  = 1 if "put"  in side else 0
        sweep   = 1 if (r.get("is_sweep") or r.get("sweep")) else 0
        aggr    = str(r.get("position") or r.get("at") or "").lower()
        is_aggr_buy = 1 if ("ask" in aggr or "above_ask" in aggr) else 0
        recs.append({
            "ticker": t,
            "prem": prem,
            "is_call": is_call,
            "is_put": is_put,
            "is_sweep": sweep,
            "is_aggr_buy": is_aggr_buy
        })
    df = pd.DataFrame(recs)
    g = df.groupby("ticker")
    out = pd.DataFrame({
        "flow_total_premium": g["prem"].sum(),
        "flow_call_prem": (df["prem"]*df["is_call"]).groupby(df["ticker"]).sum(),
        "flow_put_prem":  (df["prem"]*df["is_put"]).groupby(df["ticker"]).sum(),
        "flow_sweeps": g["is_sweep"].sum(),
        "flow_aggr_buys": g["is_aggr_buy"].sum()
    })
    out["flow_call_put_ratio"] = (out["flow_call_prem"]+1)/(out["flow_put_prem"]+1)
    raw = (
        0.5*(out["flow_total_premium"]/max(out["flow_total_premium"].max(),1)) +
        0.2*(np.log1p(out["flow_sweeps"])/max(np.log1p(out["flow_sweeps"].max()),1)) +
        0.3*(np.tanh(out["flow_call_put_ratio"]-1.0)+1)/2.0
    ).fillna(0.0)
    out["flow_score"] = (raw-raw.min())/(raw.max()-raw.min()+1e-9)
    out = out.reset_index(names="ticker")
    return out

# ---------- prices/momentum/levels ----------
def fetch_price_levels(tickers: List[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            df = yf.download(t, period="2mo", interval="1d", progress=False, auto_adjust=False)
            if df.empty: continue
            df = df.rename(columns=str.lower)
            df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
            df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
            tr = np.maximum(df["high"]-df["low"],
                            np.maximum(abs(df["high"]-df["close"].shift(1)),
                                       abs(df["low"]-df["close"].shift(1))))
            df["atr20"] = tr.rolling(20).mean()
            # last week high/low
            df["wk"] = df.index.to_period("W").astype(str)
            last_week = sorted(df["wk"].unique())[-2] if len(df["wk"].unique())>=2 else df["wk"].iloc[-1]
            wdf = df[df["wk"]==last_week]
            prev_hi, prev_lo = float(wdf["high"].max()), float(wdf["low"].min())
            close = float(df["close"].iloc[-1])
            ema20=float(df["ema20"].iloc[-1]); ema50=float(df["ema50"].iloc[-1]); atr=float(df["atr20"].iloc[-1])
            week_ret = (df["close"].iloc[-1]/df["close"].iloc[-5]-1.0) if len(df)>5 else 0.0
            momentum = 0.6*max(0.0,(close/ema20)-1.0) + 0.4*max(-0.2,min(0.2,week_ret))
            rows.append({
                "ticker": t,"close": close,"ema20": ema20,"ema50": ema50,
                "atr20": atr,"prev_week_high": prev_hi,"prev_week_low": prev_lo,"momentum": momentum
            })
        except Exception as e:
            log(f"yf {t}: {e}")
    return pd.DataFrame(rows)

def build_levels(r: pd.Series) -> Dict[str, Any]:
    atr=float(r.get("atr20",0) or 0)
    hi=float(r.get("prev_week_high",0) or 0)
    lo=float(r.get("prev_week_low",0) or 0)
    return {
        "breakout": round(hi,2),"support": round(lo,2),
        "R1": round(hi+0.5*atr,2),"R2": round(hi+1.0*atr,2),
        "S1": round(lo-0.5*atr,2),"S2": round(lo-1.0*atr,2),
        "ema20": round(float(r.get("ema20",0)),2),"ema50": round(float(r.get("ema50",0)),2),"atr20": round(atr,2)
    }

# ---------- agent ----------
class WeekendAgent:
    def __init__(self):
        self.project_id = os.environ["PROJECT_ID"]
        self.dataset    = os.environ.get("BQ_DATASET","trading")
        self.discord    = os.environ.get("DISCORD_WEBHOOK")
        self.uw_key     = os.environ["UW_API_KEY"]
        self.base_url   = os.environ.get("UW_BASE_URL","https://api.unusualwhales.com")
        self.flow_path  = os.environ.get("UW_FLOW_PATH","/v2/flow/search")
        self.lookback   = int(os.environ.get("UW_LOOKBACK_DAYS","5"))
        self.min_prem   = int(os.environ.get("UW_MIN_PREMIUM","100000"))
        self.stock_cap  = float(os.environ.get("STOCK_MAX_PRICE","200"))
        self.run_id     = str(uuid.uuid4())
        self.started_at = datetime.now(timezone.utc)
        self.bq         = bigquery.Client(project=self.project_id)

    def observe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        start = self.started_at - timedelta(days=self.lookback)
        rows  = fetch_uw_flow(self.uw_key, self.base_url, self.flow_path, start, self.started_at, self.min_prem)
        flow  = aggregate_flow(rows)
        tickers = sorted(set(flow["ticker"].tolist()))  # <-- no manual watchlist
        px = fetch_price_levels(tickers)
        return flow, px

    def decide(self, flow: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
        if flow.empty and px.empty: return pd.DataFrame()
        df = pd.merge(px, flow, on="ticker", how="left").fillna(0.0)
        df = df[(df["close"] <= self.stock_cap) &
                (df["momentum"] > 0) &
                (df["flow_total_premium"] >= self.min_prem) &
                (df["flow_call_put_ratio"] >= 2.0)]
        if df.empty: return df
        df["momentum_n"] = normalize(df["momentum"].tolist())
        df["flow_n"]     = normalize(df["flow_score"].tolist())
        df["score"]      = 0.55*df["momentum_n"] + 0.45*df["flow_n"]
        df["levels"]     = df.apply(build_levels, axis=1)
        df = df.sort_values("score", ascending=False).head(15).reset_index(drop=True)
        return df

    def act(self, picks: pd.DataFrame):
        if picks.empty:
            log("no picks")
            return
        table = f"{self.project_id}.{self.dataset}.weekend_decisions"
        rows = []
        for _, r in picks.iterrows():
            rows.append({
                "run_id": self.run_id,
                "decided_at": z(self.started_at),
                "ticker": r["ticker"],
                "rationale": f"flow+momentum score={round(float(r['score']),3)}",
                "score": float(r["score"]),
                "action": "watchlist",
                "inputs": json.dumps({
                    "levels": r["levels"],
                    "features": {"close": r["close"], "momentum": r["momentum"],
                                 "flow_score": r.get("flow_score",0.0),
                                 "flow_total_premium": r.get("flow_total_premium",0.0),
                                 "call_put_ratio": r.get("flow_call_put_ratio",0.0)}}
                )
            })
        errs = self.bq.insert_rows_json(table, rows)
        if errs: log(f"bq errors: {errs}")

        if self.discord:
            lines = []
            for _, r in picks.iterrows():
                lv = r["levels"]; t = r["ticker"]
                lines.append(
                    f"**{t}**  s={round(float(r['score']),2)}  close {round(float(r['close']),2)} | "
                    f"BO {lv['breakout']}  R1 {lv['R1']}  R2 {lv['R2']} | "
                    f"S1 {lv['S1']}  S2 {lv['S2']} | EMA20 {lv['ema20']} ATR {lv['atr20']}\n"
                    f"```tr-bigflow {t} | tr-topflow {t} | tr-dplevels {t}```"
                )
            msg = "📓 **Weekend Watchlist (UW flow + momentum, price<200)**\n" + "\n".join(lines)
            try:
                requests.post(self.discord, json={"content": msg}, timeout=10)
            except Exception as e:
                log(f"discord error: {e}")

    def run(self):
        log("weekend: start")
        try:
            flow, px = self.observe()
            picks = self.decide(flow, px)
            self.act(picks)
            log(f"weekend: done. picks={len(picks)}")
        except Exception as e:
            log(f"weekend failed: {e}")
            traceback.print_exc()
            raise

if __name__ == "__main__":
    WeekendAgent().run()

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from datetime import datetime
from bson.objectid import ObjectId

# --------------------------
# 1) Page setup
# --------------------------
st.set_page_config(page_title="Fraud Streaming Dashboard", layout="wide")

# --------------------------
# 2) Sidebar controls
# --------------------------
st.sidebar.header("Controls")

auto_refresh = st.sidebar.checkbox("Auto refresh", value=True)
refresh_seconds = 120  # 2 minutes

if auto_refresh:
    st_autorefresh(interval=refresh_seconds * 1000, limit=None, key="fraud_dash")

show_not_fraud = st.sidebar.checkbox("Show ML Not-Fraud table (ml_not_fraud)", value=False)

st.sidebar.subheader("Alerts filters (Q5)")
high_amount_threshold = st.sidebar.number_input("High amount threshold", value=1000.0, step=100.0)
alerts_limit = st.sidebar.slider("Alerts rows (display)", min_value=50, max_value=2000, value=500, step=50)

st.sidebar.caption(f"Auto refresh is set to {refresh_seconds} seconds (2 minutes).")

if st.sidebar.button("🔄 Refresh now (hard)"):
    for k in list(st.session_state.keys()):
        if k.startswith("cursor_") or k.startswith("store_"):
            del st.session_state[k]
    st.rerun()

# --------------------------
# 3) MongoDB connection
# --------------------------
client = MongoClient("mongodb://127.0.0.1:27017")
db = client["frauddb"]

loss_collection = db["fraud_loss_2m"]                 # Q2
minmax_collection = db["fraud_minmax_2m"]             # Q3
alerts_collection = db["fraud_alerts_realtime"]       # Q5

# Q6 (Option 2): split into totals + fraud counts
summary_total_collection = db["summary_total_by_type_2m"]   # from raw transactions
summary_fraud_collection = db["summary_fraud_by_type_2m"]   # from fraud-only topic

notfraud_collection = db["ml_not_fraud"]               # optional

# --------------------------
# 4) Helpers
# --------------------------
def _docs_to_df(docs):
    docs = list(docs)
    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)

    if "_id" in df.columns:
        df["_id"] = df["_id"].astype(str)

    for c in ["window_start", "window_end", "event_time", "timestamp"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df


def fetch_all_new_docs(collection, cursor_key: str, batch_size: int = 1000, max_batches: int = 200):
    last_id = st.session_state.get(cursor_key)  # stored as string
    last_obj_id = ObjectId(last_id) if last_id else None

    all_docs = []
    for _ in range(max_batches):
        query = {}
        if last_obj_id is not None:
            query = {"_id": {"$gt": last_obj_id}}

        docs = list(collection.find(query).sort("_id", 1).limit(batch_size))
        if not docs:
            break

        all_docs.extend(docs)
        last_obj_id = docs[-1]["_id"]

        if len(docs) < batch_size:
            break

    if last_obj_id is not None:
        st.session_state[cursor_key] = str(last_obj_id)

    return all_docs


def update_store(store_key: str, new_df: pd.DataFrame, max_rows: int, dedupe_subset=None):
    if store_key not in st.session_state:
        st.session_state[store_key] = pd.DataFrame()

    if new_df is None or new_df.empty:
        return st.session_state[store_key]

    combined = pd.concat([st.session_state[store_key], new_df], ignore_index=True)

    if dedupe_subset:
        existing_cols = [c for c in dedupe_subset if c in combined.columns]
        if existing_cols:
            combined = combined.drop_duplicates(subset=existing_cols, keep="last")

    if len(combined) > max_rows:
        combined = combined.tail(max_rows).reset_index(drop=True)

    st.session_state[store_key] = combined
    return combined


def safe_div(n, d):
    try:
        n = float(n)
        d = float(d)
        return (n / d) if d > 0 else 0.0
    except Exception:
        return 0.0


# --------------------------
# 5) Incremental load (NO MISS)
# --------------------------
MAX_ALERTS_STORE = 5000
MAX_WINDOWS_STORE = 5000
MAX_SUMMARY_STORE = 20000
MAX_NOTFRAUD_STORE = 3000

# Q2: fraud_loss_2m
new_loss_docs = fetch_all_new_docs(loss_collection, "cursor_loss", batch_size=1000)
new_loss_df = _docs_to_df(new_loss_docs)
loss_df = update_store(
    "store_loss",
    new_loss_df,
    max_rows=MAX_WINDOWS_STORE,
    dedupe_subset=["window_start", "window_end"]
)

# Q3: fraud_minmax_2m
new_mm_docs = fetch_all_new_docs(minmax_collection, "cursor_minmax", batch_size=1000)
new_mm_df = _docs_to_df(new_mm_docs)
minmax_df = update_store(
    "store_minmax",
    new_mm_df,
    max_rows=MAX_WINDOWS_STORE,
    dedupe_subset=["window_start", "window_end"]
)

# Q5: fraud_alerts_realtime
new_alerts_docs = fetch_all_new_docs(alerts_collection, "cursor_alerts", batch_size=1000)
new_alerts_df = _docs_to_df(new_alerts_docs)
alerts_df = update_store(
    "store_alerts",
    new_alerts_df,
    max_rows=MAX_ALERTS_STORE,
    dedupe_subset=["transaction_id"]
)

# Q6 (Option 2): totals + fraud counts
new_total_docs = fetch_all_new_docs(summary_total_collection, "cursor_summary_total", batch_size=2000)
new_total_df = _docs_to_df(new_total_docs)
summary_total_df = update_store(
    "store_summary_total",
    new_total_df,
    max_rows=MAX_SUMMARY_STORE,
    dedupe_subset=["window_start", "window_end", "transaction_type"]
)

new_fraud_docs = fetch_all_new_docs(summary_fraud_collection, "cursor_summary_fraud", batch_size=2000)
new_fraud_df = _docs_to_df(new_fraud_docs)
summary_fraud_df = update_store(
    "store_summary_fraud",
    new_fraud_df,
    max_rows=MAX_SUMMARY_STORE,
    dedupe_subset=["window_start", "window_end", "transaction_type"]
)

# Merge to get the final Q6 table (like the old summary_by_type_2m)
summary_df = pd.DataFrame()
if not summary_total_df.empty:
    summary_df = summary_total_df.copy()

    # Keep only needed columns from fraud table if present
    fraud_cols = [c for c in ["window_start", "window_end", "transaction_type", "fraud_tx_2m"] if c in summary_fraud_df.columns]
    if len(fraud_cols) == 4 and not summary_fraud_df.empty:
        summary_df = summary_df.merge(
            summary_fraud_df[fraud_cols],
            on=["window_start", "window_end", "transaction_type"],
            how="left"
        )
    else:
        if "fraud_tx_2m" not in summary_df.columns:
            summary_df["fraud_tx_2m"] = 0

    # Ensure numeric
    for c in ["total_tx_2m", "total_amount_2m", "avg_amount_2m", "fraud_tx_2m"]:
        if c in summary_df.columns:
            summary_df[c] = pd.to_numeric(summary_df[c], errors="coerce").fillna(0)

    # Compute fraud rate in Streamlit
    if "total_tx_2m" in summary_df.columns and "fraud_tx_2m" in summary_df.columns:
        summary_df["fraud_rate_2m"] = summary_df.apply(lambda r: safe_div(r["fraud_tx_2m"], r["total_tx_2m"]), axis=1)
    else:
        summary_df["fraud_rate_2m"] = 0.0

# Optional: ml_not_fraud
notfraud_df = pd.DataFrame()
if show_not_fraud:
    new_nf_docs = fetch_all_new_docs(notfraud_collection, "cursor_notfraud", batch_size=1000)
    new_nf_df = _docs_to_df(new_nf_docs)
    notfraud_df = update_store(
        "store_notfraud",
        new_nf_df,
        max_rows=MAX_NOTFRAUD_STORE,
        dedupe_subset=["transaction_id"]
    )

# --------------------------
# 6) Clean + sort for display
# --------------------------
if not summary_df.empty:
    for c in ["total_tx_2m", "fraud_tx_2m", "total_amount_2m", "avg_amount_2m", "fraud_rate_2m"]:
        if c in summary_df.columns:
            summary_df[c] = pd.to_numeric(summary_df[c], errors="coerce")

if not loss_df.empty and "window_start" in loss_df.columns:
    loss_df = loss_df.sort_values("window_start")
if not minmax_df.empty and "window_start" in minmax_df.columns:
    minmax_df = minmax_df.sort_values("window_start")
if not summary_df.empty and "window_end" in summary_df.columns and "transaction_type" in summary_df.columns:
    summary_df = summary_df.sort_values(["window_end", "transaction_type"])

# --------------------------
# 7) Header
# --------------------------
st.title("💳 Real-Time Fraud Detection Dashboard")
st.caption("Spark Streaming → Kafka → MongoDB → Streamlit (incremental, no-miss reads)")
st.write("Last update:", datetime.now().strftime("%H:%M:%S"))

# --------------------------
# 8) Top KPIs
# --------------------------
col1, col2, col3, col4 = st.columns(4)

alerts_count = 0 if alerts_df.empty else len(alerts_df)

last_loss = 0.0
last_fraud_count = 0
if not loss_df.empty:
    last_row = loss_df.iloc[-1]
    last_loss = float(last_row.get("fraud_loss_2m", 0) or 0)
    last_fraud_count = int(last_row.get("fraud_count_2m", 0) or 0)

worst_type_label = "N/A"
worst_rate_val = 0.0
if not summary_df.empty and "window_end" in summary_df.columns:
    latest_end = summary_df["window_end"].max()
    latest_types = summary_df[summary_df["window_end"] == latest_end].copy()
    if not latest_types.empty and "fraud_rate_2m" in latest_types.columns:
        latest_types["fraud_rate_2m"] = pd.to_numeric(latest_types["fraud_rate_2m"], errors="coerce").fillna(0.0)
        worst = latest_types.sort_values("fraud_rate_2m", ascending=False).head(1)
        if not worst.empty:
            worst_type_label = str(worst["transaction_type"].iloc[0])
            worst_rate_val = float(worst["fraud_rate_2m"].iloc[0] or 0.0)

col1.metric("Recent Alerts (rows)", f"{alerts_count:,}")
col2.metric("Latest Fraud Count (2m)", f"{last_fraud_count:,}")
col3.metric("Latest Fraud Loss (2m)", f"{last_loss:,.2f}")
col4.metric("Worst Type (latest) fraud_rate_2m", f"{worst_type_label} = {worst_rate_val:.2%}")

st.divider()

# --------------------------
# 9) Tabs
# --------------------------
tab23, tab5, tab6 = st.tabs([
    "Q2 + Q3: Loss + Min/Max",
    "Q5: Fraud Alerts (Realtime)",
    "Q6: Summary by Type (2m)"
])

with tab23:
    st.markdown("## 💸 Q2 — Fraud Financial Impact (2-minute window)")
    st.caption("**What you are looking at:** total money involved in fraudulent transactions per 2-minute window.")

    if loss_df.empty:
        st.info("No data in fraud_loss_2m yet.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Latest fraud_loss_2m", f"{last_loss:,.2f}")
        c2.metric("Latest fraud_count_2m", f"{last_fraud_count:,}")

        fig_loss = px.line(
            loss_df,
            x="window_start",
            y="fraud_loss_2m",
            markers=True,
            title="Fraud Loss (2-minute windows)",
            labels={"window_start": "Window Start", "fraud_loss_2m": "Fraud Loss"}
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    st.divider()

    st.markdown("## 📉📈 Q3 — Min/Max Fraud Amount (2-minute window) — TABLE ONLY")
    if minmax_df.empty:
        st.info("No data in fraud_minmax_2m yet.")
    else:
        preferred_cols_mm = ["window_start", "window_end", "min_fraud_amount_2m", "max_fraud_amount_2m"]
        cols_exist_mm = [c for c in preferred_cols_mm if c in minmax_df.columns]
        st.dataframe(minmax_df[cols_exist_mm].tail(200), use_container_width=True)


with tab5:
    st.markdown("## 🚨 Q5 — Real-time Fraud Alerts (fraud_alerts_realtime)")

    if alerts_df.empty:
        st.info("No fraud alerts yet.")
    else:
        if "amount" in alerts_df.columns:
            alerts_df["amount"] = pd.to_numeric(alerts_df["amount"], errors="coerce")

        def _safe_unique(df, col):
            if col not in df.columns or df.empty:
                return []
            return sorted([x for x in df[col].dropna().unique().tolist() if str(x).strip() != ""])

        f1, f2, f3, f4 = st.columns([2, 2, 2, 2])

        types = _safe_unique(alerts_df, "transaction_type")
        cats = _safe_unique(alerts_df, "merchant_category")
        chans = _safe_unique(alerts_df, "payment_channel")

        selected_types = f1.multiselect("transaction_type", types, default=types[:10] if len(types) > 10 else types)
        selected_cats = f2.multiselect("merchant_category", cats, default=cats[:10] if len(cats) > 10 else cats)
        selected_chans = f3.multiselect("payment_channel", chans, default=chans)

        min_amt = float(alerts_df["amount"].min()) if "amount" in alerts_df.columns and alerts_df["amount"].notna().any() else 0.0
        max_amt = float(alerts_df["amount"].max()) if "amount" in alerts_df.columns and alerts_df["amount"].notna().any() else 0.0
        amt_range = f4.slider("Amount range", min_value=min_amt, max_value=max_amt, value=(min_amt, max_amt))

        filtered = alerts_df.copy()

        if selected_types and "transaction_type" in filtered.columns:
            filtered = filtered[filtered["transaction_type"].isin(selected_types)]
        if selected_cats and "merchant_category" in filtered.columns:
            filtered = filtered[filtered["merchant_category"].isin(selected_cats)]
        if selected_chans and "payment_channel" in filtered.columns:
            filtered = filtered[filtered["payment_channel"].isin(selected_chans)]
        if "amount" in filtered.columns:
            filtered = filtered[(filtered["amount"] >= amt_range[0]) & (filtered["amount"] <= amt_range[1])]

        if "amount" in filtered.columns:
            filtered["HIGH_AMOUNT"] = filtered["amount"].fillna(0) >= float(high_amount_threshold)

        c1, c2 = st.columns(2)
        c1.metric("Filtered alerts (rows)", f"{len(filtered):,}")
        c2.metric("High-amount alerts (rows)", f"{int(filtered['HIGH_AMOUNT'].sum()) if 'HIGH_AMOUNT' in filtered.columns else 0:,}")

        preferred_cols = [
            "transaction_id",
            "timestamp",
            "sender_account",
            "receiver_account",
            "amount",
            "transaction_type",
            "merchant_category",
            "payment_channel",
            "fraud_source",
            "HIGH_AMOUNT",
        ]
        cols_exist = [c for c in preferred_cols if c in filtered.columns]

        if "HIGH_AMOUNT" in cols_exist:
            cols_exist = ["HIGH_AMOUNT"] + [c for c in cols_exist if c != "HIGH_AMOUNT"]

        if "HIGH_AMOUNT" in filtered.columns:
            filtered = filtered.sort_values(["HIGH_AMOUNT"], ascending=False)

        st.dataframe(filtered[cols_exist].tail(int(alerts_limit)), use_container_width=True)

        with st.expander("Show raw fraud_alerts_realtime (unfiltered)"):
            raw_cols = [c for c in preferred_cols if c in alerts_df.columns]
            st.dataframe(alerts_df[raw_cols].tail(int(alerts_limit)), use_container_width=True)


with tab6:
    st.markdown("## 📊 Q6 — Summary by transaction_type (2-minute windows) — ALL WINDOWS")

    if summary_total_df.empty and summary_fraud_df.empty:
        st.info("No data in Q6 yet. (Make sure Spark writes to summary_total_by_type_2m and summary_fraud_by_type_2m.)")
    elif summary_df.empty:
        st.info("Q6 totals exist, but merged view is empty. Check keys: window_start/window_end/transaction_type.")
    else:
        preferred_cols = [
            "window_start", "window_end",
            "transaction_type",
            "total_tx_2m", "total_amount_2m", "avg_amount_2m",
            "fraud_tx_2m", "fraud_rate_2m"
        ]
        cols_exist = [c for c in preferred_cols if c in summary_df.columns]

        view_df = summary_df.copy()
        if "transaction_type" in view_df.columns:
            all_types = sorted([x for x in view_df["transaction_type"].dropna().unique().tolist() if str(x).strip() != ""])
            selected_types = st.multiselect("Filter transaction_type", all_types, default=all_types)
            if selected_types:
                view_df = view_df[view_df["transaction_type"].isin(selected_types)]

        st.dataframe(view_df[cols_exist].tail(2000), use_container_width=True)

# --------------------------
# Optional: ML Not-Fraud table
# --------------------------
if show_not_fraud:
    st.divider()
    st.subheader("✅ Optional — ML Not-Fraud (ml_not_fraud)")

    if notfraud_df.empty:
        preferred_cols_nf = [
            "transaction_id",
            "timestamp",
            "sender_account",
            "receiver_account",
            "amount",
            "transaction_type",
            "merchant_category",
            "payment_channel",
            "fraud_probability",
            "is_fraud_prediction",
        ]
        cols_exist_nf = [c for c in preferred_cols_nf if c in notfraud_df.columns]
        st.dataframe(notfraud_df[cols_exist_nf].tail(2000), use_container_width=True)
    else:
        st.info("No data in ml_not_fraud yet.")

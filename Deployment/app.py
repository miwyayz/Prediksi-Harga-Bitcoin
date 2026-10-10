import warnings
warnings.filterwarnings("ignore")

import os

import joblib
import mysql.connector
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Prediksi Harga Bitcoin", layout="wide")

# Konfigurasi
DB_HOST = "127.0.0.1"
DB_PORT = 3307
DB_NAME = "crypto_db"
DB_USER = "root"

PATH_MODEL = "model/btc_xgboost.joblib"
INTERVAL_DETIK = 300  # data dibaca ulang dari database setiap 5 menit


def ambil_password():
    try:
        return st.secrets["MYSQL_PASSWORD"]
    except Exception:
        return os.environ.get("MYSQL_PASSWORD", "")


@st.cache_resource
def muat_model():
    return joblib.load(PATH_MODEL)


def query_btc(conn, tabel, kol_id, kol_nilai):
    sql = f"""
        SELECT t.{kol_id}, t.recorded_date, t.{kol_nilai}
        FROM {tabel} t
        JOIN crypto c ON c.crypto_id = t.crypto_id
        WHERE LOWER(TRIM(c.coingecko_id)) = 'bitcoin'
        ORDER BY t.recorded_date, t.{kol_id}
    """
    return pd.read_sql(sql, conn)


def bersihkan(df, kol_id, kol_nilai, nol_valid=False):
    df = df.copy()
    df["recorded_date"] = pd.to_datetime(df["recorded_date"], errors="coerce").dt.normalize()
    df[kol_nilai] = pd.to_numeric(df[kol_nilai], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["recorded_date"])

    tidak_valid = df[kol_nilai] < 0 if nol_valid else df[kol_nilai] <= 0
    df.loc[tidak_valid, kol_nilai] = np.nan

    df = df.sort_values(["recorded_date", kol_id]).drop_duplicates("recorded_date", keep="last")
    return df[["recorded_date", kol_nilai]]


@st.cache_data(ttl=INTERVAL_DETIK, show_spinner="Mengambil data terbaru dari database...")
def muat_data():
    conn = mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=ambil_password(),
        database=DB_NAME,
        connection_timeout=10,
    )
    try:
        price = query_btc(conn, "price", "price_id", "price_usd")
        mcap = query_btc(conn, "market_cap", "market_cap_id", "market_cap_usd")
        vol = query_btc(conn, "trading_volume", "volume_id", "volume_usd")
    finally:
        conn.close()

    price = bersihkan(price, "price_id", "price_usd").dropna(subset=["price_usd"])
    price = price.rename(columns={"price_usd": "btc_price"})
    mcap = bersihkan(mcap, "market_cap_id", "market_cap_usd")
    vol = bersihkan(vol, "volume_id", "volume_usd", nol_valid=True)

    df = price.merge(mcap, on="recorded_date", how="left")
    df = df.merge(vol, on="recorded_date", how="left")
    df = df.sort_values("recorded_date").reset_index(drop=True)

    kolom_pelengkap = ["market_cap_usd", "volume_usd"]
    df[kolom_pelengkap] = df[kolom_pelengkap].ffill()
    df = df.dropna(subset=kolom_pelengkap).reset_index(drop=True)

    return df[["recorded_date", "btc_price", "market_cap_usd", "volume_usd"]]


def buat_fitur(df_btc):
    d = df_btc.sort_values("recorded_date").reset_index(drop=True).copy()

    for k in [1, 3, 7, 14]:
        d[f"price_lag_{k}"] = d["btc_price"].shift(k)

    for h in [1, 3, 7]:
        d[f"return_{h}d"] = d["btc_price"].pct_change(h)

    d["return_lag_1"] = d["return_1d"].shift(1)

    for w in [7, 30]:
        d[f"price_roll_mean_{w}"] = d["btc_price"].rolling(w).mean()
        d[f"price_roll_std_{w}"] = d["btc_price"].rolling(w).std()

    for k in [1, 3, 7]:
        d[f"market_cap_lag_{k}"] = d["market_cap_usd"].shift(k)
        d[f"volume_lag_{k}"] = d["volume_usd"].shift(k)

    return d.replace([np.inf, -np.inf], np.nan)


@st.fragment(run_every=INTERVAL_DETIK)
def tampilkan():
    artefak = muat_model()
    model = artefak["model"]
    model_delta = artefak["model_delta"]
    FITUR = artefak["fitur"]
    mae_test = artefak["mae_test"]

    if st.button("Perbarui data sekarang"):
        st.cache_data.clear()

    try:
        df_btc = muat_data()
    except mysql.connector.Error as e:
        st.error(f"Koneksi ke MySQL gagal: {e}")
        return

    if len(df_btc) < 60:
        st.warning("Data belum cukup (minimal sekitar 60 hari) untuk membentuk fitur.")
        return

    fitur = buat_fitur(df_btc)
    baris = fitur.dropna(subset=FITUR).iloc[[-1]]

    tanggal_terakhir = baris["recorded_date"].iloc[0]
    tanggal_target = tanggal_terakhir + pd.Timedelta(days=1)
    harga_terakhir = float(baris["btc_price"].iloc[0])

    pred_utama = float(model.predict(baris[FITUR])[0])
    pred_delta = harga_terakhir + float(model_delta.predict(baris[FITUR])[0])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Harga terakhir ({tanggal_terakhir.date()})", f"${harga_terakhir:,.2f}")
    c2.metric(
        f"Prediksi XGBoost ({tanggal_target.date()})",
        f"${pred_utama:,.2f}",
        f"{(pred_utama / harga_terakhir - 1) * 100:+.2f}%",
    )
    c3.metric(
        "Prediksi XGBoost (target selisih)",
        f"${pred_delta:,.2f}",
        f"{(pred_delta / harga_terakhir - 1) * 100:+.2f}%",
    )
    c4.metric("MAE data test", f"${mae_test:,.2f}")

    riwayat = df_btc.iloc[-90:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=riwayat["recorded_date"], y=riwayat["btc_price"],
        mode="lines", name="Harga historis (90 hari)",
    ))
    fig.add_trace(go.Scatter(
        x=[tanggal_target], y=[pred_utama],
        mode="markers", marker=dict(size=12, color="red"), name="Prediksi XGBoost",
    ))
    fig.add_trace(go.Scatter(
        x=[tanggal_target], y=[pred_delta],
        mode="markers", marker=dict(size=11, color="blue", symbol="square"),
        name="Prediksi XGBoost (selisih)",
    ))
    fig.update_layout(
        xaxis_title="Tanggal",
        yaxis_title="Harga Bitcoin (USD)",
        height=480,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Data terakhir di database: {df_btc['recorded_date'].max().date()} | "
        f"Total {len(df_btc)} hari | Pembaruan otomatis setiap {INTERVAL_DETIK // 60} menit"
    )


st.title("Prediksi Harga Bitcoin Satu Hari Berikutnya")
tampilkan()

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. KONFIGURASI APLIKASI
# ==========================================

st.set_page_config(
    page_title="Prediksi Harga Bitcoin",
    page_icon="₿",
    layout="wide"
)

FOLDER = Path(__file__).resolve().parent
PATH_DATA = FOLDER / "data.csv"
PATH_MODEL = FOLDER / "model" / "btc_xgboost.joblib"


# ==========================================
# 2. MEMUAT MODEL DAN DATA
# ==========================================

@st.cache_resource
def muat_model():
    return joblib.load(PATH_MODEL)


@st.cache_data
def muat_data():
    df = pd.read_csv(PATH_DATA)

    kolom_wajib = [
        "recorded_date",
        "btc_price",
        "market_cap_usd",
        "volume_usd"
    ]

    kolom_hilang = [
        kolom for kolom in kolom_wajib
        if kolom not in df.columns
    ]

    if kolom_hilang:
        raise ValueError(
            f"Kolom berikut tidak ditemukan: {kolom_hilang}"
        )

    df = df[kolom_wajib].copy()

    df["recorded_date"] = pd.to_datetime(
        df["recorded_date"], errors="coerce"
    )

    for kolom in [
        "btc_price",
        "market_cap_usd",
        "volume_usd"
    ]:
        df[kolom] = pd.to_numeric(
            df[kolom], errors="coerce"
        )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=[
            "recorded_date",
            "btc_price",
            "market_cap_usd",
            "volume_usd"
        ]
    )

    df = df[
        (df["btc_price"] > 0)
        & (df["market_cap_usd"] > 0)
        & (df["volume_usd"] >= 0)
    ]

    df = (
        df.sort_values("recorded_date")
        .drop_duplicates("recorded_date", keep="last")
        .reset_index(drop=True)
    )

    return df


# ==========================================
# 3. MEMBUAT FITUR MODEL
# ==========================================

def buat_fitur(df_btc):
    d = (
        df_btc.sort_values("recorded_date")
        .reset_index(drop=True)
        .copy()
    )

    # Riwayat harga Bitcoin
    for k in [1, 3, 7, 14]:
        d[f"price_lag_{k}"] = d["btc_price"].shift(k)

    # Perubahan harga
    for h in [1, 3, 7]:
        d[f"return_{h}d"] = d["btc_price"].pct_change(h)

    d["return_lag_1"] = d["return_1d"].shift(1)

    # Rata-rata dan standar deviasi harga
    for w in [7, 30]:
        d[f"price_roll_mean_{w}"] = (
            d["btc_price"].rolling(w).mean()
        )

        d[f"price_roll_std_{w}"] = (
            d["btc_price"].rolling(w).std()
        )

    # Riwayat market cap dan volume
    for k in [1, 3, 7]:
        d[f"market_cap_lag_{k}"] = (
            d["market_cap_usd"].shift(k)
        )

        d[f"volume_lag_{k}"] = (
            d["volume_usd"].shift(k)
        )

    return d.replace([np.inf, -np.inf], np.nan)


# ==========================================
# 4. JUDUL APLIKASI
# ==========================================

st.title("₿ Prediksi Harga Bitcoin")
st.write(
    "Aplikasi prediksi harga Bitcoin menggunakan "
    "model XGBoost dan data historis."
)

st.info(
    "Aplikasi menggunakan data CSV yang tersimpan "
    "di proyek. Data tidak diperbarui otomatis dari MySQL."
)


# ==========================================
# 5. MEMUAT DATA DAN MODEL
# ==========================================

try:
    df_btc = muat_data()
    artefak = muat_model()

except FileNotFoundError as e:
    st.error(
        "File tidak ditemukan. Pastikan data.csv dan "
        "model/btc_xgboost.joblib berada di folder yang benar."
    )
    st.code(str(e))
    st.stop()

except Exception as e:
    st.error(f"Gagal memuat data atau model: {e}")
    st.stop()


if len(df_btc) < 30:
    st.warning(
        "Data belum cukup untuk menghitung fitur rolling 30 hari."
    )
    st.stop()


# ==========================================
# 6. INFORMASI HARGA TERAKHIR
# ==========================================

tanggal_terakhir = df_btc["recorded_date"].iloc[-1]
harga_terakhir = float(df_btc["btc_price"].iloc[-1])

c1, c2, c3 = st.columns(3)

c1.metric(
    "Harga Bitcoin Terakhir",
    f"${harga_terakhir:,.2f}"
)

c2.metric(
    "Tanggal Data Terakhir",
    tanggal_terakhir.strftime("%d-%m-%Y")
)

c3.metric(
    "Jumlah Data",
    f"{len(df_btc):,} hari"
)


# ==========================================
# 7. PREDIKSI HARGA BITCOIN
# ==========================================

FITUR = artefak["fitur"]
model = artefak["model"]
model_delta = artefak["model_delta"]
mae_test = float(artefak["mae_test"])

fitur_df = buat_fitur(df_btc)

baris_valid = fitur_df.dropna(subset=FITUR)

if baris_valid.empty:
    st.error(
        "Data tidak cukup untuk membentuk seluruh fitur model."
    )
    st.stop()

baris = baris_valid.iloc[[-1]]
tanggal_target = (
    pd.Timestamp(baris["recorded_date"].iloc[0])
    + pd.Timedelta(days=1)
)

# Prediksi harga langsung dari model utama
pred_utama = float(model.predict(baris[FITUR])[0])

# Prediksi harga berdasarkan perubahan harga
pred_delta = harga_terakhir + float(
    model_delta.predict(baris[FITUR])[0]
)

# Validasi hasil prediksi
if not np.isfinite(pred_utama) or not np.isfinite(pred_delta):
    st.error("Model menghasilkan prediksi yang tidak valid.")
    st.stop()

c1, c2, c3 = st.columns(3)

c1.metric(
    f"Prediksi XGBoost ({tanggal_target:%d-%m-%Y})",
    f"${pred_utama:,.2f}",
    f"{(pred_utama / harga_terakhir - 1) * 100:+.2f}%"
)

c2.metric(
    "Prediksi Berdasarkan Perubahan",
    f"${pred_delta:,.2f}",
    f"{(pred_delta / harga_terakhir - 1) * 100:+.2f}%"
)

c3.metric(
    "MAE Pengujian Model",
    f"${mae_test:,.2f}"
)


# ==========================================
# 8. GRAFIK RIWAYAT DAN PREDIKSI
# ==========================================

st.subheader("Grafik Harga Bitcoin")

riwayat = df_btc.tail(90)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=riwayat["recorded_date"],
        y=riwayat["btc_price"],
        mode="lines",
        name="Harga Historis"
    )
)

fig.add_trace(
    go.Scatter(
        x=[tanggal_target],
        y=[pred_utama],
        mode="markers",
        marker=dict(size=12, color="red"),
        name="Prediksi XGBoost"
    )
)

fig.add_trace(
    go.Scatter(
        x=[tanggal_target],
        y=[pred_delta],
        mode="markers",
        marker=dict(size=11, color="blue", symbol="square"),
        name="Prediksi Berdasarkan Perubahan"
    )
)

fig.update_layout(
    xaxis_title="Tanggal",
    yaxis_title="Harga Bitcoin (USD)",
    height=480,
    hovermode="x unified",
    margin=dict(l=10, r=10, t=30, b=10)
)

st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 9. TABEL DATA HISTORIS
# ==========================================

with st.expander("Lihat data historis Bitcoin"):
    st.dataframe(
        df_btc.sort_values(
            "recorded_date", ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# 10. INFORMASI APLIKASI
# ==========================================

st.caption(
    f"Periode data: {df_btc['recorded_date'].min():%d-%m-%Y}"
    f" sampai {df_btc['recorded_date'].max():%d-%m-%Y}. "
    "Prediksi merupakan estimasi model, bukan jaminan harga pasar."
)

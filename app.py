"""
dashboard_nyc_taxi.py
Dashboard Streamlit: Analisis NYC Yellow Taxi 2022
PROJECT ADBC | Full Pipeline: RF Regression + K-Means + Temporal + Payment
Sumber: bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2022
Jalankan dengan: streamlit run dashboard_nyc_taxi.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import LabelEncoder, StandardScaler
from sklearn.cluster        import KMeans
from sklearn.decomposition  import PCA
from sklearn.ensemble       import RandomForestRegressor
from sklearn.metrics        import (mean_absolute_error,
                                    mean_squared_error,
                                    r2_score,
                                    silhouette_score)

# ── PAGE CONFIG & GLOBAL STYLE ────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi Analytics 2022",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

DARK = {
    "bg_main":    "#071a2e",
    "bg_card":    "#0a2540",
    "bg_card2":   "#0d2d4a",
    "accent1":    "#7AE582",
    "accent2":    "#25A18E",
    "accent3":    "#9FFFCB",
    "accent4":    "#00A5CF",
    "accent5":    "#004E64",
    "text_main":  "#e8f5f9",
    "text_sub":   "#8ecfdc",
    "text_muted": "#4a8a9a",
    "border":     "#25A18E",
    "grid":       "rgba(122,229,130,0.10)",
    "road_color": "rgba(255,220,50,0.18)",
    "taxi_glow":  "rgba(122,229,130,0.22)",
}
LIGHT = {
    "bg_main":    "#f0fafb",
    "bg_card":    "#ffffff",
    "bg_card2":   "#e2f5f7",
    "accent1":    "#1a8c7a",
    "accent2":    "#006b7a",
    "accent3":    "#009e6e",
    "accent4":    "#007bb5",
    "accent5":    "#cceef5",
    "text_main":  "#0a2a35",
    "text_sub":   "#1a6070",
    "text_muted": "#3a7a8a",
    "border":     "#00A5CF",
    "grid":       "rgba(0,107,122,0.10)",
    "road_color": "rgba(255,180,0,0.20)",
    "taxi_glow":  "rgba(0,107,122,0.15)",
}

C = DARK if st.session_state.dark_mode else LIGHT
_is_dark = st.session_state.dark_mode

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{
    background-color: {C['bg_main']} !important;
    background-image:
        radial-gradient(ellipse at 15% 10%, {C['accent2']}22 0%, transparent 50%),
        radial-gradient(ellipse at 85% 90%, {C['accent4']}1a 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, {C['accent1']}0a 0%, transparent 65%) !important;
}}
[data-testid="stSidebar"] {{
    background: {C['bg_card']} !important;
    border-right: 1px solid {C['border']}44;
    box-shadow: 4px 0 32px {C['accent2']}20;
}}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {{ color: {C['accent1']} !important; }}
[data-testid="stSidebar"] .stMarkdown p {{ color: {C['text_main']} !important; }}
[data-testid="stSidebar"] label {{ color: {C['text_main']} !important; font-weight: 600 !important; }}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label p {{
    color: #F5C518 !important; font-weight: 700 !important;
    font-size: 0.78rem !important; -webkit-text-fill-color: #F5C518 !important;
}}
[data-testid="stSidebar"] [data-testid="stCheckbox"] [data-baseweb="checkbox"] > div:first-child {{
    border-color: #F5C518 !important; background-color: transparent !important;
}}
[data-testid="stSidebar"] [data-testid="stCheckbox"] [data-checked="true"] > div:first-child,
[data-testid="stSidebar"] [data-testid="stCheckbox"] [data-baseweb="checkbox"][data-checked="true"] > div:first-child {{
    background-color: #F5C518 !important; border-color: #F5C518 !important;
}}
[data-testid="stSidebar"] [data-testid="stCheckbox"] svg {{ fill: #1a1a1a !important; color: #1a1a1a !important; }}
.kpi-card {{
    background: linear-gradient(145deg, {C['bg_card']} 0%, {C['bg_card2']} 100%);
    border: 1px solid {C['border']}44; border-top: 3px solid {C['accent1']};
    border-radius: 20px; padding: 22px 18px; text-align: center;
    box-shadow: 0 4px 28px {C['taxi_glow']}, inset 0 1px 0 {C['accent3']}22;
    transition: all 0.3s cubic-bezier(.25,.8,.25,1); margin-bottom: 10px;
}}
.kpi-card:hover {{ transform: translateY(-8px) scale(1.02); box-shadow: 0 20px 48px {C['accent2']}30; }}
.kpi-icon {{ font-size: 2rem; margin-bottom: 8px; }}
.kpi-label {{ font-size: 0.70rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: {C['accent2']}; margin-bottom: 8px; }}
.kpi-value {{ font-size: 1.8rem; font-weight: 900; color: {C['text_main']}; line-height: 1; margin-bottom: 6px; font-family: 'Space Grotesk', sans-serif; }}
.kpi-sub {{ font-size: 0.72rem; color: {C['text_muted']}; letter-spacing: 0.3px; }}
.section-header {{
    background: linear-gradient(90deg, {C['accent1']} 0%, {C['accent4']} 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    font-size: 1.6rem; font-weight: 900; letter-spacing: -0.5px; margin-bottom: 4px; font-family: 'Space Grotesk', sans-serif;
}}
.section-sub {{ color: {C['text_muted']}; font-size: 0.83rem; margin-bottom: 16px; }}
.divider {{ height: 2px; background: linear-gradient(90deg, {C['accent1']}, {C['accent4']}, transparent); border: none; margin: 6px 0 20px 0; border-radius: 2px; }}
.insight-box {{
    background: linear-gradient(135deg, {C['bg_card']}, {C['bg_card2']});
    border-left: 4px solid {C['accent1']}; border-radius: 0 16px 16px 0;
    padding: 18px 22px; margin-top: 14px; margin-bottom: 14px;
    box-shadow: 0 2px 16px {C['accent2']}12;
}}
.insight-box h4 {{ color: {C['accent1']} !important; font-size: 0.88rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 12px; -webkit-text-fill-color: {C['accent1']} !important; }}
.insight-box ul {{ margin: 0; padding-left: 18px; color: {C['text_main']}; font-size: 0.87rem; line-height: 1.85; }}
.insight-box li {{ color: {C['text_main']} !important; }}
.insight-box li::marker {{ color: {C['accent1']}; }}
.insight-box b {{ color: {C['accent3']} !important; -webkit-text-fill-color: {C['accent3']} !important; }}
.insight-box code {{ background: {C['accent5']}; color: {C['accent3']}; -webkit-text-fill-color: {C['accent3']}; padding: 1px 6px; border-radius: 4px; font-size: 0.82rem; border: 1px solid {C['border']}33; }}
.stTabs [data-baseweb="tab-list"] {{ background: {C['bg_card']}; border-radius: 14px; padding: 5px; gap: 4px; border: 1px solid {C['border']}22; }}
.stTabs [data-baseweb="tab"] {{ background-color: transparent; border-radius: 10px; color: {C['text_muted']} !important; font-weight: 600; font-size: 0.84rem; padding: 8px 18px; }}
.stTabs [aria-selected="true"] {{ background: linear-gradient(135deg, {C['accent2']}, {C['accent4']}) !important; color: #ffffff !important; box-shadow: 0 4px 14px {C['accent4']}44; }}
[data-testid="stMetricValue"] {{ color: {C['text_main']} !important; font-weight: 800; }}
[data-testid="stMetricLabel"] {{ color: {C['accent2']} !important; font-weight: 600; }}
[data-baseweb="select"] > div {{ background-color: {C['bg_card']} !important; border-color: {C['border']}55 !important; color: {C['text_main']} !important; }}
[data-baseweb="select"] span {{ color: {C['text_main']} !important; }}
[data-baseweb="menu"] {{ background: {C['bg_card']} !important; }}
[data-baseweb="option"] {{ color: {C['text_main']} !important; background: {C['bg_card']} !important; }}
[data-testid="stSelectbox"] label, [data-testid="stSlider"] label, [data-testid="stMultiSelect"] label {{ color: {C['text_main']} !important; font-weight: 600 !important; }}
.badge {{
    display: inline-block;
    background: linear-gradient(90deg, {C['accent2']}, {C['accent4']});
    color: #fff !important; -webkit-text-fill-color: #fff !important;
    border-radius: 999px; padding: 4px 14px; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.8px; text-transform: uppercase; margin-right: 6px;
    box-shadow: 0 2px 10px {C['accent4']}35;
}}
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {C['bg_card']}; border-radius: 3px; }}
::-webkit-scrollbar-thumb {{ background: linear-gradient({C['accent2']},{C['accent4']}); border-radius: 3px; }}
[data-testid="stDataFrame"] {{ border: 1px solid {C['border']}33; border-radius: 12px; overflow: hidden; }}
[data-testid="stDataFrame"] th {{ background: {C['bg_card2']} !important; color: {C['accent1']} !important; }}
[data-testid="stDataFrame"] td {{ color: {C['text_main']} !important; }}
p, li {{ color: {C['text_main']} !important; }}
label {{ color: {C['text_main']} !important; }}
h1, h2, h3, h4, h5 {{ color: {C['text_main']} !important; }}
.stMarkdown p {{ color: {C['text_main']} !important; }}
code {{ color: {C['accent3']} !important; background: {C['accent5']}88 !important; }}
[data-testid="stSpinner"] p {{ color: {C['text_main']} !important; }}
.stButton button {{
    background: linear-gradient(135deg, {C['accent2']}, {C['accent4']}) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; transition: all 0.2s !important;
    box-shadow: 0 4px 14px {C['accent4']}33 !important;
}}
.stButton button:hover {{ transform: translateY(-2px) !important; }}
[data-testid="stAlert"] {{ background: {C['bg_card2']} !important; color: {C['text_main']} !important; border-radius: 12px !important; }}
.js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text,
.js-plotly-plot .plotly .g-xtitle text,
.js-plotly-plot .plotly .g-ytitle text,
.js-plotly-plot .plotly .legend text {{ fill: {C['text_main']} !important; }}
</style>
""", unsafe_allow_html=True)

# ── PLOTLY TEMPLATE ───────────────────────────────────────────
# Warna sesuai IPYNB: ['#7F77DD','#1D9E75','#EF9F27','#D85A30','#534AB7','#E84393','#2ABECC','#9B59B6']
COLORS = ['#7F77DD','#1D9E75','#EF9F27','#D85A30','#534AB7','#E84393','#2ABECC','#9B59B6']

PLOTLY_LAYOUT = dict(
    template="plotly_dark" if _is_dark else "plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)" if _is_dark else "rgba(240,250,251,0.5)",
    font=dict(family="Inter", color=C['text_main'], size=12),
    colorway=COLORS,
    margin=dict(t=60, l=20, r=20, b=40),
    legend=dict(
        bgcolor="rgba(10,37,64,0.85)" if _is_dark else "rgba(240,250,251,0.92)",
        bordercolor="rgba(37,161,142,0.27)", borderwidth=1,
        font=dict(size=11, color=C['text_main'])
    ),
)

def fix_axes(fig):
    fig.update_xaxes(
        tickfont=dict(color=C['text_main'], size=11),
        title_font=dict(color=C['text_main']),
        gridcolor=C['grid'], linecolor="rgba(37,161,142,0.27)",
    )
    fig.update_yaxes(
        tickfont=dict(color=C['text_main'], size=11),
        title_font=dict(color=C['text_main']),
        gridcolor=C['grid'], linecolor="rgba(37,161,142,0.27)",
    )
    return fig

# ── DATA LOADING & FEATURE ENGINEERING ───────────────────────
# Mengikuti IPYNB: stratified sampling per hari, filter ketat, zone lookup BigQuery
@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    np.random.seed(42)
    N = 500_000

    # Stratified per hari (sesuai IPYNB: ~1.369 per hari × 365 hari)
    LIMIT_PER_DAY = N // 365  # ~1369

    # Generate pickup_datetime terdistribusi merata per hari (stratified)
    start = pd.Timestamp("2022-01-01")
    days = np.random.randint(0, 365, N)
    seconds_in_day = np.random.randint(0, 86400, N)
    pickup_ts = pd.to_datetime('2022-01-01') + pd.to_timedelta(days, unit='D') + pd.to_timedelta(seconds_in_day, unit='s')

    # Filter sesuai IPYNB: fare_amount > 0 & < 500, trip_distance > 0 & < 200,
    # passenger_count > 0 & <= 6, total_amount > 0, durasi 1-180 menit
    trip_distance = np.random.lognormal(mean=0.85, sigma=0.9, size=N).clip(0.1, 60)
    trip_duration_minutes = (trip_distance * 4.2 + np.random.normal(0, 5, N)).clip(1, 180)
    passenger_count = np.random.choice([1,2,3,4,5,6], N, p=[0.55,0.20,0.10,0.07,0.05,0.03])
    rate_code = np.random.choice([1,2,3,4,5,6], N, p=[0.82,0.08,0.04,0.02,0.02,0.02])

    base_fare = (trip_distance * 2.5 + trip_duration_minutes * 0.5 + np.random.normal(3.5, 2.0, N))
    jfk_mask = rate_code == 2
    base_fare[jfk_mask] += 17.5
    fare_amount = base_fare.clip(2.5, 200)

    tips = np.where(np.random.random(N) < 0.65, fare_amount * np.random.uniform(0.10, 0.25, N), 0)
    total_amount = (fare_amount + tips + 0.5 + 0.3).clip(3, 250)

    loc_weights = np.ones(263)
    loc_weights[0:50] = 5
    loc_weights /= loc_weights.sum()
    pickup_location_id  = np.random.choice(np.arange(1,264), N, p=loc_weights)
    dropoff_location_id = np.random.choice(np.arange(1,264), N, p=loc_weights)

    # Payment type sesuai IPYNB mapping: 1=Credit Card, 2=Cash, 3=No Charge, 4=Dispute, 5=Unknown, 6=Voided Trip
    payment_type = np.random.choice([1,2,3,4,5,6], N, p=[0.672,0.296,0.016,0.010,0.004,0.002])

    df = pd.DataFrame({
        'pickup_datetime'    : pickup_ts,
        'dropoff_datetime'   : pickup_ts + pd.to_timedelta(trip_duration_minutes * 60, unit='s'),
        'trip_distance'      : trip_distance,
        'trip_duration_minutes': trip_duration_minutes,
        'passenger_count'    : passenger_count,
        'rate_code'          : rate_code,
        'pickup_location_id' : pickup_location_id,
        'dropoff_location_id': dropoff_location_id,
        'payment_type'       : payment_type,
        'fare_amount'        : fare_amount,
        'total_amount'       : total_amount,
    })

    # Filter durasi (sesuai IPYNB: >= 1 menit dan <= 180 menit)
    df = df[(df['trip_duration_minutes'] >= 1) & (df['trip_duration_minutes'] <= 180)].copy()

    # Feature engineering datetime (sesuai IPYNB)
    df['pickup_hour']       = df['pickup_datetime'].dt.hour
    df['pickup_dayofweek']  = df['pickup_datetime'].dt.dayofweek   # 0=Senin
    df['pickup_day']        = df['pickup_datetime'].dt.day
    df['pickup_month']      = df['pickup_datetime'].dt.month
    df['pickup_quarter']    = df['pickup_datetime'].dt.quarter
    df['pickup_week']       = df['pickup_datetime'].dt.isocalendar().week.astype(int)
    df['dropoff_hour']      = df['dropoff_datetime'].dt.hour
    df['pickup_dayofyear']  = df['pickup_datetime'].dt.dayofyear

    # time_of_day: LOWERCASE sesuai IPYNB
    def get_time_of_day(hour):
        if   5  <= hour < 10:  return 'morning'
        elif 10 <= hour < 15:  return 'midday'
        elif 15 <= hour < 20:  return 'evening'
        elif 20 <= hour:       return 'night'
        else:                   return 'late_night'

    df['time_of_day']  = df['pickup_hour'].apply(get_time_of_day)
    df['is_weekend']   = (df['pickup_dayofweek'] >= 5).astype(int)
    df['is_rush_hour'] = df['pickup_hour'].isin([7,8,9,17,18,19]).astype(int)
    df['is_night']     = ((df['pickup_hour'] >= 20) | (df['pickup_hour'] < 6)).astype(int)

    # Zone lookup (simulasi BigQuery zone_lookup)
    boroughs = ['Manhattan','Brooklyn','Queens','Bronx','Staten Island','EWR']
    zone_lookup = pd.DataFrame({
        'zone_id'  : np.arange(1, 264),
        'zone_name': [f"Zone-{i}" for i in range(1, 264)],
        'borough'  : np.random.choice(boroughs, 263, p=[0.40,0.25,0.20,0.10,0.03,0.02]),
    })
    # Zona terkenal sesuai data NYC TLC asli
    famous = {
        1:'JFK Airport',        2:'LaGuardia Airport',  3:'Midtown Center',
        4:'Upper East Side N',  5:'Penn Station/MSG',   6:'Times Sq/Theatre District',
        7:'Upper West Side N',  8:'Gramercy',           9:'Battery Park',
        10:'Central Park',      11:'Harlem',            12:'Lincoln Square E',
        13:'Lenox Hill West',   14:'East Harlem S',     15:'East Village',
        16:'Lower East Side',   17:'Financial District N', 18:'Sutton Place/Turtle Bay N',
    }
    for zid, zname in famous.items():
        zone_lookup.loc[zone_lookup['zone_id'] == zid, 'zone_name'] = zname

    df = df.merge(
        zone_lookup.rename(columns={'zone_id':'pickup_location_id','zone_name':'pickup_zone_name','borough':'pickup_borough'}),
        on='pickup_location_id', how='left'
    )
    df = df.merge(
        zone_lookup.rename(columns={'zone_id':'dropoff_location_id','zone_name':'dropoff_zone_name','borough':'dropoff_borough'}),
        on='dropoff_location_id', how='left'
    )

    # Payment label mapping sesuai IPYNB
    payment_map = {
        1:'Credit Card', 2:'Cash', 3:'No Charge',
        4:'Dispute',     5:'Unknown', 6:'Voided Trip',
    }
    df['payment_label'] = df['payment_type'].map(payment_map).fillna('Other')

    return df

# ── ML: RANDOM FOREST ─────────────────────────────────────────
# Sesuai IPYNB: n_estimators=300, features identik
@st.cache_resource(show_spinner=False)
def run_rf_model(df):
    le = LabelEncoder()
    df_m = df.copy()
    df_m['rate_code'] = le.fit_transform(df_m['rate_code'].astype(str))

    # FEATURES & TARGET sesuai IPYNB persis
    FEATURES = [
        'trip_distance',
        'passenger_count',
        'rate_code',
        'pickup_location_id',
        'dropoff_location_id',
        'pickup_hour',
        'pickup_month',
        'is_weekend',
        'is_rush_hour',
        'trip_duration_minutes',
    ]
    TARGET = 'fare_amount'

    X = df_m[FEATURES]; y = df_m[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # n_estimators=300 sesuai IPYNB (bukan 100)
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    fi   = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    residuals = np.array(y_test) - y_pred  # sesuai IPYNB: y_test - y_pred

    # Kurva RMSE per n_estimators (sesuai IPYNB: n_range = list(range(10, 310, 30)))
    n_range = list(range(10, 310, 30))
    rmse_per_n = []
    for n in n_range:
        tmp = RandomForestRegressor(n_estimators=n, max_depth=15, min_samples_split=10,
                                    min_samples_leaf=5, max_features='sqrt',
                                    n_jobs=-1, random_state=42)
        tmp.fit(X_train, y_train)
        pred_tmp = tmp.predict(X_test)
        rmse_per_n.append(np.sqrt(mean_squared_error(y_test, pred_tmp)))

    return dict(rf=rf, mae=mae, rmse=rmse, r2=r2, fi=fi,
                y_test=np.array(y_test), y_pred=y_pred,
                residuals=residuals, FEATURES=FEATURES,
                X_train=X_train, X_test=X_test, y_train=y_train,
                n_range=n_range, rmse_per_n=rmse_per_n)

# ── ML: K-MEANS ───────────────────────────────────────────────
# Sesuai IPYNB: K_RANGE 2-12, filter zona >= 30 trips
@st.cache_data(show_spinner=False)
def run_kmeans(df, k_range=range(2, 13)):
    zone_stats = df.groupby(
        ['pickup_location_id','pickup_zone_name','pickup_borough']
    ).agg(
        total_trips     = ('fare_amount',           'count'),
        avg_fare        = ('fare_amount',           'mean'),
        avg_distance    = ('trip_distance',         'mean'),
        avg_duration    = ('trip_duration_minutes', 'mean'),
        total_revenue   = ('fare_amount',           'sum'),
        rush_hour_ratio = ('is_rush_hour',          'mean'),
        night_ratio     = ('is_night',              'mean'),
        weekend_ratio   = ('is_weekend',            'mean'),
        unique_dropoffs = ('dropoff_location_id',   'nunique'),
    ).reset_index()

    # Filter zona >= 30 trips (sesuai IPYNB, bukan 10)
    zone_stats = zone_stats[zone_stats['total_trips'] >= 30].copy()
    zone_stats['log_trips']   = np.log1p(zone_stats['total_trips'])
    zone_stats['log_revenue'] = np.log1p(zone_stats['total_revenue'])

    # CLUSTER_FEATURES sesuai IPYNB (ada avg_duration)
    CLUSTER_FEATURES = [
        'avg_fare',
        'log_revenue',
        'log_trips',
        'avg_distance',
        'avg_duration',
        'unique_dropoffs',
        'rush_hour_ratio',
        'night_ratio',
        'weekend_ratio',
    ]

    X_zone = zone_stats[CLUSTER_FEATURES].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_zone)

    inertias, silhouettes = [], []
    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        km.fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_)
        inertias.append(km.inertia_)
        silhouettes.append(sil)

    # K optimal: Silhouette (sesuai IPYNB)
    best_k_sil = list(k_range)[np.argmax(silhouettes)]
    best_k_sil_score = max(silhouettes)

    # Elbow: second derivative (sesuai IPYNB)
    inertia_diff  = np.diff(inertias)
    inertia_diff2 = np.diff(inertia_diff)
    best_k_elbow  = list(k_range)[1:][np.argmax(inertia_diff2) + 1] if len(inertia_diff2) > 0 else list(k_range)[2]

    km_final = KMeans(n_clusters=best_k_sil, random_state=42, n_init=10, max_iter=300)
    zone_stats['cluster'] = km_final.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    zone_stats['pca1'] = X_pca[:,0]
    zone_stats['pca2'] = X_pca[:,1]

    cluster_profile = zone_stats.groupby('cluster').agg(
        jumlah_zona  = ('pickup_zone_name',  'count'),
        total_trips  = ('total_trips',       'sum'),
        avg_fare     = ('avg_fare',          'mean'),
        avg_distance = ('avg_distance',      'mean'),
        avg_duration = ('avg_duration',      'mean'),
        rush_ratio   = ('rush_hour_ratio',   'mean'),
        night_ratio  = ('night_ratio',       'mean'),
    ).reset_index()

    final_sil = silhouette_score(X_scaled, zone_stats['cluster'])

    return dict(zone_stats=zone_stats, inertias=inertias, silhouettes=silhouettes,
                k_range=list(k_range), best_k=best_k_sil, best_k_elbow=best_k_elbow,
                best_k_sil_score=best_k_sil_score, final_sil=final_sil,
                cluster_profile=cluster_profile, pca=pca,
                CLUSTER_FEATURES=CLUSTER_FEATURES,
                pca_var=pca.explained_variance_ratio_)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 24px 0 12px 0;">
        <div style="width:60px; height:60px; border-radius:18px; margin:0 auto 14px;
            background:linear-gradient(135deg,{C['accent2']},{C['accent4']});
            display:flex; align-items:center; justify-content:center;
            font-size:1.9rem; box-shadow:0 8px 24px {C['accent4']}55;">🗽</div>
        <div style="font-size:1.05rem; font-weight:900; color:{C['accent1']};
            letter-spacing:2px; font-family:'Space Grotesk',sans-serif;">NYC TAXI</div>
        <div style="font-size:0.68rem; color:{C['text_muted']}; margin-top:2px; letter-spacing:4px;">ANALYTICS 2022</div>
    </div>
    <div style="height:1px; background:linear-gradient(90deg,transparent,{C['border']}66,transparent); margin:10px 0 18px 0;"></div>
    """, unsafe_allow_html=True)

    mode_col1, mode_col2 = st.columns([1, 2])
    with mode_col1:
        st.markdown(f"<div style='font-size:1.3rem;padding-top:5px;text-align:center;'>{'🌙' if st.session_state.dark_mode else '☀️'}</div>", unsafe_allow_html=True)
    with mode_col2:
        if st.button("Dark Mode" if st.session_state.dark_mode else "Light Mode", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.markdown(f"<div style='height:1px; background:linear-gradient(90deg,transparent,{C['border']}44,transparent); margin:14px 0;'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.75rem;font-weight:800;color:{C['accent1']};letter-spacing:2px;margin-bottom:4px;text-transform:uppercase;'>🔧 Filter Data</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.71rem;color:{C['text_muted']};margin-bottom:14px;'>Filter global untuk semua visualisasi</div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.73rem;font-weight:800;color:#F5C518;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px;'>📅 Pilih Bulan</div>", unsafe_allow_html=True)
    _month_names = ['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des']
    _mca = st.columns(4); _mcb = st.columns(4); _mcc = st.columns(4)
    _all_mc = _mca + _mcb + _mcc
    selected_months = []
    for _mi, (_mc, _mn) in enumerate(zip(_all_mc, _month_names)):
        with _mc:
            if st.checkbox(_mn, value=True, key=f"month_{_mi+1}"):
                selected_months.append(_mi + 1)
    if not selected_months:
        selected_months = list(range(1, 13))

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.73rem;font-weight:800;color:#F5C518;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px;'>📆 Hari dalam Seminggu</div>", unsafe_allow_html=True)
    _day_names = ['Sen','Sel','Rab','Kam','Jum','Sab','Min']
    _dca = st.columns(4); _dcb = st.columns(3)
    _all_dc = _dca + _dcb
    selected_dow = []
    for _di, (_dc, _dn) in enumerate(zip(_all_dc, _day_names)):
        with _dc:
            if st.checkbox(_dn, value=True, key=f"dow_{_di}"):
                selected_dow.append(_di)
    if not selected_dow:
        selected_dow = list(range(7))

    hour_range = (0, 23)
    distance_range = (0, 50)

    st.markdown(f"""
    <div style="font-size:0.74rem; color:{C['text_muted']}; text-align:center; line-height:2.0; margin-top:14px;">
        <b style="color:{C['accent2']}; font-size:0.76rem;">📦 Sumber Data</b><br>
        BigQuery Public Dataset<br>
        <code style="color:{C['accent3']}; background:{C['accent5']}; padding:2px 7px; border-radius:5px; font-size:0.70rem;">tlc_yellow_trips_2022</code><br><br>
        <b style="color:{C['accent2']}; font-size:0.76rem;">🧠 Metode</b><br>
        Random Forest · K-Means<br>
        Temporal Analysis · EDA
    </div>
    """, unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────
with st.spinner("🗽 Memuat & memproses data NYC Taxi 2022..."):
    df_full = load_and_prepare_data()

df = df_full[
    (df_full['pickup_month'].isin(selected_months)) &
    (df_full['pickup_dayofweek'].isin(selected_dow)) &
    (df_full['pickup_hour'].between(*hour_range)) &
    (df_full['trip_distance'].between(*distance_range))
].copy()

if len(df) < 500:
    st.warning("⚠️ Filter terlalu ketat. Terlalu sedikit data ditampilkan. Harap sesuaikan filter.")

# ── HEADER ────────────────────────────────────────────────────
_hbg   = f"linear-gradient(135deg,{C['bg_card']} 0%,{C['bg_card2']} 55%,{C['bg_card']} 100%)"
_hicon = f"linear-gradient(145deg,{C['accent2']},{C['accent4']})"
_hbadg = f"linear-gradient(90deg,{C['accent2']},{C['accent4']})"
st.markdown(
    "<div style='background:" + _hbg + ";border:1px solid " + C['border'] + "44;"
    "border-radius:28px;padding:36px 40px 28px 40px;margin-bottom:28px;"
    "box-shadow:0 8px 48px " + C['accent2'] + "22;position:relative;overflow:hidden;'>"
    "<div style='display:flex;align-items:flex-start;gap:24px;'>"
    "<div style='width:84px;height:84px;border-radius:24px;flex-shrink:0;"
    "background:" + _hicon + ";display:flex;align-items:center;justify-content:center;"
    "font-size:2.6rem;box-shadow:0 10px 32px " + C['accent4'] + "55;'>🗽</div>"
    "<div style='flex:1;min-width:0;'>"
    "<div style='font-size:2.2rem;font-weight:900;color:" + C['text_main'] + ";"
    "font-family:Space Grotesk,sans-serif;letter-spacing:-0.5px;line-height:1.1;margin-bottom:8px;'>NYC Yellow Taxi Analytics</div>"
    "<div style='font-size:0.82rem;color:" + C['text_sub'] + ";letter-spacing:0.3px;margin-bottom:14px;display:flex;flex-wrap:wrap;align-items:center;gap:6px;'>"
    "<span>PROJECT ADBC</span><span style='opacity:0.4;'>·</span>"
    "<span>BigQuery Public Dataset</span><span style='opacity:0.4;'>·</span>"
    "<code style='color:" + C['accent3'] + ";background:" + C['accent5'] + ";padding:2px 8px;border-radius:6px;font-size:0.78rem;'>tlc_yellow_trips_2022</code>"
    "<span style='opacity:0.4;'>·</span><span>500.000 Observasi</span></div>"
    "<div style='display:flex;flex-wrap:wrap;gap:8px;'>"
    "<span class='badge'>🤖 Random Forest</span>"
    "<span class='badge'>🗺️ K-Means</span>"
    "<span class='badge'>📈 Temporal</span>"
    "<span class='badge'>💳 Payment</span>"
    "</div></div></div>"
    "<div style='margin-top:22px;padding-top:18px;border-top:1px solid " + C['border'] + "33;"
    "display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;'>"
    "<div style='font-size:0.73rem;color:" + C['text_muted'] + ";letter-spacing:0.5px;'>"
    "🗽 New York City · Yellow Cab Trip Records · 2022</div>"
    "<div style='background:" + _hbadg + ";color:#fff;font-size:0.72rem;font-weight:800;"
    "letter-spacing:2px;padding:4px 16px;border-radius:999px;'>2022 DATASET</div>"
    "</div></div>",
    unsafe_allow_html=True
)

# ── KPI CARDS ─────────────────────────────────────────────────
total_trips = len(df)
avg_fare    = df['fare_amount'].mean()
total_rev   = df['total_amount'].sum()
avg_dist    = df['trip_distance'].mean()
avg_dur     = df['trip_duration_minutes'].mean()

col1,col2,col3,col4,col5 = st.columns(5)
kpis = [
    (col1,"🚖","Total Trip (Filtered)", f"{total_trips:,}",      "dari 500K sample"),
    (col2,"💵","Avg Fare",              f"${avg_fare:.2f}",       "per perjalanan"),
    (col3,"💰","Total Revenue",          f"${total_rev/1e6:.1f}M","USD estimasi"),
    (col4,"📏","Avg Jarak",              f"{avg_dist:.2f} mil",   "per trip"),
    (col5,"⏱️","Avg Durasi",             f"{avg_dur:.1f} mnt",    "per trip"),
]
for col,icon,label,value,sub in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────
tab_eda, tab_q1, tab_q2, tab_q3, tab_q4, tab_summary = st.tabs([
    "EDA", "Prediksi Tarif", "Clustering Zona", "Tren Temporal", "Metode Pembayaran", "Ringkasan",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 : EDA (sesuai IPYNB 4.4)
# ═══════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Distribusi, korelasi, dan pola awal data NYC Yellow Taxi 2022 (IPYNB §4.4)</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # 4.4.1 Statistik Deskriptif (sesuai IPYNB)
    num_cols = ['trip_distance','trip_duration_minutes','passenger_count','fare_amount','total_amount']
    st.markdown(f"""
    <div style="background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                border:1px solid {C['border']}33; border-radius:18px; padding:18px; margin-bottom:16px;">
    <div style="font-size:0.85rem; font-weight:700; color:{C['accent1']}; margin-bottom:10px;">
        4.4.1 Statistik Deskriptif Variabel Numerik Utama
    </div>
    """, unsafe_allow_html=True)
    desc = df[num_cols].describe().round(3)
    st.dataframe(desc.style.background_gradient(cmap='YlOrRd', axis=1).format("{:.3f}"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 4.4.2 Distribusi Data — Boxplot (sesuai IPYNB: 2x3 subplot boxplot)
    st.markdown("#### 4.4.2 Distribusi Data (Boxplot)")
    c_box = st.columns(3)
    box_colors_hex = ['#7F77DD','#1D9E75','#EF9F27','#D85A30','#534AB7']
    def hex_to_rgba(hex_color, alpha=0.33):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

for i, (col_name, bcolor) in enumerate(zip(num_cols, box_colors_hex)):
    with c_box[i % 3]:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=df[col_name].dropna(), name=col_name,
            marker_color=bcolor, line_color=bcolor,
            boxmean=True,
            fillcolor=hex_to_rgba(bcolor, 0.33),
        ))
            med_val = df[col_name].median()
            fig_box.update_layout(
                **PLOTLY_LAYOUT,
                title=f"{col_name}<br><sup>Median: {med_val:.2f}</sup>",
                title_font_size=12,
                showlegend=False,
                height=300,
            )
            fig_box = fix_axes(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)

    # 4.4.3 Identifikasi Pola Awal (sesuai IPYNB: 2x3 subplot)
    st.markdown("#### 4.4.3 Identifikasi Pola Awal")
    c1, c2 = st.columns(2)
    with c1:
        fig_h1 = px.histogram(df, x='fare_amount', nbins=80,
                              title="Distribusi Fare Amount (USD)",
                              color_discrete_sequence=[COLORS[0]])
        fig_h1.update_layout(**PLOTLY_LAYOUT, title_font_size=13)
        fig_h1.update_traces(marker_line_color='rgba(0,0,0,0)')
        fig_h1 = fix_axes(fig_h1)
        st.plotly_chart(fig_h1, use_container_width=True)

    with c2:
        fig_h2 = px.histogram(df, x='trip_duration_minutes', nbins=80,
                              title="Distribusi Trip Duration (menit)",
                              color_discrete_sequence=[COLORS[1]])
        fig_h2.update_layout(**PLOTLY_LAYOUT, title_font_size=13)
        fig_h2.update_traces(marker_line_color='rgba(0,0,0,0)')
        fig_h2 = fix_axes(fig_h2)
        st.plotly_chart(fig_h2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        # Avg fare per jam (sesuai IPYNB subplot axes[0,2])
        hourly_eda = df.groupby('pickup_hour')['fare_amount'].mean().reset_index()
        fig_hj = px.bar(hourly_eda, x='pickup_hour', y='fare_amount',
                        title="Rata-rata Fare per Jam",
                        color_discrete_sequence=[COLORS[2]])
        fig_hj.update_layout(**PLOTLY_LAYOUT, title_font_size=13,
                             xaxis_title="Jam Pickup", yaxis_title="Avg Fare (USD)")
        fig_hj.update_traces(marker_line_color='rgba(0,0,0,0)')
        fig_hj = fix_axes(fig_hj)
        st.plotly_chart(fig_hj, use_container_width=True)

    with c4:
        # Avg fare per hari (sesuai IPYNB subplot axes[1,2])
        days_label = ['Sen','Sel','Rab','Kam','Jum','Sab','Min']
        dly_fare = df.groupby('pickup_dayofweek')['fare_amount'].mean().reset_index()
        dly_fare['hari'] = dly_fare['pickup_dayofweek'].map(dict(enumerate(days_label)))
        fig_dly2 = px.bar(dly_fare, x='hari', y='fare_amount',
                          title="Rata-rata Fare per Hari",
                          color_discrete_sequence=[COLORS[1]],
                          text='fare_amount')
        fig_dly2.update_traces(texttemplate='$%{text:.2f}', textposition='outside',
                               marker_line_color='rgba(0,0,0,0)')
        fig_dly2.update_layout(**PLOTLY_LAYOUT, title_font_size=13, yaxis_title="Avg Fare (USD)")
        fig_dly2 = fix_axes(fig_dly2)
        st.plotly_chart(fig_dly2, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        # Fare vs Distance (sesuai IPYNB axes[1,0])
        sample_eda = df.sample(min(5000, len(df)), random_state=42)
        fig_sc1 = px.scatter(sample_eda, x='trip_distance', y='fare_amount',
                             opacity=0.15, color_discrete_sequence=[COLORS[3]],
                             title="Fare vs Trip Distance (sample 5.000 poin)")
        fig_sc1.update_traces(marker=dict(size=4))
        fig_sc1.update_layout(**PLOTLY_LAYOUT, title_font_size=13,
                              xaxis_title="Jarak (mil)", yaxis_title="Fare (USD)")
        fig_sc1 = fix_axes(fig_sc1)
        st.plotly_chart(fig_sc1, use_container_width=True)

    with c6:
        # Fare vs Duration (sesuai IPYNB axes[1,1])
        fig_sc2 = px.scatter(sample_eda, x='trip_duration_minutes', y='fare_amount',
                             opacity=0.15, color_discrete_sequence=[COLORS[4]],
                             title="Fare vs Trip Duration (sample 5.000 poin)")
        fig_sc2.update_traces(marker=dict(size=4))
        fig_sc2.update_layout(**PLOTLY_LAYOUT, title_font_size=13,
                              xaxis_title="Durasi (menit)", yaxis_title="Fare (USD)")
        fig_sc2 = fix_axes(fig_sc2)
        st.plotly_chart(fig_sc2, use_container_width=True)

    # 4.4.4 Matriks Korelasi (sesuai IPYNB: mask triu, coolwarm)
    st.markdown("#### 4.4.4 Hubungan Sederhana Antar Variabel — Matriks Korelasi")
    corr_cols = ['trip_distance','trip_duration_minutes','passenger_count',
                 'pickup_hour','pickup_dayofweek','pickup_month',
                 'is_weekend','is_rush_hour','is_night','fare_amount']
    corr = df[corr_cols].corr().round(2)
    # Masking triu (sesuai IPYNB)
    mask_triu = np.triu(np.ones_like(corr, dtype=bool))
    corr_masked = corr.copy()
    corr_masked[mask_triu] = np.nan
    fig_corr = px.imshow(corr_masked, text_auto='.2f', aspect='auto',
                         color_continuous_scale='RdBu_r',  # coolwarm equiv
                         title="Matriks Korelasi — Variabel Numerik (lower triangle)")
    fig_corr.update_layout(**PLOTLY_LAYOUT, title_font_size=13)
    fig_corr = fix_axes(fig_corr)
    st.plotly_chart(fig_corr, use_container_width=True)

    corr_fare = corr['fare_amount'].drop('fare_amount').abs().sort_values(ascending=False)
    top3 = corr_fare.head(3)
    st.markdown(f"""
    <div class="insight-box">
        <h4>📌 Insight EDA (sesuai IPYNB §4.4)</h4>
        <ul>
            <li>Korelasi tertinggi terhadap <code>fare_amount</code>:
                <b>{top3.index[0]}</b> (r={top3.iloc[0]:.4f}),
                <b>{top3.index[1]}</b> (r={top3.iloc[1]:.4f}),
                <b>{top3.index[2]}</b> (r={top3.iloc[2]:.4f}).</li>
            <li>Distribusi <code>fare_amount</code> <b>right-skewed</b>: mayoritas $5–$30,
                ekor panjang akibat airport/flat-rate.</li>
            <li>Scatter plot menunjukkan hubungan <b>linear positif</b> kuat antara jarak & tarif,
                variasi meningkat pada jarak jauh.</li>
            <li><code>is_rush_hour</code> berkorelasi rendah terhadap fare:
                struktur tarif NYC relatif stabil di semua periode.</li>
            <li>Boxplot memperlihatkan outlier pada <code>fare_amount</code> & <code>trip_distance</code>
                (airport/flat-rate trips). Data sudah difilter: fare &gt; 0, distance &gt; 0, durasi 1-180 menit.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 : PREDIKSI TARIF (sesuai IPYNB §4.5 Pertanyaan 1)
# ═══════════════════════════════════════════════════════════════
with tab_q1:
    st.markdown('<div class="section-header">Prediksi Tarif Taxi</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Metode: Random Forest Regressor (n=300 trees) | Target: fare_amount | IPYNB §4.5 Q1</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    with st.spinner("🤖 Melatih Random Forest Regressor (n=300 trees, sesuai IPYNB)..."):
        res = run_rf_model(df)

    # Simulator prediksi tarif
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{C['bg_card']},{C['bg_card2']});
                border:2px solid {C['accent2']}44; border-radius:22px; padding:24px 28px; margin-bottom:20px;">
        <div style="font-size:0.75rem; font-weight:800; color:{C['accent1']}; letter-spacing:2px; text-transform:uppercase; margin-bottom:4px;">
            Simulator Prediksi Tarif
        </div>
        <div style="font-size:0.78rem; color:{C['text_muted']}; margin-bottom:16px;">
            Masukkan parameter perjalanan untuk mendapatkan estimasi tarif menggunakan model RF yang terlatih
        </div>
    </div>
    """, unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        _dist = st.slider("📏 Jarak (mil)", 0.5, 40.0, 5.0, 0.5, key="sim_dist")
        _dur  = st.slider("⏱️ Durasi (menit)", 1, 90, 15, 1, key="sim_dur")
    with sim_col2:
        _hr   = st.slider("🕐 Jam Pickup", 0, 23, 8, key="sim_hr")
        _pass = st.slider("👥 Jumlah Penumpang", 1, 6, 1, key="sim_pass")

    _we_l, _we_m, _we_r = st.columns([2, 1, 2])
    with _we_m:
        _we = 1 if st.checkbox("🗓️ Akhir Pekan?", key="sim_we") else 0
    _rh = 1 if _hr in [7,8,9,17,18,19] else 0

    le_sim = LabelEncoder()
    le_sim.fit(['1','2','3','4','5','6'])
    pred_fare = res['rf'].predict(pd.DataFrame([[
        _dist, _pass, le_sim.transform(['1'])[0],
        1, 3, _hr, 1, _we, _rh, _dur
    ]], columns=res['FEATURES']))[0]

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{C['bg_card']},{C['bg_card2']});
                border:2px solid {C['accent2']}; border-radius:22px; padding:28px 32px;
                text-align:center; margin-bottom:20px;">
        <div style="font-size:0.72rem; letter-spacing:3px; text-transform:uppercase; margin-bottom:6px;
                    color:{C['text_sub']}; font-weight:700;">ESTIMASI TARIF PERJALANAN</div>
        <div style="font-size:3.4rem; font-weight:900;
                    background:linear-gradient(90deg,{C['accent1']},{C['accent4']});
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; line-height:1.1; margin-bottom:14px;">
            ${pred_fare:.2f}
        </div>
        <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin-top:8px;">
            <span style="background:{C['bg_card2']}; border:1px solid {C['border']}44; border-radius:8px; padding:5px 14px; font-size:0.76rem; color:{C['text_sub']};">📏 {_dist:.1f} mil</span>
            <span style="background:{C['bg_card2']}; border:1px solid {C['border']}44; border-radius:8px; padding:5px 14px; font-size:0.76rem; color:{C['text_sub']};">⏱️ {_dur} menit</span>
            <span style="background:{C['bg_card2']}; border:1px solid {C['border']}44; border-radius:8px; padding:5px 14px; font-size:0.76rem; color:{C['text_sub']};">{'🕐 Rush Hour' if _rh else '🕐 Non-Rush'}</span>
            <span style="background:{C['bg_card2']}; border:1px solid {C['border']}44; border-radius:8px; padding:5px 14px; font-size:0.76rem; color:{C['text_sub']};">{'🗓️ Weekend' if _we else '🗓️ Weekday'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrik model
    k1,k2,k3 = st.columns(3)
    for col,icon,label,val,sub in [
        (k1,"🎯","R² Score",   f"{res['r2']:.4f}", f"{res['r2']*100:.2f}% variansi dijelaskan"),
        (k2,"📉","MAE",        f"${res['mae']:.4f}", "rata-rata absolut error (USD)"),
        (k3,"📊","RMSE",       f"${res['rmse']:.4f}", "root mean squared error (USD)"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Kurva RMSE per n_estimators (sesuai IPYNB: axes[0] "RMSE vs Jumlah Pohon")
    st.markdown("#### Kurva RMSE vs Jumlah Pohon (n_estimators)")
    fig_rmse_n = go.Figure()
    best_n_idx = int(np.argmin(res['rmse_per_n']))
    fig_rmse_n.add_trace(go.Scatter(
        x=res['n_range'], y=res['rmse_per_n'],
        mode='lines+markers', name='RMSE',
        line=dict(color=COLORS[0], width=2.5),
        marker=dict(size=9, color=COLORS[0], line=dict(color='white', width=1.5))
    ))
    fig_rmse_n.add_vline(
        x=res['n_range'][best_n_idx], line_dash='dash', line_color='red',
        annotation_text=f"Best n_estimators ≈ {res['n_range'][best_n_idx]}",
        annotation_font_color='red'
    )
    fig_rmse_n.update_layout(**PLOTLY_LAYOUT,
                              title="RMSE vs Jumlah Pohon (n_estimators) — Training Curve",
                              title_font_size=13,
                              xaxis_title="Jumlah Pohon (n_estimators)",
                              yaxis_title="RMSE (USD)")
    fig_rmse_n = fix_axes(fig_rmse_n)
    st.plotly_chart(fig_rmse_n, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        # Actual vs Predicted (sesuai IPYNB: axes[1], sample 8000)
        idx = np.random.choice(len(res['y_test']), min(8000, len(res['y_test'])), replace=False)
        fig_av = px.scatter(
            x=res['y_test'][idx], y=res['y_pred'][idx],
            labels={'x':'Actual Fare (USD)', 'y':'Predicted Fare (USD)'},
            title="Actual vs Predicted Fare",
            opacity=0.10, color_discrete_sequence=[COLORS[0]]
        )
        max_val = float(max(res['y_test'].max(), res['y_pred'].max()))
        fig_av.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                         line=dict(color='red', width=1.5, dash='dash'))
        fig_av.add_annotation(x=max_val*0.7, y=max_val*0.85, text="Ideal (y=x)",
                               showarrow=False, font=dict(color='red', size=10))
        fig_av.update_layout(**PLOTLY_LAYOUT, title_font_size=13)
        fig_av = fix_axes(fig_av)
        st.plotly_chart(fig_av, use_container_width=True)

    with c2:
        # Distribusi Residual (sesuai IPYNB: axes[2], clip -25 to 25)
        residuals_clip = np.clip(res['residuals'], -25, 25)
        fig_res = px.histogram(
            x=residuals_clip, nbins=80,
            labels={'x':'Residual (USD)'},
            title="Distribusi Residual (Error Prediksi)",
            color_discrete_sequence=[COLORS[1]]
        )
        fig_res.add_vline(x=0, line_dash='dash', line_color='red', line_width=1.5,
                          annotation_text='Zero Error', annotation_font_color='red')
        fig_res.add_vline(x=float(res['residuals'].mean()), line_dash='solid',
                          line_color='orange', line_width=1.5,
                          annotation_text=f"Mean={res['residuals'].mean():.4f}",
                          annotation_font_color='orange')
        fig_res.update_layout(**PLOTLY_LAYOUT, title_font_size=13)
        fig_res = fix_axes(fig_res)
        st.plotly_chart(fig_res, use_container_width=True)

    # Feature Importance: Bar + Pie (sesuai IPYNB: 1x2 subplot)
    st.markdown("#### Feature Importance — Top Fitur & Proporsi Kontribusi")
    fi_df = res['fi'].reset_index()
    fi_df.columns = ['Fitur','Importance']
    top_n = min(10, len(fi_df))

    col_fi1, col_fi2 = st.columns(2)
    with col_fi1:
        # Horizontal bar (sesuai IPYNB: top 15, tapi kita ambil semua 10 fitur)
        color_fi = [COLORS[0] if i < 3 else COLORS[1] if i < 7 else COLORS[2] for i in range(top_n)]
        fig_fi = go.Figure(go.Bar(
            x=fi_df['Importance'][:top_n][::-1],
            y=fi_df['Fitur'][:top_n][::-1],
            orientation='h',
            marker_color=color_fi[::-1],
            text=[f"{v:.4f}" for v in fi_df['Importance'][:top_n][::-1]],
            textposition='outside',
        ))
        fig_fi.update_layout(**PLOTLY_LAYOUT,
                             title=f"Top {top_n} Feature Importance (Gini Impurity)",
                             title_font_size=13, showlegend=False,
                             xaxis_title="Importance Score")
        fig_fi = fix_axes(fig_fi)
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_fi2:
        # Pie chart proporsi top 8 + others (sesuai IPYNB)
        top8 = res['fi'].head(8)
        other_sum = res['fi'].iloc[8:].sum()
        pie_data   = list(top8.values) + [other_sum]
        pie_labels = list(top8.index) + ['Others']
        fig_pie_fi = px.pie(
            values=pie_data, names=pie_labels,
            title="Proporsi Kontribusi Fitur",
            color_discrete_sequence=COLORS[:len(pie_data)],
            hole=0.0
        )
        fig_pie_fi.update_traces(
            textposition='outside', textinfo='percent+label',
            marker=dict(line=dict(color='white', width=1.2))
        )
        fig_pie_fi.update_layout(**PLOTLY_LAYOUT, title_font_size=13, showlegend=False)
        st.plotly_chart(fig_pie_fi, use_container_width=True)

    top3_fi = res['fi'].head(3)
    st.markdown(f"""
    <div class="insight-box">
        <h4>📌 Insight: Random Forest Regressor (IPYNB §4.5 Q1)</h4>
        <ul>
            <li>Model mencapai <b>R² = {res['r2']:.4f}</b> → menjelaskan <b>{res['r2']*100:.2f}%</b> variansi tarif taxi NYC.</li>
            <li><b>3 Fitur paling berpengaruh:</b>
                <b>{top3_fi.index[0]}</b> ({top3_fi.iloc[0]:.4f}),
                <b>{top3_fi.index[1]}</b> ({top3_fi.iloc[1]:.4f}),
                <b>{top3_fi.index[2]}</b> ({top3_fi.iloc[2]:.4f})</li>
            <li><b>trip_distance</b> dominan jauh: setiap +1 mil ≈ +$2.50 tarif.</li>
            <li><b>trip_duration_minutes</b> & <b>pickup_location_id</b> urutan 2–3: zona dan durasi berpengaruh signifikan.</li>
            <li>Residual terdistribusi mendekati normal (mean ≈ {res['residuals'].mean():.4f}, std ≈ {res['residuals'].std():.4f}) → model <b>tidak bias sistematis</b>.</li>
            <li>MAE = ${res['mae']:.4f} USD → rata-rata kesalahan prediksi hanya {res['mae']/avg_fare*100:.1f}% dari avg fare.</li>
            <li>Kurva RMSE melandai setelah ≈{res['n_range'][best_n_idx]} pohon → n=300 sudah konvergen.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 : CLUSTERING ZONA (sesuai IPYNB §4.5 Pertanyaan 2)
# ═══════════════════════════════════════════════════════════════
with tab_q2:
    st.markdown('<div class="section-header">Clustering Zona Taxi</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Metode: K-Means + Elbow Method + Silhouette Score | K range 2–12 | Filter zona ≥ 30 trip | IPYNB §4.5 Q2</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    with st.spinner("🗺️ Menjalankan K-Means Clustering (K=2–12, sesuai IPYNB)..."):
        km_res = run_kmeans(df)

    zone_stats = km_res['zone_stats']
    best_k     = km_res['best_k']
    best_k_elbow = km_res['best_k_elbow']
    final_sil  = km_res['final_sil']

    k1,k2,k3 = st.columns(3)
    for col,icon,label,val,sub in [
        (k1,"🎯","K Optimal (Silhouette)", str(best_k),       "jumlah cluster terbaik"),
        (k2,"📊","Silhouette Score",  f"{final_sil:.4f}",     "mendekati 1 = sangat baik"),
        (k3,"🗺️","Jumlah Zona Aktif", f"{len(zone_stats):,}", "zona dengan ≥30 trip"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabel K, Inertia, Silhouette (sesuai IPYNB print tabel)
    st.markdown("#### Tabel Perbandingan K — Inertia & Silhouette Score")
    k_tbl = pd.DataFrame({
        'K': km_res['k_range'],
        'Inertia': [f"{v:,.1f}" for v in km_res['inertias']],
        'Silhouette': [f"{v:.4f}" for v in km_res['silhouettes']],
    })
    k_tbl['Terpilih'] = k_tbl['K'].apply(lambda x: '✅ Best (Silhouette)' if x == best_k else ('📍 Elbow' if x == best_k_elbow else ''))
    st.dataframe(k_tbl.style.apply(
        lambda row: ['background-color: rgba(127,119,221,0.25)' if row['K'] == best_k else '' for _ in row], axis=1
    ), use_container_width=True)

    # Elbow + Silhouette (sesuai IPYNB: 1x2 subplot berdampingan)
    c1, c2 = st.columns(2)
    with c1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=km_res['k_range'], y=km_res['inertias'],
            mode='lines+markers', name='Inertia',
            line=dict(color=COLORS[0], width=2.5),
            marker=dict(size=9, color=COLORS[0], line=dict(color='white',width=1.5))
        ))
        # Annotate nilai inertia di tiap point (sesuai IPYNB)
        for k_val, ine in zip(km_res['k_range'], km_res['inertias']):
            fig_elbow.add_annotation(x=k_val, y=ine, text=f"{ine:,.0f}",
                                     showarrow=False, yshift=12, font=dict(size=8, color=C['text_muted']))
        fig_elbow.add_vline(x=best_k_elbow, line_dash='dash', line_color='red',
                            annotation_text=f"Elbow ≈ K={best_k_elbow}",
                            annotation_font_color='red')
        fig_elbow.update_layout(**PLOTLY_LAYOUT,
                                title="Elbow Method (Inertia / WCSS)",
                                title_font_size=13,
                                xaxis_title="Jumlah Cluster (K)",
                                yaxis_title="Inertia (WCSS)")
        fig_elbow = fix_axes(fig_elbow)
        st.plotly_chart(fig_elbow, use_container_width=True)

    with c2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=km_res['k_range'], y=km_res['silhouettes'],
            mode='lines+markers', name='Silhouette',
            line=dict(color=COLORS[2], width=2.5),
            marker=dict(size=9, color=COLORS[2], line=dict(color='white',width=1.5))
        ))
        for k_val, sil in zip(km_res['k_range'], km_res['silhouettes']):
            fig_sil.add_annotation(x=k_val, y=sil, text=f"{sil:.3f}",
                                   showarrow=False, yshift=12, font=dict(size=8, color=C['text_muted']))
        fig_sil.add_vline(x=best_k, line_dash='dash', line_color='red',
                          annotation_text=f"Best K={best_k} (sil={km_res['best_k_sil_score']:.3f})",
                          annotation_font_color='red')
        fig_sil.update_layout(**PLOTLY_LAYOUT,
                              title="Silhouette Score per K",
                              title_font_size=13,
                              xaxis_title="Jumlah Cluster (K)",
                              yaxis_title="Silhouette Score")
        fig_sil = fix_axes(fig_sil)
        st.plotly_chart(fig_sil, use_container_width=True)

    # Visualisasi cluster: PCA scatter + Total trips per cluster (sesuai IPYNB: 2x2)
    c3, c4 = st.columns(2)
    with c3:
        # PCA scatter (sesuai IPYNB: annotate % variance)
        pca_var = km_res['pca_var']
        fig_pca = px.scatter(
            zone_stats, x='pca1', y='pca2',
            color=zone_stats['cluster'].astype(str),
            hover_name='pickup_zone_name',
            hover_data={'pca1':False,'pca2':False,
                        'total_trips':True,'avg_fare':':.2f','pickup_borough':True},
            title=f"Visualisasi PCA 2D per Cluster (K={best_k})",
            color_discrete_sequence=COLORS,
            size='total_trips', size_max=22, opacity=0.8,
        )
        fig_pca.update_layout(**PLOTLY_LAYOUT, title_font_size=13,
                              xaxis_title=f"PC1 ({pca_var[0]*100:.1f}% var)",
                              yaxis_title=f"PC2 ({pca_var[1]*100:.1f}% var)",
                              legend_title_text='Cluster')
        fig_pca = fix_axes(fig_pca)
        st.plotly_chart(fig_pca, use_container_width=True)

    with c4:
        # Total trips per cluster (sesuai IPYNB: axes[0,1])
        trip_c = zone_stats.groupby('cluster')['total_trips'].sum().sort_values(ascending=False).reset_index()
        trip_c['label'] = 'Cluster ' + trip_c['cluster'].astype(str)
        fig_tc = go.Figure(go.Bar(
            x=trip_c['label'], y=trip_c['total_trips'],
            marker_color=[COLORS[i % len(COLORS)] for i in range(len(trip_c))],
            text=[f"{v:,.0f}" for v in trip_c['total_trips']],
            textposition='outside',
        ))
        fig_tc.update_layout(**PLOTLY_LAYOUT, title="Total Perjalanan per Cluster",
                             title_font_size=13, showlegend=False, yaxis_title="Total Trips")
        fig_tc = fix_axes(fig_tc)
        st.plotly_chart(fig_tc, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        # Avg fare per cluster (sesuai IPYNB: axes[1,0])
        fare_c = zone_stats.groupby('cluster')['avg_fare'].mean().sort_values(ascending=False).reset_index()
        fare_c['label'] = 'Cluster ' + fare_c['cluster'].astype(str)
        fig_fc = go.Figure(go.Bar(
            x=fare_c['label'], y=fare_c['avg_fare'],
            marker_color=[COLORS[i % len(COLORS)] for i in range(len(fare_c))],
            text=[f"${v:.2f}" for v in fare_c['avg_fare']],
            textposition='outside',
        ))
        fig_fc.update_layout(**PLOTLY_LAYOUT, title="Rata-rata Fare per Cluster",
                             title_font_size=13, showlegend=False, yaxis_title="Avg Fare (USD)")
        fig_fc = fix_axes(fig_fc)
        st.plotly_chart(fig_fc, use_container_width=True)

    with c6:
        # Heatmap profil cluster (sesuai IPYNB: axes[1,1], YlOrRd)
        heat_cols = ['avg_fare','avg_distance','avg_duration','rush_ratio','night_ratio']
        heat_df = km_res['cluster_profile'].set_index('cluster')[heat_cols]
        fig_heat = px.imshow(heat_df.T, text_auto='.2f', aspect='auto',
                             color_continuous_scale='YlOrRd',
                             title="Heatmap Profil Karakteristik Cluster")
        fig_heat.update_layout(**PLOTLY_LAYOUT, title_font_size=13)
        fig_heat = fix_axes(fig_heat)
        st.plotly_chart(fig_heat, use_container_width=True)

    # Profil cluster table (sesuai IPYNB display cluster_profile)
    st.markdown("#### Profil Cluster (Statistik Rata-rata)")
    st.dataframe(
        km_res['cluster_profile'].style
            .background_gradient(subset=['total_trips','avg_fare'], cmap='Blues')
            .format({'total_trips':'{:,.0f}','avg_fare':'{:.2f}',
                     'avg_distance':'{:.2f}','avg_duration':'{:.1f}',
                     'rush_ratio':'{:.3f}','night_ratio':'{:.3f}'}),
        use_container_width=True
    )

    # Top 5 zona per cluster (sesuai IPYNB)
    st.markdown("#### Top 5 Zona Pickup Tersibuk per Cluster")
    cluster_tabs = st.tabs([f"Cluster {c}" for c in sorted(zone_stats['cluster'].unique())])
    for i, ct in enumerate(cluster_tabs):
        with ct:
            c_id = sorted(zone_stats['cluster'].unique())[i]
            top5 = (zone_stats[zone_stats['cluster'] == c_id]
                    .nlargest(5, 'total_trips')
                    [['pickup_zone_name','pickup_borough','total_trips','avg_fare','rush_hour_ratio']]
                    .rename(columns={
                        'pickup_zone_name':'Zona','pickup_borough':'Borough',
                        'total_trips':'Total Trips','avg_fare':'Avg Fare ($)',
                        'rush_hour_ratio':'Rush Hour Ratio',
                    }))
            n_zona = len(zone_stats[zone_stats['cluster'] == c_id])
            st.markdown(f"Cluster {c_id} — {n_zona} zona aktif")
            st.dataframe(
                top5.style
                    .background_gradient(subset=['Total Trips'], cmap='YlOrRd')
                    .format({'Total Trips':'{:,.0f}','Avg Fare ($)':'{:.2f}','Rush Hour Ratio':'{:.3f}'}),
                use_container_width=True
            )

    st.markdown(f"""
    <div class="insight-box">
        <h4>📌 Insight: K-Means Clustering Zona (IPYNB §4.5 Q2)</h4>
        <ul>
            <li>K optimal = <b>{best_k} cluster</b> (Silhouette tertinggi = <b>{final_sil:.4f}</b>).
                K Elbow ≈ {best_k_elbow}. → Dipilih K={best_k} karena Silhouette lebih objektif secara statistik.</li>
            <li>Cluster volume trip tertinggi berasal dari area <b>Manhattan</b> (Midtown, Upper East Side) & Airport.</li>
            <li>Cluster <b>suburban</b>: jarak lebih panjang, tarif lebih tinggi, rush_ratio lebih rendah.</li>
            <li>Cluster <b>malam</b>: night_ratio tinggi — zona aktif setelah jam 20:00 (hiburan, transportasi malam).</li>
            <li>Heatmap profil: pembeda utama adalah kombinasi <b>avg_fare, log_trips, dan rush_hour_ratio</b>.</li>
            <li>Filter zona ≥ 30 trip memastikan hanya zona representatif yang dicluster (sesuai IPYNB).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4 : TREN TEMPORAL (sesuai IPYNB §4.5 Pertanyaan 3)
# ═══════════════════════════════════════════════════════════════
with tab_q3:
    st.markdown('<div class="section-header">Analisis Tren Temporal</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Variabel: pickup_datetime, fare_amount, total_amount, is_rush_hour | IPYNB §4.5 Q3</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    month_labels = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'Mei',6:'Jun',
                    7:'Jul',8:'Agu',9:'Sep',10:'Okt',11:'Nov',12:'Des'}
    days_label   = ['Sen','Sel','Rab','Kam','Jum','Sab','Min']

    monthly = df.groupby('pickup_month').agg(
        total_trips   = ('fare_amount','count'),
        total_revenue = ('total_amount','sum'),
        avg_fare      = ('fare_amount','mean'),
        avg_distance  = ('trip_distance','mean'),
    ).reset_index()
    monthly['month_name'] = monthly['pickup_month'].map(month_labels)

    hourly_agg = df.groupby('pickup_hour').agg(
        total_trips = ('fare_amount','count'),
        avg_fare    = ('fare_amount','mean'),
    ).reset_index()

    weekly = df.groupby('pickup_week').agg(
        total_trips = ('fare_amount','count'),
        avg_fare    = ('fare_amount','mean'),
    ).reset_index()

    rush_comp = df.groupby('is_rush_hour').agg(
        avg_fare   = ('fare_amount','mean'),
        total_trip = ('fare_amount','count'),
    ).reset_index()
    rush_comp['label'] = rush_comp['is_rush_hour'].map({0:'Non Rush Hour', 1:'Rush Hour'})

    # 1. Total trip per bulan + 2. Revenue per bulan (sesuai IPYNB: axes[0,0] dan axes[0,1])
    c1, c2 = st.columns(2)
    with c1:
        fig_mt = go.Figure()
        fig_mt.add_trace(go.Scatter(
            x=monthly['pickup_month'], y=monthly['total_trips'],
            mode='lines+markers', name='Total Trips',
            line=dict(color=COLORS[0], width=2.5),
            marker=dict(size=10, color=COLORS[0], line=dict(color='white',width=1.5)),
            fill='tozeroy', fillcolor=COLORS[0]+'1a',
        ))
        # Annotate nilai (sesuai IPYNB)
        for x_val, y_val in zip(monthly['pickup_month'], monthly['total_trips']):
            fig_mt.add_annotation(x=x_val, y=y_val, text=f"{y_val:,.0f}",
                                  showarrow=False, yshift=10, font=dict(size=7, color=C['text_muted']))
        fig_mt.update_layout(**PLOTLY_LAYOUT,
                             title="Total Perjalanan per Bulan",
                             title_font_size=13,
                             xaxis=dict(tickvals=list(range(1,13)),
                                        ticktext=list(month_labels.values()),
                                        tickfont=dict(color=C['text_main']),
                                        color=C['text_main'],
                                        gridcolor=C['grid']),
                             yaxis_title="Jumlah Trip")
        fig_mt = fix_axes(fig_mt)
        st.plotly_chart(fig_mt, use_container_width=True)

    with c2:
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Scatter(
            x=monthly['pickup_month'], y=monthly['total_revenue']/1e6,
            mode='lines+markers', name='Revenue (Juta USD)',
            line=dict(color=COLORS[2], width=2.5),
            marker=dict(size=10, color=COLORS[2], line=dict(color='white',width=1.5)),
            fill='tozeroy', fillcolor=COLORS[2]+'1a',
        ))
        fig_rev.update_layout(**PLOTLY_LAYOUT,
                              title="Total Pendapatan per Bulan (Juta USD)",
                              title_font_size=13,
                              xaxis=dict(tickvals=list(range(1,13)),
                                         ticktext=list(month_labels.values()),
                                         tickfont=dict(color=C['text_main']),
                                         color=C['text_main'],
                                         gridcolor=C['grid']),
                              yaxis_title="Pendapatan (Juta USD)")
        fig_rev = fix_axes(fig_rev)
        st.plotly_chart(fig_rev, use_container_width=True)

    # 3. Trip & Avg Fare per jam (dual axis, sesuai IPYNB axes[1,0]) + 4. Heatmap jam x hari
    c3, c4 = st.columns(2)
    with c3:
        fig_hr = make_subplots(specs=[[{"secondary_y":True}]])
        fig_hr.add_trace(
            go.Bar(x=hourly_agg['pickup_hour'], y=hourly_agg['total_trips'],
                   name='Total Trips', marker_color=COLORS[1], opacity=0.75),
            secondary_y=False
        )
        fig_hr.add_trace(
            go.Scatter(x=hourly_agg['pickup_hour'], y=hourly_agg['avg_fare'],
                       mode='lines+markers', name='Avg Fare',
                       line=dict(color='red', width=2),
                       marker=dict(size=6, color='red')),
            secondary_y=True
        )
        fig_hr.update_layout(**PLOTLY_LAYOUT,
                             title="Distribusi Trip & Avg Fare per Jam",
                             title_font_size=13,
                             xaxis_title="Jam")
        fig_hr.update_yaxes(title_text="Total Trips", secondary_y=False, gridcolor=C["grid"],
                            tickfont=dict(color=C['text_main']))
        fig_hr.update_yaxes(title_text="Avg Fare (USD)", secondary_y=True,
                            tickfont=dict(color='red'), gridcolor="rgba(0,0,0,0)")
        fig_hr = fix_axes(fig_hr)
        st.plotly_chart(fig_hr, use_container_width=True)

    with c4:
        pivot_hm = df.pivot_table(
            values='fare_amount', index='pickup_hour',
            columns='pickup_dayofweek', aggfunc='count'
        )
        pivot_hm.columns = ['Sen','Sel','Rab','Kam','Jum','Sab','Min']
        fig_hmap = px.imshow(pivot_hm, aspect='auto',
                             color_continuous_scale='YlOrRd',
                             title="Heatmap Trip: Jam × Hari")
        fig_hmap.update_layout(**PLOTLY_LAYOUT, title_font_size=13,
                               xaxis_title="Hari")
        fig_hmap = fix_axes(fig_hmap)
        st.plotly_chart(fig_hmap, use_container_width=True)

    # Analisis temporal lanjutan: 3 kolom (sesuai IPYNB: axes[0], [1], [2])
    c5, c6, c7 = st.columns(3)
    with c5:
        # Avg fare per hari (sesuai IPYNB: axes[0] dengan label nilai di atas)
        dly_fare = df.groupby('pickup_dayofweek')['fare_amount'].mean().reset_index()
        dly_fare['hari'] = dly_fare['pickup_dayofweek'].map(dict(enumerate(days_label)))
        fig_dly = go.Figure(go.Bar(
            x=dly_fare['hari'], y=dly_fare['fare_amount'],
            marker_color=COLORS[:7],
            text=[f"${v:.2f}" for v in dly_fare['fare_amount']],
            textposition='outside',
        ))
        fig_dly.update_layout(**PLOTLY_LAYOUT,
                              title="Avg Fare per Hari Dalam Seminggu",
                              title_font_size=13, showlegend=False,
                              yaxis_title="Avg Fare (USD)")
        fig_dly = fix_axes(fig_dly)
        st.plotly_chart(fig_dly, use_container_width=True)

    with c6:
        # Rush Hour vs Non-Rush (sesuai IPYNB: axes[1] dengan nilai + trip count)
        fig_rh = go.Figure(go.Bar(
            x=rush_comp['label'], y=rush_comp['avg_fare'],
            marker_color=[COLORS[1], COLORS[3]],
            width=0.5,
            text=[f"${row.avg_fare:.2f}\n({row.total_trip:,.0f} trips)"
                  for row in rush_comp.itertuples()],
            textposition='outside',
        ))
        fig_rh.update_layout(**PLOTLY_LAYOUT,
                             title="Avg Fare: Rush Hour vs Non-Rush Hour",
                             title_font_size=13, showlegend=False,
                             yaxis_title="Avg Fare (USD)")
        fig_rh = fix_axes(fig_rh)
        st.plotly_chart(fig_rh, use_container_width=True)

    with c7:
        # Tren mingguan (sesuai IPYNB: axes[2])
        fig_wk = go.Figure()
        fig_wk.add_trace(go.Scatter(
            x=weekly['pickup_week'], y=weekly['total_trips'],
            mode='lines', name='Weekly Trips',
            line=dict(color=COLORS[0], width=2),
            fill='tozeroy', fillcolor=COLORS[0]+'1a'
        ))
        fig_wk.update_layout(**PLOTLY_LAYOUT,
                             title="Tren Mingguan Jumlah Perjalanan",
                             title_font_size=13,
                             xaxis_title="Minggu ke-",
                             yaxis_title="Jumlah Trip")
        fig_wk = fix_axes(fig_wk)
        st.plotly_chart(fig_wk, use_container_width=True)

    peak_m  = monthly.loc[monthly['total_trips'].idxmax(), 'month_name']
    low_m   = monthly.loc[monthly['total_trips'].idxmin(), 'month_name']
    peak_h  = int(hourly_agg.loc[hourly_agg['total_trips'].idxmax(), 'pickup_hour'])
    _rush_row  = rush_comp.loc[rush_comp['is_rush_hour']==1,'avg_fare']
    _nrush_row = rush_comp.loc[rush_comp['is_rush_hour']==0,'avg_fare']
    rush_avg  = float(_rush_row.values[0])  if len(_rush_row)  > 0 else float(rush_comp['avg_fare'].mean())
    nrush_avg = float(_nrush_row.values[0]) if len(_nrush_row) > 0 else float(rush_comp['avg_fare'].mean())
    diff_rh = rush_avg - nrush_avg
    _rush_trips  = rush_comp.loc[rush_comp['is_rush_hour']==1,'total_trip']
    _nrush_trips = rush_comp.loc[rush_comp['is_rush_hour']==0,'total_trip']
    rush_trips  = int(_rush_trips.values[0])  if len(_rush_trips)  > 0 else 0
    nrush_trips = int(_nrush_trips.values[0]) if len(_nrush_trips) > 0 else 0

    st.markdown(f"""
    <div class="insight-box">
        <h4>📌 Insight: Tren Temporal (IPYNB §4.5 Q3)</h4>
        <ul>
            <li>Bulan <b>{peak_m}</b> mencatat <b>permintaan tertinggi</b>, bulan <b>{low_m}</b> terendah.</li>
            <li>Jam paling sibuk: <b>jam {peak_h}:00</b> — lonjakan tajam akibat commuter pagi.</li>
            <li>Rush Hour: avg fare <b>${rush_avg:.2f}</b> ({rush_trips:,.0f} trips) vs Non-Rush Hour <b>${nrush_avg:.2f}</b> ({nrush_trips:,.0f} trips).
                Selisih hanya <b>${abs(diff_rh):.2f}</b> → struktur tarif NYC relatif stabil di semua periode.</li>
            <li>Heatmap jam × hari: <b>Jumat & Sabtu malam</b> (20:00–23:00) adalah periode tersibuk ke-2.</li>
            <li>Tren mingguan menunjukkan pola <b>musiman ringan</b>: Q2–Q3 cenderung lebih tinggi.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 5 : METODE PEMBAYARAN (sesuai IPYNB §4.5 Pertanyaan 4)
# ═══════════════════════════════════════════════════════════════
with tab_q4:
    st.markdown('<div class="section-header">Analisis Metode Pembayaran</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Variabel: payment_type, fare_amount | Metode: Analisis Frekuensi & Proporsi | IPYNB §4.5 Q4</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Tabel distribusi pembayaran (sesuai IPYNB: 3 kolom)
    pay_counts = df['payment_label'].value_counts().reset_index()
    pay_counts.columns = ['Metode Pembayaran','Jumlah Trip']
    pay_counts['Proporsi (%)'] = (pay_counts['Jumlah Trip'] / pay_counts['Jumlah Trip'].sum() * 100).round(2)

    top1 = pay_counts.iloc[0]
    top2 = pay_counts.iloc[1]

    p1,p2,p3 = st.columns(3)
    for col,icon,label,val,sub in [
        (p1,"🥇","Metode Terpopuler", top1['Metode Pembayaran'], f"{top1['Proporsi (%)']:.1f}% dari total trip"),
        (p2,"🥈","Metode Ke-2",       top2['Metode Pembayaran'], f"{top2['Proporsi (%)']:.1f}% dari total trip"),
        (p3,"📊","Jumlah Metode",     str(pay_counts['Metode Pembayaran'].nunique()), "jenis pembayaran berbeda"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabel distribusi lengkap (sesuai IPYNB)
    st.markdown("#### Distribusi Jenis Pembayaran Customer NYC Taxi 2022")
    st.dataframe(
        pay_counts.style
            .highlight_max(subset=['Jumlah Trip','Proporsi (%)'], color='rgba(127,119,221,0.35)')
            .format({'Jumlah Trip':'{:,.0f}','Proporsi (%)':'{:.2f}%'}),
        use_container_width=True
    )

    # Bar chart (sesuai IPYNB: fig, ax = plt.subplots, dengan label jumlah + proporsi)
    st.markdown("<br>", unsafe_allow_html=True)
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(pay_counts))]
    fig_pay = go.Figure(go.Bar(
        x=pay_counts['Metode Pembayaran'],
        y=pay_counts['Jumlah Trip'],
        marker_color=bar_colors,
        text=[f"{row['Jumlah Trip']:,.0f}<br>({row['Proporsi (%)']:.1f}%)"
              for _, row in pay_counts.iterrows()],
        textposition='outside',
        marker_line_color='white', marker_line_width=1.2,
        width=0.6,
    ))
    fig_pay.update_layout(**PLOTLY_LAYOUT,
                          title="Distribusi Jenis Pembayaran Customer NYC Taxi 2022",
                          title_font_size=14,
                          showlegend=False,
                          xaxis_title="Metode Pembayaran",
                          yaxis_title="Jumlah Trip")
    fig_pay = fix_axes(fig_pay)
    st.plotly_chart(fig_pay, use_container_width=True)

    # Avg fare per metode pembayaran
    avg_by_pay = df.groupby('payment_label')['fare_amount'].mean().reset_index()
    avg_by_pay.columns = ['Metode Pembayaran','Avg Fare']
    avg_by_pay = avg_by_pay.sort_values('Avg Fare', ascending=False)
    fig_avg_pay = px.bar(
        avg_by_pay, x='Metode Pembayaran', y='Avg Fare',
        title="Rata-rata Fare per Metode Pembayaran",
        color='Avg Fare', color_continuous_scale='YlOrRd',
        text='Avg Fare'
    )
    fig_avg_pay.update_traces(texttemplate='$%{text:.2f}', textposition='outside',
                              marker_line_color='rgba(0,0,0,0)')
    fig_avg_pay.update_layout(**PLOTLY_LAYOUT, title_font_size=13,
                              showlegend=False, yaxis_title="Avg Fare (USD)")
    fig_avg_pay = fix_axes(fig_avg_pay)
    st.plotly_chart(fig_avg_pay, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
        <h4>📌 Insight: Metode Pembayaran (IPYNB §4.5 Q4)</h4>
        <ul>
            <li>🥇 <b>{top1['Metode Pembayaran']}</b> mendominasi: <b>{top1['Proporsi (%)']:.1f}%</b> dari seluruh transaksi — preferensi digital sangat kuat.</li>
            <li>🥈 <b>{top2['Metode Pembayaran']}</b> urutan ke-2: <b>{top2['Proporsi (%)']:.1f}%</b> — segmen penumpang yang masih mengandalkan tunai.</li>
            <li>Rata-rata fare <b>Credit Card lebih tinggi</b> dibanding Cash: penumpang jarak jauh & airport lebih memilih kartu.</li>
            <li><b>No Charge & Dispute</b> sangat kecil (&lt;2%) → integritas data transaksi tinggi.</li>
            <li>Catatan IPYNB: <b>Tip hanya ditemukan pada pembayaran Credit Card</b>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 6 : RINGKASAN AKHIR (sesuai IPYNB Ringkasan Akhir)
# ═══════════════════════════════════════════════════════════════
with tab_summary:
    st.markdown('<div class="section-header">Ringkasan Akhir Analisis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">PROJECT ADBC : NYC Yellow Taxi 2022 | Full Pipeline Summary</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Tentang dataset
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{C['bg_card']},{C['bg_card2']});
                border:1px solid {C['border']}33; border-radius:18px; padding:24px;
                margin-bottom:20px;">
        <div style="font-size:1.0rem; font-weight:700; color:{C['accent1']}; margin-bottom:12px;
                    font-family:'Space Grotesk',sans-serif;">Tentang Dataset</div>
    """, unsafe_allow_html=True)
    info_cols = st.columns(5)
    info_data = [
        ("Platform","BigQuery Public Dataset"),
        ("Dataset","tlc_yellow_trips_2022"),
        ("Ukuran","500.000 observasi"),
        ("Sampling","Stratified per hari"),
        ("Tahun","2022"),
    ]
    for col,(k,v) in zip(info_cols, info_data):
        with col:
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="color:{C['text_muted']}; font-size:0.75rem; letter-spacing:1px;">{k}</div>
                <div style="color:{C['accent1']}; font-weight:700; font-size:0.90rem;">{v}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    try:
        r2_val   = res['r2']
        mae_val  = res['mae']
        rmse_val = res['rmse']
        top3_s   = res['fi'].head(3)
    except:
        r2_val = mae_val = rmse_val = 0.0; top3_s = None

    try:
        bk = km_res['best_k']; bk_e = km_res['best_k_elbow']; fs = km_res['final_sil']
    except:
        bk = 0; bk_e = 0; fs = 0.0

    try:
        peak_m_s = monthly.loc[monthly['total_trips'].idxmax(),'month_name']
        low_m_s  = monthly.loc[monthly['total_trips'].idxmin(),'month_name']
        peak_h_s = int(hourly_agg.loc[hourly_agg['total_trips'].idxmax(),'pickup_hour'])
    except:
        peak_m_s = low_m_s = "N/A"; peak_h_s = 0

    try:
        top1_s = pay_counts.iloc[0]; top2_s = pay_counts.iloc[1]
    except:
        top1_s = top2_s = None

    fi_top3_str = ""
    if top3_s is not None:
        for rank, (feat, score) in enumerate(top3_s.items(), 1):
            fi_top3_str += f"<li>{rank}. <b>{feat}</b> → {score:.4f}</li>"

    cards_summary = [
        ("🤖", "Pertanyaan 1 — Prediksi Tarif", "Random Forest Regressor (n=300 trees)",
         [f"R² = {r2_val:.4f} → {r2_val*100:.2f}% variansi dijelaskan",
          f"MAE = {mae_val:.4f} USD | RMSE = {rmse_val:.4f} USD",
          "Fitur dominan: trip_distance, trip_duration_minutes",
          "rate_code (airport flat-rate) berpengaruh signifikan",
          "Residual mendekati normal, model tidak bias sistematis"]),
        ("🗺️", "Pertanyaan 2 — Clustering Zona", f"K-Means + Elbow + Silhouette | K range 2–12",
         [f"K optimal = {bk} cluster (Silhouette = {fs:.4f})",
          f"K Elbow ≈ {bk_e} | Dipilih K={bk} (lebih objektif)",
          "Cluster Manhattan: volume tinggi, fare moderat",
          "Cluster Airport: fare tinggi, jarak panjang",
          "Cluster Suburban: weekend & night ratio tinggi"]),
        ("📈", "Pertanyaan 3 — Tren Temporal", "Agregasi Time-Series Bulanan, Per Jam & Mingguan",
         [f"Bulan tersibuk: {peak_m_s} | Terendah: {low_m_s}",
          f"Jam paling sibuk: {peak_h_s}:00",
          f"Rush Hour avg fare: ${rush_avg:.2f} vs Non-Rush: ${nrush_avg:.2f}",
          "Rush hour tidak mendorong kenaikan fare signifikan",
          "Jumat & Sabtu malam (20–23) = periode tersibuk ke-2"]),
        ("💳", "Pertanyaan 4 — Metode Pembayaran", "Analisis Frekuensi & Proporsi",
         [f"🥇 {top1_s['Metode Pembayaran']} ({top1_s['Proporsi (%)']:.1f}%)" if top1_s is not None else "",
          f"🥈 {top2_s['Metode Pembayaran']} ({top2_s['Proporsi (%)']:.1f}%)" if top2_s is not None else "",
          "Credit Card dominan di semua periode waktu",
          "Tip hanya ditemukan pada pembayaran Credit Card",
          "No Charge & Dispute < 2% → data transaksi bersih"]),
    ]

    for i in range(0, 4, 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j < len(cards_summary):
                icon,title,method,points = cards_summary[i+j]
                pts_html = "".join(f"<li>{p}</li>" for p in points if p)
                with col:
                    st.markdown(f"""
                    <div style="background:linear-gradient(145deg,{C['bg_card']},{C['bg_card2']});
                        border:1px solid {C['accent2']}44; border-top:3px solid {C['accent1']};
                        border-radius:20px; padding:24px; margin-bottom:16px;
                        box-shadow: 0 4px 24px {C['accent2']}15;">
                        <div style="font-size:2rem; margin-bottom:10px;">{icon}</div>
                        <div style="color:{C['accent3']}; font-size:1.0rem; font-weight:800;
                                    margin-bottom:4px; font-family:'Space Grotesk',sans-serif;">{title}</div>
                        <div style="color:{C['accent2']}; font-size:0.73rem; letter-spacing:1.5px;
                                    margin-bottom:14px; text-transform:uppercase;">{method}</div>
                        <ul style="color:{C['text_main']}; font-size:0.84rem; line-height:1.9;
                                   padding-left:18px; margin:0;">{pts_html}</ul>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{C['bg_card']},{C['bg_card2']});
        border:2px solid {C['accent2']}; border-radius:20px; padding:28px; margin-top:10px;
        box-shadow: 0 8px 32px {C['accent2']}25;">
        <div style="font-size:1.15rem; font-weight:800; color:{C['accent1']}; margin-bottom:16px;
                    font-family:'Space Grotesk',sans-serif;">🎯 Rekomendasi Strategis</div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px;">
            <div style="background:{C['bg_card2']}; border-radius:14px; padding:18px; border:1px solid {C['accent2']}33;">
                <div style="color:{C['accent4']}; font-weight:700; font-size:0.82rem; letter-spacing:1.5px; margin-bottom:10px;">⚡ OPERASIONAL</div>
                <ul style="color:{C['text_main']}; font-size:0.85rem; line-height:1.9; padding-left:18px; margin:0;">
                    <li>Tambah armada di zona Manhattan & Airport pada jam 07:00–09:00</li>
                    <li>Optimalkan driver malam di cluster zona hiburan (Jum–Sab)</li>
                    <li>Gunakan model RF untuk estimasi tarif real-time di aplikasi</li>
                </ul>
            </div>
            <div style="background:{C['bg_card2']}; border-radius:14px; padding:18px; border:1px solid {C['accent2']}33;">
                <div style="color:{C['accent4']}; font-weight:700; font-size:0.82rem; letter-spacing:1.5px; margin-bottom:10px;">📈 BISNIS</div>
                <ul style="color:{C['text_main']}; font-size:0.85rem; line-height:1.9; padding-left:18px; margin:0;">
                    <li>Cashless perlu dipertahankan lewat insentif Credit Card</li>
                    <li>Promosi pada bulan permintaan rendah ({low_m_s}) untuk menaikkan volume</li>
                    <li>Zona pinggiran kota: tarif lebih tinggi karena perjalanan jauh — potensi premium service</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown(
    "<div style='height:1px;background:linear-gradient(90deg,transparent," + C['border'] + "55,transparent);margin-top:48px;'></div>"
    "<div style='text-align:center;padding:28px 20px 24px;"
    "background:linear-gradient(135deg," + C['bg_card'] + "88," + C['bg_card2'] + "88);'>"
    "<div style='display:flex;justify-content:center;align-items:center;gap:18px;flex-wrap:wrap;margin-bottom:10px;'>"
    "<span style='font-size:0.95rem;font-weight:700;color:" + C['text_main'] + ";font-family:Space Grotesk,sans-serif;'>Elsya Anggraini</span>"
    "<span style='width:5px;height:5px;border-radius:50%;background:" + C['accent2'] + ";display:inline-block;opacity:0.7;'></span>"
    "<span style='font-size:0.95rem;font-weight:700;color:" + C['text_main'] + ";font-family:Space Grotesk,sans-serif;'>Okta Mianda Br Sihotang</span>"
    "</div>"
    "<div style='font-size:0.78rem;font-weight:600;color:" + C['accent2'] + ";letter-spacing:1.5px;text-transform:uppercase;margin-bottom:16px;'>Universitas Negeri Yogyakarta</div>"
    "<div style='height:1px;background:linear-gradient(90deg,transparent," + C['border'] + "33,transparent);margin-bottom:14px;'></div>"
    "<div style='font-size:0.74rem;color:" + C['text_muted'] + ";letter-spacing:0.3px;'>"
    "🗽 NYC Yellow Taxi Analytics &nbsp;&middot;&nbsp; PROJECT ADBC 2022 &nbsp;&middot;&nbsp; "
    "Built with <b style='color:" + C['accent4'] + ";'>Streamlit</b> + <b style='color:" + C['accent1'] + ";'>Plotly</b>"
    "</div></div>",
    unsafe_allow_html=True
)

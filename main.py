import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

st.set_page_config(
    page_title="서울 기온 변화 분석 (1980년대 전후)",
    page_icon="🌡️",
    layout="wide"
)

# ──────────────────────────────────────────────────────────────────────────────
# 스타일
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2d3250;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .section-title {
        font-size: 1.3rem; font-weight: 600;
        border-left: 4px solid #e05c5c;
        padding-left: 12px;
        margin: 24px 0 12px 0;
    }
    .highlight-box {
        background: linear-gradient(135deg, #1a1f35, #1e2540);
        border: 1px solid #3a4a7a;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 10px 0;
    }
    .before-color { color: #5bc0eb; }
    .after-color  { color: #e05c5c; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("ta_20260601093156.csv")
    df.columns = df.columns.str.strip()
    df["날짜"] = df["날짜"].str.strip()
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df = df.dropna(subset=["날짜"])
    df["연도"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month
    return df

df = load_data()

# 2026 제외 (데이터 불완전)
df = df[df["연도"] <= 2025]

# ──────────────────────────────────────────────────────────────────────────────
# 사이드바 – 기준 연도 선택
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ 분석 설정")
st.sidebar.markdown("---")

cutoff_year = st.sidebar.slider(
    "기준 연도 (전·후 구분)",
    min_value=1960,
    max_value=2000,
    value=1980,
    step=1,
    help="이 연도를 기준으로 '이전'과 '이후'를 구분합니다."
)

season_map = {"전체": None, "봄 (3-5월)": [3,4,5], "여름 (6-8월)": [6,7,8],
              "가을 (9-11월)": [9,10,11], "겨울 (12-2월)": [12,1,2]}
selected_season = st.sidebar.selectbox("계절 선택", list(season_map.keys()))

show_rolling = st.sidebar.checkbox("10년 이동평균 표시", value=True)
show_ci = st.sidebar.checkbox("추세선 신뢰구간 표시", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("📊 **데이터 정보**")
st.sidebar.markdown(f"- 기간: {df['연도'].min()} ~ {df['연도'].max()}")
st.sidebar.markdown(f"- 총 일수: {len(df):,}일")
st.sidebar.markdown("- 지점: 서울 (108)")

# ──────────────────────────────────────────────────────────────────────────────
# 필터링
# ──────────────────────────────────────────────────────────────────────────────
months = season_map[selected_season]
if months:
    df_filtered = df[df["월"].isin(months)]
else:
    df_filtered = df.copy()

yearly = df_filtered.groupby("연도")["평균기온(℃)"].mean().reset_index()
yearly.columns = ["연도", "평균기온"]

before = yearly[yearly["연도"] < cutoff_year]
after  = yearly[yearly["연도"] >= cutoff_year]

# ──────────────────────────────────────────────────────────────────────────────
# 헤더
# ──────────────────────────────────────────────────────────────────────────────
st.title("🌡️ 서울 기온 변화 분석")
st.markdown(f"**{cutoff_year}년을 기준으로 기온 상승 패턴이 달라졌는가?**")
st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# 핵심 지표 카드
# ──────────────────────────────────────────────────────────────────────────────
avg_before = before["평균기온"].mean()
avg_after  = after["평균기온"].mean()
diff       = avg_after - avg_before

# 선형 회귀로 상승 속도 계산
slope_b, *_ = stats.linregress(before["연도"], before["평균기온"])
slope_a, *_ = stats.linregress(after["연도"],  after["평균기온"])

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    f"{cutoff_year}년 이전 평균기온",
    f"{avg_before:.2f}°C",
    help=f"{df['연도'].min()}~{cutoff_year-1}년 연평균"
)
col2.metric(
    f"{cutoff_year}년 이후 평균기온",
    f"{avg_after:.2f}°C",
    delta=f"+{diff:.2f}°C",
    help=f"{cutoff_year}~{df['연도'].max()}년 연평균"
)
col3.metric(
    "이전 기온 상승 속도",
    f"{slope_b*10:.3f}°C/10년",
    help="선형 회귀 기울기 × 10"
)
col4.metric(
    "이후 기온 상승 속도",
    f"{slope_a*10:.3f}°C/10년",
    delta=f"+{(slope_a - slope_b)*10:.3f}°C/10년",
    help="선형 회귀 기울기 × 10"
)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# 차트 1 – 연도별 기온 + 추세선
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📈 연도별 평균기온 추이</div>', unsafe_allow_html=True)

fig1 = go.Figure()

# 배경 영역
fig1.add_vrect(
    x0=df["연도"].min(), x1=cutoff_year,
    fillcolor="rgba(91,192,235,0.06)", line_width=0,
    annotation_text=f"{cutoff_year}년 이전", annotation_position="top left",
    annotation_font_color="#5bc0eb"
)
fig1.add_vrect(
    x0=cutoff_year, x1=df["연도"].max(),
    fillcolor="rgba(224,92,92,0.06)", line_width=0,
    annotation_text=f"{cutoff_year}년 이후", annotation_position="top right",
    annotation_font_color="#e05c5c"
)

# 기준선
fig1.add_vline(x=cutoff_year, line_dash="dash", line_color="#ffffff60", line_width=1.5)

# 산점도
fig1.add_trace(go.Scatter(
    x=before["연도"], y=before["평균기온"],
    mode="markers", name=f"{cutoff_year}년 이전",
    marker=dict(color="#5bc0eb", size=5, opacity=0.7),
))
fig1.add_trace(go.Scatter(
    x=after["연도"], y=after["평균기온"],
    mode="markers", name=f"{cutoff_year}년 이후",
    marker=dict(color="#e05c5c", size=5, opacity=0.7),
))

# 추세선
def add_trendline(fig, data, color, name, show_ci=True):
    x = data["연도"].values
    y = data["평균기온"].values
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    if show_ci:
        n = len(x)
        t_val = stats.t.ppf(0.975, df=n-2)
        x_mean = x.mean()
        ss_x = np.sum((x - x_mean)**2)
        se_line = se * np.sqrt(1/n + (x_line - x_mean)**2 / ss_x)
        y_upper = y_line + t_val * se_line
        y_lower = y_line - t_val * se_line
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_line, x_line[::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill="toself", fillcolor=color.replace("1)", "0.12)"),
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))

    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines", name=f"{name} 추세",
        line=dict(color=color, width=2.5),
        hovertemplate=f"<b>{name} 추세</b><br>연도: %{{x:.0f}}<br>기온: %{{y:.2f}}°C<extra></extra>"
    ))
    return slope, p

slope_b2, p_b = add_trendline(fig1, before, "rgba(91,192,235,1)", f"{cutoff_year}년 이전", show_ci)
slope_a2, p_a = add_trendline(fig1, after,  "rgba(224,92,92,1)",  f"{cutoff_year}년 이후",  show_ci)

# 이동평균
if show_rolling:
    yearly_all = yearly.set_index("연도")["평균기온"]
    roll = yearly_all.rolling(10, center=True).mean().reset_index()
    fig1.add_trace(go.Scatter(
        x=roll["연도"], y=roll["평균기온"],
        mode="lines", name="10년 이동평균",
        line=dict(color="#ffd700", width=2, dash="dot"),
    ))

fig1.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
    height=460,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    margin=dict(l=40, r=20, t=40, b=40),
)
st.plotly_chart(fig1, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 차트 2 & 3 – 분포 + 10년 단위 박스플롯
# ──────────────────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="section-title">📊 기온 분포 비교</div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=before["평균기온"], name=f"{cutoff_year}년 이전",
        marker_color="rgba(91,192,235,0.75)", nbinsx=20,
        histnorm="probability density"
    ))
    fig2.add_trace(go.Histogram(
        x=after["평균기온"], name=f"{cutoff_year}년 이후",
        marker_color="rgba(224,92,92,0.75)", nbinsx=20,
        histnorm="probability density"
    ))
    # KDE 곡선
    for data, color in [(before["평균기온"], "#5bc0eb"), (after["평균기온"], "#e05c5c")]:
        kde_x = np.linspace(data.min()-0.5, data.max()+0.5, 200)
        kde = stats.gaussian_kde(data)
        fig2.add_trace(go.Scatter(
            x=kde_x, y=kde(kde_x), mode="lines",
            line=dict(color=color, width=2.5), showlegend=False
        ))
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        height=360, barmode="overlay",
        xaxis_title="연평균기온 (°C)", yaxis_title="밀도",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.markdown('<div class="section-title">📦 10년 단위 기온 분포</div>', unsafe_allow_html=True)
    yearly["decade"] = (yearly["연도"] // 10 * 10).astype(str) + "s"
    decade_order = sorted(yearly["decade"].unique())
    colors_box = []
    for d in decade_order:
        yr = int(d.replace("s",""))
        colors_box.append("#5bc0eb" if yr < cutoff_year else "#e05c5c")

    fig3 = go.Figure()
    for dec, col in zip(decade_order, colors_box):
        sub = yearly[yearly["decade"] == dec]
        fig3.add_trace(go.Box(
            x=sub["decade"], y=sub["평균기온"],
            name=dec, marker_color=col,
            line_color=col, boxmean=True,
        ))
    fig3.update_layout(
        template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        height=360, showlegend=False,
        xaxis_title="연대", yaxis_title="연평균기온 (°C)",
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 차트 4 – 월별 기온 변화 히트맵
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🗓️ 월별 평균기온 히트맵 (연도별)</div>', unsafe_allow_html=True)

# 10년 단위로 집계
df_heatmap = df.copy()
df_heatmap["decade"] = (df_heatmap["연도"] // 10 * 10).astype(str) + "s"
heat = df_heatmap.groupby(["decade", "월"])["평균기온(℃)"].mean().unstack()
heat = heat.loc[sorted(heat.index)]
month_labels = ["1월","2월","3월","4월","5월","6월","7월","8월","9월","10월","11월","12월"]

fig4 = go.Figure(go.Heatmap(
    z=heat.values,
    x=month_labels,
    y=heat.index.tolist(),
    colorscale="RdYlBu_r",
    colorbar=dict(title="기온(°C)"),
    hovertemplate="연대: %{y}<br>월: %{x}<br>평균기온: %{z:.1f}°C<extra></extra>"
))
# 기준선 표시
cutoff_decade = str(cutoff_year // 10 * 10) + "s"
if cutoff_decade in heat.index.tolist():
    idx = heat.index.tolist().index(cutoff_decade)
    fig4.add_hline(y=idx-0.5, line_color="white", line_dash="dash", line_width=1.5,
                   annotation_text=f"← {cutoff_year}년 기준", annotation_font_color="white")

fig4.update_layout(
    template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
    height=380, margin=dict(l=50, r=20, t=30, b=40),
    xaxis_title="월", yaxis_title="연대",
)
st.plotly_chart(fig4, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 차트 5 – 기온 편차 (아노말리)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📉 기온 편차 (전체 평균 대비)</div>', unsafe_allow_html=True)

baseline = yearly["평균기온"].mean()
yearly["편차"] = yearly["평균기온"] - baseline

fig5 = go.Figure()
fig5.add_hline(y=0, line_color="#ffffff40", line_width=1)
fig5.add_vline(x=cutoff_year, line_dash="dash", line_color="#ffffff60", line_width=1.5)

for _, row in yearly.iterrows():
    color = "#e05c5c" if row["편차"] > 0 else "#5bc0eb"
    fig5.add_trace(go.Bar(
        x=[row["연도"]], y=[row["편차"]],
        marker_color=color, showlegend=False,
        hovertemplate=f"연도: {int(row['연도'])}<br>편차: {row['편차']:+.2f}°C<extra></extra>"
    ))

# 5년 이동평균
yearly_sorted = yearly.sort_values("연도")
roll5 = yearly_sorted["편차"].rolling(5, center=True).mean()
fig5.add_trace(go.Scatter(
    x=yearly_sorted["연도"], y=roll5,
    mode="lines", name="5년 이동평균",
    line=dict(color="#ffd700", width=2.5),
))

fig5.update_layout(
    template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
    height=380, bargap=0.05,
    xaxis_title="연도", yaxis_title=f"편차 (°C, 기준: {baseline:.2f}°C)",
    showlegend=True,
    legend=dict(orientation="h", y=1.05),
    margin=dict(l=40, r=20, t=30, b=40),
)
st.plotly_chart(fig5, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 통계 검증
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔬 통계 검증</div>', unsafe_allow_html=True)

t_stat, p_val = stats.ttest_ind(before["평균기온"], after["평균기온"])
mw_stat, mw_p = stats.mannwhitneyu(before["평균기온"], after["평균기온"], alternative="less")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.markdown("**📐 t-검정 (독립표본)**")
    st.markdown(f"- t 통계량: `{t_stat:.4f}`")
    st.markdown(f"- p-value: `{p_val:.6f}`")
    significance = "✅ **통계적으로 유의미** (p < 0.05)" if p_val < 0.05 else "❌ 유의미하지 않음"
    st.markdown(f"- 결론: {significance}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.markdown("**📐 Mann-Whitney U 검정 (비모수)**")
    st.markdown(f"- U 통계량: `{mw_stat:.4f}`")
    st.markdown(f"- p-value: `{mw_p:.6f}`")
    significance2 = "✅ **통계적으로 유의미** (p < 0.05)" if mw_p < 0.05 else "❌ 유의미하지 않음"
    st.markdown(f"- 결론: {significance2}")
    st.markdown('</div>', unsafe_allow_html=True)

# 추세 비교 요약
st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
st.markdown("**📋 추세 분석 요약**")
col_s1, col_s2, col_s3 = st.columns(3)
col_s1.markdown(f"""
<div style='text-align:center'>
  <div class='before-color' style='font-size:1.5rem;font-weight:700;'>{slope_b*10:+.3f}°C</div>
  <div style='color:#aaa;font-size:0.85rem;'>{cutoff_year}년 이전<br>/10년 상승</div>
</div>""", unsafe_allow_html=True)
col_s2.markdown(f"""
<div style='text-align:center'>
  <div class='after-color' style='font-size:1.5rem;font-weight:700;'>{slope_a*10:+.3f}°C</div>
  <div style='color:#aaa;font-size:0.85rem;'>{cutoff_year}년 이후<br>/10년 상승</div>
</div>""", unsafe_allow_html=True)
col_s3.markdown(f"""
<div style='text-align:center'>
  <div style='color:#ffd700;font-size:1.5rem;font-weight:700;'>{(slope_a-slope_b)*10:+.3f}°C</div>
  <div style='color:#aaa;font-size:0.85rem;'>상승 가속도<br>(이후 - 이전)</div>
</div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 원시 데이터
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("📄 연도별 평균기온 데이터 보기"):
    display = yearly[["연도", "평균기온", "편차"]].copy()
    display["구분"] = display["연도"].apply(lambda y: f"{cutoff_year}년 이전" if y < cutoff_year else f"{cutoff_year}년 이후")
    display["평균기온"] = display["평균기온"].round(2)
    display["편차"] = display["편차"].round(2)
    st.dataframe(display, use_container_width=True, height=300)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.8rem;'>데이터: 기상청 서울 지점(108) | 분석: Claude</div>",
    unsafe_allow_html=True
)

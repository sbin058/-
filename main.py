# -*- coding: utf-8 -*-
"""
🚦 안전한 등하굣길 분석실
실행 명령: streamlit run main.py

데이터 폴더 구조 예시
data/
 ├─ 19_25_pedstrians.csv
 ├─ 12_25_schoolzone.csv
 ├─ 한국도로교통공단_사고유형별 교통사고 통계_20251231.csv
 ├─ 한국도로교통공단_사고유형별 월별 교통사고 통계_20251231.csv
 └─ 한국도로교통공단_최근 5년 어린이보호구역내 어린이(12세 이하) 교통사고 통계_20251231.csv

CSV 인코딩은 UTF-8-SIG, CP949, EUC-KR 순서로 자동 시도합니다.
"""

from pathlib import Path
import re
import warnings

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# 1. 페이지 설정 및 디자인
# -------------------------------------------------------------------
st.set_page_config(
    page_title="안전한 등하굣길 분석실",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#17365D"
BLUE = "#4A90E2"
SKY = "#8CCFF0"
RED = "#D94848"
GREEN = "#2E9D68"
GRAY = "#64748B"
COLORS = [NAVY, BLUE, SKY, "#86A9D4", "#B8D9E8", "#E8A0A0"]

st.markdown("""
<style>
.stApp {background-color:#F5F8FC;}
[data-testid="stSidebar"] {background-color:#EAF1F8;}
h1,h2,h3 {color:#17365D;}
div[data-testid="stMetric"] {
    background:#FFFFFF; border:1px solid #DCE6F0;
    padding:14px; border-radius:12px;
}
.note {color:#64748B;font-size:0.88rem;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. CSV 파일 읽기 및 열 이름 탐색
# -------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"

FILE_HINTS = {
    "pedestrian": ["pedstrian", "pedestrian", "보행자"],
    "schoolzone": ["schoolzone", "어린이보호구역"],
    "monthly": ["월별", "monthly"],
    "type": ["사고유형별 교통사고", "사고유형별_교통사고"],
    "child_year": ["최근 5년", "12세 이하", "어린이(12세"],
    "facilities": ["시설", "facility"],
    "volume": ["교통량", "volume"],
}

ALIASES = {
    "location": ["지점명", "지역", "시군구명", "시도시군구명", "행정구역", "주소", "도로명", "location", "address"],
    "accidents": ["사고건수", "교통사고건수", "발생건수", "사고 수", "accidents"],
    "deaths": ["사망자수", "사망자", "사망", "deaths"],
    "injuries": ["부상자수", "부상자", "사상자수", "injuries"],
    "serious": ["중상자수", "중상자", "serious_injuries"],
    "minor": ["경상자수", "경상자", "minor_injuries"],
    "reported": ["부상신고자수", "부상신고자", "reported_injuries"],
    "latitude": ["위도", "latitude", "lat"],
    "longitude": ["경도", "longitude", "lon", "lng"],
    "type": ["사고유형", "사고유형중분류", "사고유형대분류", "유형", "type"],
    "month": ["발생월", "월", "month"],
    "year": ["발생년", "발생연도", "연도", "년도", "year"],
    "hour": ["발생시간", "사고시간", "시간대", "시간", "시", "hour"],
    "label": ["위험도", "위험등급", "위험지역여부", "risk_level", "label"],
}

def normalize_name(value):
    return re.sub(r"[\s_\-()（）]", "", str(value)).lower()

def find_col(df, key):
    """실제 데이터에 존재하는 열 중 별칭과 일치하는 첫 번째 열을 찾습니다."""
    if df is None or df.empty:
        return None
    aliases = {normalize_name(x) for x in ALIASES.get(key, [])}
    for col in df.columns:
        if normalize_name(col) in aliases:
            return col
    # 열 이름에 별칭이 포함된 경우도 허용
    for col in df.columns:
        name = normalize_name(col)
        for alias in aliases:
            if alias and (alias in name or name in alias):
                return col
    return None

def read_csv_safely(path):
    """공공데이터 CSV에서 자주 쓰이는 인코딩을 순서대로 시도합니다."""
    errors = []
    for encoding in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError("CSV 인코딩을 확인할 수 없습니다: " + " / ".join(errors))

@st.cache_data(show_spinner=False)
def load_all_csvs():
    result = {}
    if not DATA_DIR.exists():
        return result
    files = list(DATA_DIR.glob("*.csv"))
    for kind, hints in FILE_HINTS.items():
        matches = [
            p for p in files
            if any(normalize_name(hint) in normalize_name(p.stem) for hint in hints)
        ]
        if matches:
            try:
                result[kind] = read_csv_safely(matches[0])
            except Exception as exc:
                result[kind] = None
                result[kind + "_error"] = str(exc)
    # 파일명이 예상과 달라도 열 구조를 보고 유형별 파일을 탐색
    for path in files:
        if any(v is not None and isinstance(v, pd.DataFrame) and len(v.columns) for k, v in result.items() if not k.endswith("_error")):
            try:
                df = read_csv_safely(path)
            except Exception:
                continue
            cols = {normalize_name(c) for c in df.columns}
            if "사고유형" in cols and "발생월" in cols and "monthly" not in result:
                result["monthly"] = df
            elif "사고유형" in cols and "사고건수" in cols and "type" not in result:
                result["type"] = df
            elif "위도" in cols and "경도" in cols and "사고건수" in cols:
                if "pedestrian" in normalize_name(path.stem) and "pedestrian" not in result:
                    result["pedestrian"] = df
    return result

raw = load_all_csvs()

def clean_frame(df):
    """중복 행 제거 및 완전히 비어 있는 행 제거. 원본 파일은 수정하지 않습니다."""
    if df is None:
        return None, 0, 0
    before = len(df)
    cleaned = df.dropna(how="all").drop_duplicates().copy()
    removed = before - len(cleaned)
    missing = int(cleaned.isna().sum().sum())
    return cleaned, removed, missing

data = {}
quality = {}
for name, frame in raw.items():
    if name.endswith("_error"):
        continue
    cleaned, removed, missing = clean_frame(frame)
    data[name] = cleaned
    quality[name] = {"중복·빈 행 제거": removed, "결측 셀": missing}

def numeric(series):
    return pd.to_numeric(series, errors="coerce")

def get_num(df, key):
    col = find_col(df, key)
    if col is None:
        return None
    return numeric(df[col])

def get_text(df, key):
    col = find_col(df, key)
    if col is None:
        return None
    return df[col].astype(str)

def add_injury_total(df):
    """부상자 총계 열이 없으면 실제 세부 부상자 열의 합으로 계산합니다."""
    direct = get_num(df, "injuries")
    if direct is not None:
        return direct
    pieces = [get_num(df, k) for k in ["serious", "minor", "reported"]]
    pieces = [x for x in pieces if x is not None]
    if pieces:
        return sum(p.fillna(0) for p in pieces)
    return None

def show_missing(message):
    st.info("ℹ️ " + message)

def plot_empty(message):
    st.info("이 그래프를 만들 수 있는 데이터가 없습니다. " + message)

def chart_layout(fig, height=390):
    fig.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=20, r=20, t=65, b=35),
        title_font=dict(color=NAVY, size=18),
        font=dict(color="#334155"),
        legend_title_text="구분",
    )
    return fig

# -------------------------------------------------------------------
# 3. 데이터셋별 실제 열 확인 및 기본 데이터 준비
# -------------------------------------------------------------------
ped = data.get("pedestrian")
school = data.get("schoolzone")
monthly = data.get("monthly")
type_df = data.get("type")
child_year = data.get("child_year")

def prepare_location(df, source_name):
    """위치별 사고 다발 자료를 공통 분석 형식으로 변환합니다."""
    if df is None:
        return None
    loc = find_col(df, "location")
    acc = get_num(df, "accidents")
    lat = get_num(df, "latitude")
    lon = get_num(df, "longitude")
    if loc is None or acc is None:
        return None
    out = pd.DataFrame({
        "지역": df[loc].astype(str),
        "사고건수": acc,
        "위도": lat if lat is not None else pd.Series([float("nan")] * len(df), index=df.index),
        "경도": lon if lon is not None else pd.Series([float("nan")] * len(df), index=df.index),
        "사망자수": get_num(df, "deaths") if get_num(df, "deaths") is not None else 0,
        "부상자수": add_injury_total(df) if add_injury_total(df) is not None else 0,
        "자료종류": source_name,
    }, index=df.index)
    out["사고건수"] = out["사고건수"].fillna(0)
    out["사망자수"] = pd.Series(out["사망자수"], index=df.index).fillna(0)
    out["부상자수"] = pd.Series(out["부상자수"], index=df.index).fillna(0)
    out = out[(out["지역"].str.lower() != "nan") & (out["사고건수"] >= 0)]
    return out.reset_index(drop=True)

location_parts = []
for frame, label in [(ped, "보행자 사고 다발지역"), (school, "어린이보호구역 사고 다발지역")]:
    prepared = prepare_location(frame, label)
    if prepared is not None and not prepared.empty:
        location_parts.append(prepared)

locations = pd.concat(location_parts, ignore_index=True) if location_parts else pd.DataFrame()

# 유형별 통계
type_col = find_col(type_df, "type")
type_acc = get_num(type_df, "accidents")
type_deaths = get_num(type_df, "deaths")
type_injuries = add_injury_total(type_df)

# 월별 통계
month_col = find_col(monthly, "month")
month_acc = get_num(monthly, "accidents")
month_type_col = find_col(monthly, "type")

# 어린이보호구역 연도 통계
child_year_col = find_col(child_year, "year")
child_acc = get_num(child_year, "accidents")
child_deaths = get_num(child_year, "deaths")
child_injuries = add_injury_total(child_year)

# 대시보드 기본 지표: 사고유형 통계가 있으면 해당 표의 사고건수를 사용합니다.
total_accidents = int(type_acc.fillna(0).sum()) if type_acc is not None else None
total_deaths = int(type_deaths.fillna(0).sum()) if type_deaths is not None else None
total_injuries = int(type_injuries.fillna(0).sum()) if type_injuries is not None else None

# 보행자 사고 다발지역 자료의 사고건수 합은 '다발지역 자료 내 합계'로만 표시합니다.
pedestrian_accidents = None
if ped is not None:
    p_acc = get_num(ped, "accidents")
    if p_acc is not None:
        pedestrian_accidents = int(p_acc.fillna(0).sum())

# -------------------------------------------------------------------
# 4. 사이드바 메뉴
# -------------------------------------------------------------------
st.sidebar.title("🚦 안전한 등하굣길 분석실")
st.sidebar.caption("교통사고 데이터 분석 대시보드")
menu = st.sidebar.radio(
    "분석 메뉴",
    [
        "🏠 전체 분석 대시보드",
        "📊 사고 유형 및 통계 분석",
        "🕒 시간대별 사고 분석",
        "🗺️ 지역별 위험도 지도",
        "🤖 머신러닝 위험도 분류",
        "🚸 교통안전시설 분석",
        "💡 안전 개선 방안",
    ],
)
st.sidebar.divider()
st.sidebar.caption("데이터는 프로젝트의 data 폴더에서 자동으로 불러옵니다.")
if st.sidebar.button("🔄 데이터 다시 불러오기"):
    st.cache_data.clear()
    st.rerun()

st.title("🚦 안전한 등하굣길 분석실")
st.caption("공공 교통사고 데이터를 활용한 사고 특성 분석 및 통학 안전 참고 정보")

if not DATA_DIR.exists():
    st.error("`data` 폴더를 찾을 수 없습니다. main.py와 같은 위치에 data 폴더를 만들고 CSV 파일을 넣어 주세요.")
elif not data:
    st.error("data 폴더에서 CSV 파일을 찾지 못했습니다. CSV 파일명과 폴더 위치를 확인해 주세요.")
else:
    with st.expander("📁 불러온 데이터 및 전처리 현황", expanded=False):
        for key, frame in data.items():
            if frame is not None:
                st.write(f"**{key}** — {len(frame):,}행 × {len(frame.columns)}열")
                st.caption("열 이름: " + ", ".join(map(str, frame.columns)))
                q = quality.get(key, {})
                st.caption(f"중복·빈 행 제거: {q.get('중복·빈 행 제거', 0):,}개 / 결측 셀: {q.get('결측 셀', 0):,}개")
            else:
                st.warning(f"{key}: 파일을 읽지 못했습니다.")

# -------------------------------------------------------------------
# 5. 공통 분석 함수: 상대적 위험도 및 모델
# -------------------------------------------------------------------
def build_risk_table(loc_df):
    if loc_df is None or loc_df.empty:
        return None
    df = loc_df.copy()
    df["위험점수"] = (
        df["사고건수"].rank(pct=True) * 0.65
        + df["사망자수"].rank(pct=True) * 0.25
        + df["부상자수"].rank(pct=True) * 0.10
    )
    # KMeans는 최소 3개 위치와 유효한 변동이 있을 때만 실행합니다.
    features = df[["사고건수", "사망자수", "부상자수"]].fillna(0)
    if len(df) >= 3 and features.nunique().sum() > 1:
        scaled = StandardScaler().fit_transform(features)
        model = KMeans(n_clusters=min(3, len(df)), random_state=42, n_init=10)
        df["군집"] = model.fit_predict(scaled)
        means = df.groupby("군집")["위험점수"].mean().sort_values()
        levels = {cluster: ["낮음", "보통", "높음"][i] for i, cluster in enumerate(means.index)}
        df["위험도"] = df["군집"].map(levels)
        df["모델방식"] = "KMeans 상대적 군집 분석"
    else:
        # 데이터가 적으면 점수 순위의 삼분위 구간을 이용하되 모델 학습이라고 표현하지 않습니다.
        ranks = df["위험점수"].rank(method="first", pct=True)
        df["위험도"] = ranks.apply(lambda x: "높음" if x > 0.66 else ("보통" if x > 0.33 else "낮음"))
        df["모델방식"] = "데이터 순위 기반 참고 분류"
    return df

risk_table = build_risk_table(locations)

def supervised_result(loc_df):
    """실제 위험도 정답 레이블이 있을 때만 RandomForest 지도학습을 수행합니다."""
    if loc_df is None or loc_df.empty:
        return None
    label_col = find_col(loc_df, "label")
    if label_col is None:
        return None
    work = loc_df.copy()
    features = pd.DataFrame({
        "사고건수": get_num(work, "accidents"),
        "사망자수": get_num(work, "deaths"),
    })
    injuries = add_injury_total(work)
    if injuries is not None:
        features["부상자수"] = injuries
    features = features.fillna(0)
    labels = work[label_col].astype(str)
    valid = labels.notna() & (labels.str.lower() != "nan")
    features, labels = features.loc[valid], labels.loc[valid]
    if len(features) < 10 or labels.nunique() < 2:
        return {"error": "지도학습에 필요한 데이터가 부족합니다. (유효 표본 10개 이상, 레이블 2종 이상 필요)"}
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.25, random_state=42, stratify=labels
        )
        clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        return {
            "accuracy": accuracy_score(y_test, pred),
            "matrix": confusion_matrix(y_test, pred, labels=clf.classes_),
            "classes": list(clf.classes_),
            "features": list(features.columns),
            "model": clf,
            "x_test": x_test,
            "y_test": y_test,
            "pred": pred,
        }
    except Exception as exc:
        return {"error": f"모델 학습을 진행할 수 없습니다: {exc}"}

# -------------------------------------------------------------------
# 6. 메뉴 1: 전체 분석 대시보드
# -------------------------------------------------------------------
if menu == "🏠 전체 분석 대시보드":
    st.header("전체 교통사고 분석 요약")
    metrics = st.columns(4)
    metric_values = [
        ("전체 교통사고", total_accidents, "사고유형별 통계 기준"),
        ("보행자 사고 다발지 합계", pedestrian_accidents, "보행자 다발지역 자료 내 합계"),
        ("전체 사망자", total_deaths, "사고유형별 통계 기준"),
        ("전체 부상자", total_injuries, "세부 부상자 열 합산 가능 시"),
    ]
    for col, (label, value, note) in zip(metrics, metric_values):
        if value is None:
            col.metric(label, "자료 없음")
        else:
            col.metric(label, f"{value:,}명" if label in ["전체 사망자", "전체 부상자"] else f"{value:,}건")
        col.caption(note)

    st.subheader("데이터 분석 요약")
    summaries = []
    if type_df is not None and type_col and type_acc is not None:
        tmp = pd.DataFrame({"유형": type_df[type_col].astype(str), "건수": type_acc}).dropna()
        if not tmp.empty:
            top = tmp.loc[tmp["건수"].idxmax()]
            summaries.append(f"사고유형 통계에서 가장 많은 유형은 **{top['유형']}**이며, {int(top['건수']):,}건입니다.")
    if risk_table is not None:
        top_loc = risk_table.sort_values(["위험점수", "사고건수"], ascending=False).iloc[0]
        summaries.append(f"위치 자료에서 상대적 위험도가 가장 높은 지점은 **{top_loc['지역']}**입니다. 사고 {int(top_loc['사고건수']):,}건을 기록했습니다.")
    if monthly is not None and month_col and month_acc is not None:
        m = pd.DataFrame({"월": numeric(monthly[month_col]), "건수": month_acc}).dropna()
        if not m.empty:
            by_month = m.groupby("월")["건수"].sum()
            summaries.append(f"월별 통계에서 사고가 가장 많은 달은 **{int(by_month.idxmax())}월**입니다.")
    if summaries:
        for item in summaries:
            st.markdown("- " + item)
    else:
        show_missing("요약할 수 있는 사고 통계가 충분하지 않습니다.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("사고 유형별 발생 현황")
        if type_df is not None and type_col and type_acc is not None:
            chart = pd.DataFrame({"사고 유형": type_df[type_col].astype(str), "사고 건수": type_acc}).dropna()
            chart = chart.groupby("사고 유형", as_index=False)["사고 건수"].sum().sort_values("사고 건수", ascending=False).head(10)
            st.plotly_chart(chart_layout(px.bar(chart, x="사고 유형", y="사고 건수", color_discrete_sequence=[BLUE])), use_container_width=True)
        else:
            plot_empty("사고유형별 통계 파일 또는 사고유형·사고건수 열이 필요합니다.")
    with c2:
        st.subheader("월별 사고 발생 현황")
        if monthly is not None and month_col and month_acc is not None:
            chart = pd.DataFrame({"월": numeric(monthly[month_col]), "사고 건수": month_acc}).dropna()
            chart = chart.groupby("월", as_index=False)["사고 건수"].sum().sort_values("월")
            st.plotly_chart(chart_layout(px.bar(chart, x="월", y="사고 건수", color_discrete_sequence=[SKY])), use_container_width=True)
        else:
            plot_empty("월별 통계 데이터가 필요합니다.")

    st.subheader("상대적 위험 지역 TOP 5")
    if risk_table is not None:
        top5 = risk_table.sort_values(["위험점수", "사고건수"], ascending=False).head(5)
        st.dataframe(top5[["지역", "자료종류", "사고건수", "사망자수", "부상자수", "위험도"]], use_container_width=True, hide_index=True)
    else:
        show_missing("위도·경도 또는 지역명과 사고건수가 포함된 위치 자료가 없습니다.")

    st.subheader("분석 목적 및 데이터 출처")
    st.write("이 앱은 제공된 공공 교통사고 통계를 분석하여 사고 유형과 지역별 사고 다발 특성을 파악하고, 안전한 통학 환경 조성을 위한 참고 정보를 제공합니다.")
    st.caption("데이터 출처: 사용자가 제공한 한국도로교통공단 교통사고 통계 CSV 파일. 데이터의 기준연도와 집계 범위는 각 원본 파일의 설명을 확인해야 합니다.")

# -------------------------------------------------------------------
# 7. 메뉴 2: 사고 유형 및 통계 분석
# -------------------------------------------------------------------
elif menu == "📊 사고 유형 및 통계 분석":
    st.header("사고 유형 및 통계 분석")
    if type_df is None or type_col is None or type_acc is None:
        show_missing("사고유형별 통계 파일에서 사고유형과 사고건수 열을 찾지 못했습니다.")
    else:
        chart = pd.DataFrame({
            "사고 유형": type_df[type_col].astype(str),
            "사고 건수": type_acc,
        }).dropna()
        chart = chart.groupby("사고 유형", as_index=False)["사고 건수"].sum().sort_values("사고 건수", ascending=False)
        if not chart.empty:
            first = chart.iloc[0]
            st.success(f"가장 많이 발생한 사고 유형: {first['사고 유형']} ({int(first['사고 건수']):,}건)")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(chart_layout(px.bar(chart, x="사고 유형", y="사고 건수", color="사고 건수", color_continuous_scale=["#B8D9E8", NAVY])), use_container_width=True)
            with c2:
                st.plotly_chart(chart_layout(px.pie(chart, names="사고 유형", values="사고 건수", hole=0.35)), use_container_width=True)
        else:
            plot_empty("사고유형별 집계 자료가 비어 있습니다.")

    st.subheader("사고 심각도")
    if type_df is not None and type_col and type_acc is not None:
        available = []
        for key, label in [("deaths", "사망자수"), ("serious", "중상자수"), ("minor", "경상자수"), ("reported", "부상신고자수")]:
            values = get_num(type_df, key)
            if values is not None:
                available.append(pd.DataFrame({"사고 유형": type_df[type_col].astype(str), "구분": label, "인원": values}))
        if available:
            severity = pd.concat(available).dropna()
            st.plotly_chart(chart_layout(px.bar(severity, x="사고 유형", y="인원", color="구분", barmode="group")), use_container_width=True)
        else:
            plot_empty("사망자·부상자 관련 열이 없습니다.")
    else:
        plot_empty("사고유형별 통계가 없습니다.")

    if type_df is not None and type_acc is not None and type_injuries is not None:
        scatter = pd.DataFrame({"사고 건수": type_acc, "부상자 수": type_injuries}).dropna()
        st.subheader("사고 건수와 부상자 수의 관계")
        st.plotly_chart(chart_layout(px.scatter(scatter, x="사고 건수", y="부상자 수", trendline=None)), use_container_width=True)

# -------------------------------------------------------------------
# 8. 메뉴 3: 시간대별 사고 분석
# -------------------------------------------------------------------
elif menu == "🕒 시간대별 사고 분석":
    st.header("시간대별 사고 분석")
    hour_col = find_col(type_df, "hour") or find_col(monthly, "hour")
    if hour_col is None:
        show_missing("현재 제공된 CSV에는 사고 발생 시각 또는 시간대 열이 없습니다. 따라서 시간대별 사고 건수, 등교 시간(07~09시), 하교 시간(15~18시) 비교를 계산할 수 없습니다.")
    else:
        source = type_df if find_col(type_df, "hour") else monthly
        hcol = find_col(source, "hour")
        hacc = get_num(source, "accidents")
        if hacc is None:
            show_missing("시간대 열은 있지만 사고건수 열을 찾을 수 없습니다.")
        else:
            hours = source[hcol].astype(str).str.extract(r"(\d{1,2})")[0]
            frame = pd.DataFrame({"시간": pd.to_numeric(hours, errors="coerce"), "사고 건수": hacc}).dropna()
            frame = frame[(frame["시간"] >= 0) & (frame["시간"] <= 23)]
            by_hour = frame.groupby("시간", as_index=False)["사고 건수"].sum().sort_values("시간")
            if not by_hour.empty:
                st.plotly_chart(chart_layout(px.line(by_hour, x="시간", y="사고 건수", markers=True)), use_container_width=True)
                by_hour["구분"] = by_hour["시간"].apply(lambda x: "등교(07~09시)" if 7 <= x <= 9 else ("하교(15~18시)" if 15 <= x <= 18 else "그 외"))
                compare = by_hour[by_hour["구분"] != "그 외"].groupby("구분", as_index=False)["사고 건수"].sum()
                st.plotly_chart(chart_layout(px.bar(compare, x="구분", y="사고 건수", color="구분", color_discrete_sequence=[BLUE, RED])), use_container_width=True)
            else:
                plot_empty("시간 데이터 형식을 해석할 수 없습니다.")

    st.subheader("월별 사고 현황")
    if monthly is not None and month_col and month_acc is not None:
        frame = pd.DataFrame({"월": numeric(monthly[month_col]), "사고 건수": month_acc}).dropna()
        frame = frame.groupby("월", as_index=False)["사고 건수"].sum().sort_values("월")
        st.plotly_chart(chart_layout(px.bar(frame, x="월", y="사고 건수", color_discrete_sequence=[SKY])), use_container_width=True)
    else:
        plot_empty("월별 사고 통계가 없습니다.")

# -------------------------------------------------------------------
# 9. 메뉴 4: 지역별 위험도 지도
# -------------------------------------------------------------------
elif menu == "🗺️ 지역별 위험도 지도":
    st.header("지역별 교통사고 위험도 지도")
    if risk_table is None:
        show_missing("위치 자료가 없습니다. 위도·경도와 사고건수를 포함한 위치별 CSV가 필요합니다.")
    else:
        map_df = risk_table.dropna(subset=["위도", "경도"]).copy()
        if map_df.empty:
            show_missing("지역별 사고 자료는 있지만 유효한 위도·경도 값이 없어 지도에 표시할 수 없습니다.")
        else:
            colors = {"높음": RED, "보통": "#E6A23C", "낮음": GREEN}
            fig = px.scatter_mapbox(
                map_df, lat="위도", lon="경도", color="위험도",
                size="사고건수", hover_name="지역",
                hover_data={"사고건수": True, "사망자수": True, "부상자수": True, "자료종류": True, "위도": False, "경도": False},
                color_discrete_map=colors, zoom=5, height=620,
                mapbox_style="open-street-map",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=35, b=0), legend_title="상대적 위험도")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("지도는 사고 다발지점의 좌표를 표시합니다. 도로 경로 또는 실제 통학 동선을 계산한 결과가 아닙니다.")

            st.subheader("지역별 사고 건수")
            region = risk_table.groupby("지역", as_index=False)["사고건수"].sum().sort_values("사고건수", ascending=False).head(20)
            st.plotly_chart(chart_layout(px.bar(region, x="사고건수", y="지역", orientation="h", color="사고건수", color_continuous_scale=["#B8D9E8", RED])), use_container_width=True)

            st.subheader("지도 지점 상세 정보")
            chosen = st.selectbox("확인할 지역 선택", map_df["지역"].drop_duplicates().tolist())
            selected = map_df[map_df["지역"] == chosen]
            st.dataframe(selected[["지역", "자료종류", "사고건수", "사망자수", "부상자수", "위험도"]], use_container_width=True, hide_index=True)

# -------------------------------------------------------------------
# 10. 메뉴 5: 머신러닝 위험도 분류
# -------------------------------------------------------------------
elif menu == "🤖 머신러닝 위험도 분류":
    st.header("머신러닝 기반 상대적 위험도 분류")
    st.write("위치별 사고건수·사망자수·부상자수를 이용해 사고 특성이 비슷한 지점을 군집화합니다.")
    st.warning("정답 위험도 레이블이 없는 경우 KMeans는 상대적 유사성을 묶는 비지도학습입니다. 결과는 실제 사고 발생 확률이나 공식 위험등급이 아닙니다.")

    supervised = supervised_result(ped if ped is not None else school)
    if supervised is not None:
        st.subheader("RandomForest 지도학습")
        if "error" in supervised:
            show_missing(supervised["error"])
        else:
            st.metric("평가 데이터 정확도", f"{supervised['accuracy']:.1%}")
            matrix = pd.DataFrame(supervised["matrix"], index=supervised["classes"], columns=supervised["classes"])
            st.write("혼동 행렬")
            st.dataframe(matrix, use_container_width=True)
            st.caption("학습·평가 데이터를 분리하여 평가했습니다. 정확도는 해당 데이터의 레이블 품질과 표본 구성에 영향을 받습니다.")
    else:
        st.info("실제 위험도 정답 레이블 열을 찾지 못해 RandomForest 지도학습은 실행하지 않았습니다.")

    st.subheader("KMeans 군집 분석 결과")
    if risk_table is None:
        show_missing("위치별 사고건수·사망자수 등의 데이터가 부족합니다.")
    else:
        st.caption("사고건수 65%, 사망자수 25%, 부상자수 10%의 상대적 순위 점수를 참고해 군집의 위험 수준을 해석했습니다.")
        st.dataframe(
            risk_table.sort_values("위험점수", ascending=False)[["지역", "자료종류", "사고건수", "사망자수", "부상자수", "위험도", "모델방식"]],
            use_container_width=True, hide_index=True
        )
        counts = risk_table["위험도"].value_counts().reindex(["높음", "보통", "낮음"]).fillna(0).reset_index()
        counts.columns = ["상대적 위험도", "지점 수"]
        st.plotly_chart(chart_layout(px.bar(counts, x="상대적 위험도", y="지점 수", color="상대적 위험도", color_discrete_map={"높음": RED, "보통": "#E6A23C", "낮음": GREEN})), use_container_width=True)
        top = risk_table.sort_values("위험점수", ascending=False).iloc[0]
        st.markdown(f"**상대적 위험도가 가장 높은 지점:** {top['지역']} — 사고 {int(top['사고건수']):,}건, 사망자 {int(top['사망자수']):,}명")

# -------------------------------------------------------------------
# 11. 메뉴 6: 교통안전시설 분석
# -------------------------------------------------------------------
elif menu == "🚸 교통안전시설 분석":
    st.header("교통안전시설 분석")
    facilities = data.get("facilities")
    if facilities is None:
        show_missing("현재 제공된 CSV에는 횡단보도·신호등·과속방지턱의 설치 위치나 지역별 설치 수가 없습니다. 교통안전시설 CSV가 추가되면 해당 열을 자동 탐색하여 분석할 수 있습니다.")
    else:
        st.write("불러온 시설 데이터의 실제 열을 확인합니다.")
        st.dataframe(facilities.head(30), use_container_width=True)
        location_col = find_col(facilities, "location")
        if location_col is None:
            show_missing("시설 자료에서 지역명 열을 찾지 못했습니다.")
        else:
            facility_types = [c for c in facilities.columns if any(k in normalize_name(c) for k in ["횡단보도", "신호등", "과속방지턱", "시설수", "설치수"])]
            if facility_types:
                st.dataframe(facilities[[location_col] + facility_types], use_container_width=True)
            else:
                show_missing("횡단보도·신호등·과속방지턱 관련 열을 찾지 못했습니다.")

    st.subheader("시설 데이터가 없는 경우의 분석 한계")
    st.write("사고가 많이 발생했다는 사실만으로 안전시설이 부족하다고 단정할 수 없습니다. 시설 설치 현황과 사고 위치를 함께 비교할 수 있는 자료가 있어야 시설 부족 지역을 판단할 수 있습니다.")

# -------------------------------------------------------------------
# 12. 메뉴 7: 안전 개선 방안
# -------------------------------------------------------------------
elif menu == "💡 안전 개선 방안":
    st.header("데이터 기반 안전한 통학 환경 개선 방안")
    st.write("아래 제안은 제공된 사고 통계의 관찰 결과를 바탕으로 한 참고 의견입니다. 실제 시설 설치 전에는 현장 조사와 관할 기관 검토가 필요합니다.")

    if risk_table is not None:
        top5 = risk_table.sort_values(["위험점수", "사고건수"], ascending=False).head(5)
        st.subheader("상대적 위험 지역 TOP 5")
        st.dataframe(top5[["지역", "자료종류", "사고건수", "사망자수", "부상자수", "위험도"]], use_container_width=True, hide_index=True)

        highest = top5.iloc[0]
        st.markdown(f"**우선 점검 제안:** {highest['지역']}은 제공된 위치 자료에서 상대적 위험 점수가 가장 높습니다. 사고 다발 지점의 횡단 동선, 차량 속도, 운전자 시야, 보행 공간을 현장에서 우선 점검하는 것이 좋습니다.")

        high_ped = risk_table[(risk_table["자료종류"] == "보행자 사고 다발지역")].sort_values("사고건수", ascending=False).head(5)
        if not high_ped.empty:
            st.subheader("보행자 사고 다발지역")
            st.dataframe(high_ped[["지역", "사고건수", "사망자수", "부상자수", "위험도"]], use_container_width=True, hide_index=True)
            st.write("보행자 사고가 많은 지점은 보행자 신호 운영, 횡단보도 시인성, 불법 주정차로 인한 시야 가림 여부를 우선 확인하도록 제안합니다.")

        high_school = risk_table[risk_table["자료종류"] == "어린이보호구역 사고 다발지역"].sort_values("사고건수", ascending=False).head(5)
        if not high_school.empty:
            st.subheader("어린이보호구역 사고 다발지역")
            st.dataframe(high_school[["지역", "사고건수", "사망자수", "부상자수", "위험도"]], use_container_width=True, hide_index=True)
            st.write("어린이보호구역에서는 제한속도 준수 여부, 과속방지시설, 보행자 대기 공간, 운전자 시야 확보 상태를 점검하는 것이 좋습니다.")
    else:
        show_missing("위치별 사고 다발 자료가 없어 지역별 TOP 5를 계산할 수 없습니다.")

    st.subheader("시간대별 안전 대책")
    if find_col(type_df, "hour") is None and find_col(monthly, "hour") is None:
        show_missing("시간대별 사고 자료가 없어 등교·하교 중 어느 시간대가 더 위험한지 판단할 수 없습니다. 등교·하교 시간대의 사고 원자료가 필요합니다.")
    else:
        st.write("시간대별 사고 데이터가 확인되면 사고 집중 시간대에 맞춰 교통지도와 보행 안전 캠페인을 제안할 수 있습니다.")

    st.subheader("안전시설 개선 제안")
    if data.get("facilities") is None:
        st.write("- 시설 설치 현황이 제공되지 않아 특정 지역에 횡단보도나 신호등이 부족하다고 단정하지 않습니다.")
        st.write("- 사고 다발지역을 현장 점검 후보로 선정하고, 횡단보도·보행자 신호·과속방지턱 설치 필요성을 조사합니다.")
    else:
        st.write("- 사고 다발지역과 안전시설 위치를 중첩해 시설 보강 우선순위를 확인합니다.")

    st.subheader("학생들이 실천할 수 있는 안전 수칙")
    st.markdown("""
    1. 횡단보도에서는 신호를 확인하고 좌우를 살핀 뒤 건넙니다.
    2. 주차 차량이나 건물 모퉁이에서는 갑자기 도로로 나가지 않습니다.
    3. 등·하교 중 스마트폰을 보거나 이어폰 음량을 크게 하지 않습니다.
    4. 차량이 멈춘 것을 확인한 뒤 횡단보도를 이용합니다.
    5. 사고 다발지점으로 표시된 곳에서는 평소보다 주변 차량을 더 주의 깊게 살핍니다.
    """)

# -------------------------------------------------------------------
# 13. 하단 안내
# -------------------------------------------------------------------
st.divider()
st.caption("본 앱은 제공된 CSV 통계를 시각화·분석하는 교육용 도구입니다. 데이터의 조사 기간과 집계 단위가 서로 다를 수 있으므로 서로 다른 파일의 합계를 단순 합산하지 않습니다.")
st.caption("안전 개선 제안은 통계적 참고 정보이며, 공식적인 교통안전 진단이나 실제 통학 경로 안내를 대체하지 않습니다.")

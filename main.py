# -*- coding: utf-8 -*-
"""안전한 등하굣길 분석실

학교 주변 교통사고 및 교통안전시설 CSV를 업로드해 분석하는 Streamlit 앱입니다.
실제 데이터의 열 이름은 파일마다 다를 수 있으므로, 화면에서 열을 직접 연결하도록 구성했습니다.
"""

import io
import math
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------------------
# 1. 앱 기본 설정 및 공통 함수
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🚦 안전한 등하굣길 분석실",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "🚦 안전한 등하굣길 분석실"
SOURCE_URL = (
    "https://opendata.koroad.or.kr/api/down/schoolzonedown.jsp;"
    "jsessionid=6E5E44E0393754208DAE0B3296827ECB"
)

RISK_COLORS = {"높음": "#E74C3C", "보통": "#F39C12", "낮음": "#27AE60"}
MAP_COLORS = ["#2878B5", "#2CA58D", "#8E6CBB", "#D18B26", "#777777"]


def normalize_name(value: object) -> str:
    """열 이름을 비교하기 쉽도록 공백과 기호를 제거합니다."""
    return re.sub(r"[^0-9a-z가-힣]", "", str(value).strip().lower())


def guess_column(columns: List[str], aliases: List[str]) -> Optional[str]:
    """열 이름에 포함된 단어를 바탕으로 연결할 열을 자동 추천합니다."""
    normalized = {col: normalize_name(col) for col in columns}
    alias_values = [normalize_name(alias) for alias in aliases]

    # 먼저 완전히 일치하는 열을 찾습니다.
    for alias in alias_values:
        for col, name in normalized.items():
            if name == alias:
                return col

    # 완전 일치가 없으면 부분적으로 일치하는 열을 찾습니다.
    for alias in alias_values:
        if len(alias) < 2:
            continue
        for col, name in normalized.items():
            if alias in name or name in alias:
                return col
    return None


def read_csv_safely(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """UTF-8을 먼저 시도하고, 한글이 깨지거나 실패하면 CP949도 시도합니다."""
    if uploaded_file is None:
        return None, None, None

    raw_bytes = uploaded_file.getvalue()
    errors = []
    for encoding in ("utf-8-sig", "cp949", "utf-8"):
        try:
            # 구분 기호를 자동 추정하고, 추정에 실패하면 기본 쉼표로 읽습니다.
            try:
                frame = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, sep=None, engine="python")
            except Exception:
                frame = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
            frame.columns = [str(column).strip() for column in frame.columns]
            return frame, encoding, None
        except Exception as error:
            errors.append(f"{encoding}: {error}")

    return None, None, "CSV 파일을 읽지 못했습니다. 파일 형식과 인코딩을 확인해 주세요. " + " / ".join(errors)


def numeric_series(series: Optional[pd.Series], index, default: float = 0.0) -> pd.Series:
    """숫자 열을 변환합니다. 쉼표가 들어간 숫자도 처리합니다."""
    if series is None:
        return pd.Series(default, index=index, dtype="float64")
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.replace({"": None, "nan": None, "None": None, "-": None})
    converted = pd.to_numeric(cleaned, errors="coerce")
    return converted.fillna(default).astype("float64")


def parse_boolean(value: object) -> bool:
    """보행자 여부 열에서 참에 해당하는 다양한 표현을 처리합니다."""
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    positive = ("예", "유", "y", "yes", "true", "1", "보행", "보행자", "pedestrian", "사람")
    negative = ("아니오", "무", "n", "no", "false", "0", "비보행")
    if any(token in text for token in negative):
        return False
    return any(token in text for token in positive)


def make_area_key(latitude: pd.Series, longitude: pd.Series) -> pd.Series:
    """행정구역 열을 사용할 수 없을 때 위·경도 3자리 격자로 지역을 구분합니다."""
    lat = pd.to_numeric(latitude, errors="coerce")
    lon = pd.to_numeric(longitude, errors="coerce")
    result = pd.Series("위치 정보 없음", index=latitude.index, dtype="object")
    valid = lat.notna() & lon.notna()
    result.loc[valid] = (
        "좌표격자 " + lat.loc[valid].round(3).astype(str) + ", "
        + lon.loc[valid].round(3).astype(str)
    )
    return result


def make_datetime(raw: pd.DataFrame, date_col: Optional[str], time_col: Optional[str]) -> pd.Series:
    """날짜·시간 열을 하나의 날짜 시간 값으로 변환합니다.

    YYYYMMDD처럼 숫자로 저장된 날짜와 HHMMSS 형태의 시간도 읽을 수 있도록 보정합니다.
    """
    if not date_col:
        return pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    date_text = raw[date_col].astype(str).str.strip()
    # 20261105 같은 8자리 날짜 코드는 숫자형으로 읽혀도 날짜 문자열로 처리합니다.
    compact_date = date_text.str.replace(r"\.0$", "", regex=True)
    is_yyyymmdd = compact_date.str.fullmatch(r"\d{8}")
    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    if is_yyyymmdd.any():
        parsed.loc[is_yyyymmdd] = pd.to_datetime(
            compact_date.loc[is_yyyymmdd], format="%Y%m%d", errors="coerce"
        )

    if time_col and time_col != date_col:
        time_text = raw[time_col].astype(str).str.strip()
        compact_time = time_text.str.replace(r"\D", "", regex=True)
        is_compact_time = compact_time.str.fullmatch(r"\d{1,6}")
        normalized_time = time_text.copy()
        normalized_time.loc[is_compact_time] = compact_time.loc[is_compact_time].str.zfill(6).str.replace(
            r"(\d{2})(\d{2})(\d{2})", r"\1:\2:\3", regex=True
        )
        combined = date_text + " " + normalized_time
        general = pd.to_datetime(combined, errors="coerce")
        # 날짜 열이 YYYYMMDD 숫자이면 위에서 만든 정규 날짜를 기준으로 다시 조합합니다.
        if is_yyyymmdd.any():
            combined_compact = compact_date + " " + normalized_time
            general_compact = pd.to_datetime(combined_compact, errors="coerce")
            general.loc[is_yyyymmdd] = general_compact.loc[is_yyyymmdd]
        return general

    general = pd.to_datetime(date_text, errors="coerce")
    general.loc[is_yyyymmdd] = parsed.loc[is_yyyymmdd]
    # YYYYMMDDHHMMSS처럼 14자리로 저장된 값도 추가로 처리합니다.
    is_full_datetime = compact_date.str.fullmatch(r"\d{14}")
    if is_full_datetime.any():
        general.loc[is_full_datetime] = pd.to_datetime(
            compact_date.loc[is_full_datetime], format="%Y%m%d%H%M%S", errors="coerce"
        )
    return general


def format_int(value: object) -> str:
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return "자료 없음"


def select_column(
    frame: pd.DataFrame,
    label: str,
    key: str,
    aliases: List[str],
    help_text: Optional[str] = None,
) -> Optional[str]:
    """사용자가 CSV 열을 분석 항목에 직접 연결하는 선택 상자입니다."""
    columns = [str(column) for column in frame.columns]
    options = ["선택 안 함"] + columns
    guessed = guess_column(columns, aliases)
    default_index = options.index(guessed) if guessed in options else 0
    selected = st.selectbox(label, options, index=default_index, key=key, help=help_text)
    return None if selected == "선택 안 함" else selected


def schema_table(
    raw_frame: Optional[pd.DataFrame], meanings: Dict[str, str], data_label: str
) -> pd.DataFrame:
    """원본 데이터의 열 이름, 뜻, 자료형, 빈 값 개수를 표로 정리합니다."""
    if raw_frame is None or raw_frame.empty:
        return pd.DataFrame(columns=["데이터", "열 이름", "우리말 뜻", "데이터 종류", "빈 값 개수"])

    rows = []
    for column in raw_frame.columns:
        rows.append(
            {
                "데이터": data_label,
                "열 이름": str(column),
                "우리말 뜻": meanings.get(str(column), "사용자 확인 필요"),
                "데이터 종류": str(raw_frame[column].dtype),
                "빈 값 개수": int(raw_frame[column].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 2. CSV 데이터 업로드 및 전처리 설정
# -----------------------------------------------------------------------------
st.title(APP_TITLE)
st.caption("학교 주변 교통사고와 안전시설 데이터를 살펴보고, 더 안전한 통학 환경을 생각해 보는 분석 대시보드입니다.")

with st.sidebar:
    st.header("📂 데이터 업로드")
    st.markdown("**① 교통사고 데이터 CSV**")
    accident_file = st.file_uploader("사고 데이터 파일", type=["csv"], key="accident_upload")
    st.markdown("**② 교통안전시설 데이터 CSV**")
    facility_file = st.file_uploader("안전시설 데이터 파일", type=["csv"], key="facility_upload")

    st.divider()
    st.header("🧹 전처리 설정")
    remove_duplicates = st.checkbox("완전히 같은 중복 행 제거", value=True)
    st.caption("빈 숫자 값은 0 또는 기본값으로 처리하고, 위치가 없는 행은 지도에서만 제외합니다.")
    st.divider()
    st.markdown("**데이터 출처**")
    st.markdown(f"[도로교통공단 어린이보호구역 데이터 페이지]({SOURCE_URL})")

raw_accidents, accident_encoding, accident_error = read_csv_safely(accident_file)
raw_facilities, facility_encoding, facility_error = read_csv_safely(facility_file)

if accident_error:
    st.sidebar.error(accident_error)
if facility_error:
    st.sidebar.error(facility_error)

accidents_duplicates_removed = 0
facilities_duplicates_removed = 0
if raw_accidents is not None and remove_duplicates:
    before = len(raw_accidents)
    raw_accidents = raw_accidents.drop_duplicates().copy()
    accidents_duplicates_removed = before - len(raw_accidents)
if raw_facilities is not None and remove_duplicates:
    before = len(raw_facilities)
    raw_facilities = raw_facilities.drop_duplicates().copy()
    facilities_duplicates_removed = before - len(raw_facilities)


# -----------------------------------------------------------------------------
# 3. 열 연결
# -----------------------------------------------------------------------------
acc_map: Dict[str, Optional[str]] = {}
fac_map: Dict[str, Optional[str]] = {}

if raw_accidents is not None:
    with st.expander("🔧 교통사고 데이터 열 연결하기", expanded=True):
        st.caption("파일의 열 이름이 달라도 아래 항목에 맞는 열을 선택하면 됩니다. 자동 추천된 값도 직접 바꿀 수 있습니다.")
        cols = st.columns(3)
        with cols[0]:
            acc_map["lat"] = select_column(raw_accidents, "사고 발생 위도", "acc_lat", ["위도", "latitude", "lat"])
            acc_map["lon"] = select_column(raw_accidents, "사고 발생 경도", "acc_lon", ["경도", "longitude", "lon", "lng"])
            acc_map["date"] = select_column(raw_accidents, "사고 날짜 또는 날짜·시간", "acc_date", ["발생일시", "사고일시", "발생일자", "사고일자", "일시", "datetime", "date"])
            acc_map["time"] = select_column(raw_accidents, "사고 시간(날짜 열과 분리된 경우)", "acc_time", ["발생시간", "사고시간", "시간", "time"])
        with cols[1]:
            acc_map["count"] = select_column(raw_accidents, "사고 발생 건수", "acc_count", ["사고건수", "발생건수", "건수", "사고수", "count"])
            acc_map["type"] = select_column(raw_accidents, "사고 유형", "acc_type", ["사고유형", "사고종류", "유형", "type"])
            acc_map["pedestrian"] = select_column(raw_accidents, "보행자 사고 여부", "acc_ped", ["보행자여부", "보행사고여부", "보행자", "pedestrian"])
            acc_map["region"] = select_column(raw_accidents, "지역명·행정구역", "acc_region", ["시군구", "행정구역", "지역명", "읍면동", "동명", "지역", "구역"])
        with cols[2]:
            acc_map["deaths"] = select_column(raw_accidents, "사망자 수", "acc_deaths", ["사망자수", "사망자", "사망", "death", "fatalities"])
            acc_map["injuries"] = select_column(raw_accidents, "부상자 수", "acc_injuries", ["부상자수", "부상자", "부상", "injury", "injuries"])
            acc_map["volume"] = select_column(raw_accidents, "차량 통행량(선택)", "acc_volume", ["교통량", "통행량", "차량수", "trafficvolume"])
            acc_map["speed"] = select_column(raw_accidents, "평균 통행속도(선택)", "acc_speed", ["평균속도", "통행속도", "속도", "speed"])

if raw_facilities is not None:
    with st.expander("🔧 교통안전시설 데이터 열 연결하기", expanded=True):
        st.caption("시설 위치와 종류를 선택해 주세요. 지역명이 양쪽 파일에 모두 있으면 지역명 기준으로 연결합니다.")
        cols = st.columns(2)
        with cols[0]:
            fac_map["lat"] = select_column(raw_facilities, "시설 위도", "fac_lat", ["위도", "latitude", "lat"])
            fac_map["lon"] = select_column(raw_facilities, "시설 경도", "fac_lon", ["경도", "longitude", "lon", "lng"])
        with cols[1]:
            fac_map["type"] = select_column(raw_facilities, "시설 종류", "fac_type", ["시설종류", "시설유형", "시설명", "종류", "type"])
            fac_map["region"] = select_column(raw_facilities, "시설 지역명·행정구역", "fac_region", ["시군구", "행정구역", "지역명", "읍면동", "동명", "지역", "구역"])


# -----------------------------------------------------------------------------
# 4. 표준 형태로 데이터 정리
# -----------------------------------------------------------------------------
use_region_key = bool(acc_map.get("region")) and (
    raw_facilities is None or bool(fac_map.get("region"))
)

normalized_accidents: Optional[pd.DataFrame] = None
normalized_facilities: Optional[pd.DataFrame] = None
pedestrian_data_available = False

if raw_accidents is not None:
    source = raw_accidents.copy()
    result = pd.DataFrame(index=source.index)

    result["위도"] = numeric_series(source[acc_map["lat"]] if acc_map.get("lat") else None, source.index, default=float("nan"))
    result["경도"] = numeric_series(source[acc_map["lon"]] if acc_map.get("lon") else None, source.index, default=float("nan"))
    # 위·경도 값이 실제 좌표 범위에 있는지 검사합니다.
    invalid_coordinates = (
        result["위도"].notna() & ~result["위도"].between(-90, 90)
    ) | (result["경도"].notna() & ~result["경도"].between(-180, 180))
    result.loc[invalid_coordinates, ["위도", "경도"]] = float("nan")

    if acc_map.get("count"):
        result["사고건수"] = numeric_series(source[acc_map["count"]], source.index, default=1).clip(lower=0)
        result["사고건수"] = result["사고건수"].replace(0, 1)
    else:
        # 사고 1행이 사고 1건을 뜻한다고 가정합니다.
        result["사고건수"] = 1.0

    result["사망자"] = numeric_series(source[acc_map["deaths"]] if acc_map.get("deaths") else None, source.index)
    result["부상자"] = numeric_series(source[acc_map["injuries"]] if acc_map.get("injuries") else None, source.index)
    result["사고유형"] = source[acc_map["type"]].fillna("미분류").astype(str) if acc_map.get("type") else "미분류"
    result["교통량"] = numeric_series(source[acc_map["volume"]] if acc_map.get("volume") else None, source.index, default=float("nan"))
    result["평균속도"] = numeric_series(source[acc_map["speed"]] if acc_map.get("speed") else None, source.index, default=float("nan"))

    if acc_map.get("date"):
        result["사고일시"] = make_datetime(source, acc_map.get("date"), acc_map.get("time"))
    else:
        result["사고일시"] = pd.Series(pd.NaT, index=source.index, dtype="datetime64[ns]")

    if acc_map.get("pedestrian"):
        pedestrian_values = source[acc_map["pedestrian"]].apply(parse_boolean)
        pedestrian_data_available = True
    elif acc_map.get("type"):
        pedestrian_values = source[acc_map["type"]].astype(str).str.contains(
            "보행|차대사람|횡단|pedestrian", case=False, na=False, regex=True
        )
        pedestrian_data_available = True
    else:
        pedestrian_values = pd.Series(False, index=source.index)
    result["보행자사고"] = pedestrian_values.astype(bool)

    if use_region_key and acc_map.get("region"):
        result["지역"] = source[acc_map["region"]].fillna("지역 정보 없음").astype(str).str.strip()
        result.loc[result["지역"].isin(["", "nan", "None"]), "지역"] = "지역 정보 없음"
    else:
        result["지역"] = make_area_key(result["위도"], result["경도"])

    normalized_accidents = result.reset_index(drop=True)

if raw_facilities is not None:
    source = raw_facilities.copy()
    result = pd.DataFrame(index=source.index)
    result["위도"] = numeric_series(source[fac_map["lat"]] if fac_map.get("lat") else None, source.index, default=float("nan"))
    result["경도"] = numeric_series(source[fac_map["lon"]] if fac_map.get("lon") else None, source.index, default=float("nan"))
    invalid_coordinates = (
        result["위도"].notna() & ~result["위도"].between(-90, 90)
    ) | (result["경도"].notna() & ~result["경도"].between(-180, 180))
    result.loc[invalid_coordinates, ["위도", "경도"]] = float("nan")
    result["시설종류"] = source[fac_map["type"]].fillna("종류 미상").astype(str) if fac_map.get("type") else "종류 미상"

    if use_region_key and fac_map.get("region"):
        result["지역"] = source[fac_map["region"]].fillna("지역 정보 없음").astype(str).str.strip()
        result.loc[result["지역"].isin(["", "nan", "None"]), "지역"] = "지역 정보 없음"
    else:
        result["지역"] = make_area_key(result["위도"], result["경도"])

    normalized_facilities = result.reset_index(drop=True)


# -----------------------------------------------------------------------------
# 5. 지역별 사고 집계 및 위험도 점수
# -----------------------------------------------------------------------------
def build_risk_table(
    accident_data: Optional[pd.DataFrame],
    facility_data: Optional[pd.DataFrame],
    has_pedestrian_data: bool,
) -> pd.DataFrame:
    """지역별 지표를 집계하고, 지표의 상대 순위를 이용해 0~100점 위험도를 계산합니다."""
    if accident_data is None or accident_data.empty:
        return pd.DataFrame()

    work = accident_data.copy()
    work["보행자사고건수"] = work["사고건수"].where(work["보행자사고"], 0)
    work["중대도지표"] = work["사망자"] * 5 + work["부상자"]

    aggregations = {
        "사고건수": ("사고건수", "sum"),
        "보행자사고건수": ("보행자사고건수", "sum"),
        "사망자": ("사망자", "sum"),
        "부상자": ("부상자", "sum"),
        "중대도지표": ("중대도지표", "sum"),
        "위도": ("위도", "mean"),
        "경도": ("경도", "mean"),
        "평균교통량": ("교통량", "mean"),
        "평균속도": ("평균속도", "mean"),
    }
    area_table = work.groupby("지역", dropna=False).agg(**aggregations).reset_index()

    has_facility_data = facility_data is not None and not facility_data.empty
    if has_facility_data:
        facility_counts = facility_data.groupby("지역").size().rename("안전시설수").reset_index()
        area_table = area_table.merge(facility_counts, on="지역", how="left")
        area_table["안전시설수"] = area_table["안전시설수"].fillna(0).astype(int)
    else:
        area_table["안전시설수"] = float("nan")

    # 점수가 높을수록 해당 위험 요소가 상대적으로 큰 지역입니다.
    components = []
    weights = {}

    area_table["사고빈도점수"] = area_table["사고건수"].rank(pct=True, method="average") * 100
    components.append("사고빈도점수")
    weights["사고빈도점수"] = 0.35

    if has_pedestrian_data:
        area_table["보행자점수"] = area_table["보행자사고건수"].rank(pct=True, method="average") * 100
        components.append("보행자점수")
        weights["보행자점수"] = 0.25

    if (accident_data["사망자"].sum() > 0) or (accident_data["부상자"].sum() > 0):
        area_table["사고심각도점수"] = area_table["중대도지표"].rank(pct=True, method="average") * 100
        components.append("사고심각도점수")
        weights["사고심각도점수"] = 0.25

    if has_facility_data:
        # 안전시설 수가 적을수록 시설 부족 점수가 높아집니다.
        area_table["시설부족점수"] = (1 - area_table["안전시설수"].rank(pct=True, method="average")) * 100
        components.append("시설부족점수")
        weights["시설부족점수"] = 0.10

    if work["교통량"].notna().any():
        area_table["교통량점수"] = area_table["평균교통량"].rank(pct=True, method="average") * 100
        components.append("교통량점수")
        weights["교통량점수"] = 0.05

    total_weight = sum(weights.values())
    if total_weight == 0:
        area_table["위험점수"] = 0.0
    else:
        area_table["위험점수"] = sum(
            area_table[component].fillna(50) * (weight / total_weight)
            for component, weight in weights.items()
        )

    # 70점 이상은 높음, 40점 이상은 보통으로 분류합니다.
    area_table["위험등급"] = area_table["위험점수"].apply(
        lambda score: "높음" if score >= 70 else ("보통" if score >= 40 else "낮음")
    )

    def explain_row(row: pd.Series) -> str:
        reasons = []
        if row["사고건수"] > 0:
            reasons.append(f"사고 {format_int(row['사고건수'])}건")
        if has_pedestrian_data:
            reasons.append(f"보행자 사고 {format_int(row['보행자사고건수'])}건")
        if row["사망자"] > 0 or row["부상자"] > 0:
            reasons.append(f"사망 {format_int(row['사망자'])}명·부상 {format_int(row['부상자'])}명")
        if has_facility_data:
            reasons.append(f"안전시설 {format_int(row['안전시설수'])}개")
        if pd.notna(row["평균교통량"]):
            reasons.append(f"평균 교통량 {row['평균교통량']:.1f}")
        return ", ".join(reasons) if reasons else "분석 가능한 위험 요인이 부족함"

    area_table["등급 판정 근거"] = area_table.apply(explain_row, axis=1)
    return area_table.sort_values(["위험점수", "사고건수"], ascending=False).reset_index(drop=True)


risk_table = build_risk_table(normalized_accidents, normalized_facilities, pedestrian_data_available)


# -----------------------------------------------------------------------------
# 6. 페이지 탭 구성
# -----------------------------------------------------------------------------
tabs = st.tabs(["📘 데이터 소개", "📊 사고 분석", "🚸 안전시설 분석", "🤖 위험도 분류 모델", "💡 개선 방안", "⬇️ 데이터 다운로드"])


# -----------------------------------------------------------------------------
# 탭 1: 데이터 소개
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("데이터로 더 안전한 통학 환경 만들기")
    st.write(
        "이 앱은 학교 주변 교통사고가 어디에서, 언제, 어떤 형태로 발생하는지 살펴보고 "
        "교통안전시설 분포와 비교해 위험 지역을 찾는 것을 목표로 합니다. "
        "분석 결과는 통학 안전을 위한 참고 자료이며, 실제 도로 조사와 함께 활용해야 합니다."
    )

    if normalized_accidents is None:
        st.info("왼쪽 메뉴에서 교통사고 CSV를 업로드해 주세요. 안전시설 CSV도 함께 올리면 사고 지역과 시설 분포를 비교할 수 있습니다.")
        st.markdown("**업로드 방법**")
        st.markdown("1. 도로교통공단 등에서 CSV 데이터를 준비합니다.\n2. 왼쪽 메뉴에서 사고 데이터와 안전시설 데이터를 각각 업로드합니다.\n3. 자동 추천된 열 연결을 확인하고, 맞지 않는 항목은 직접 선택합니다.")
        st.markdown(f"데이터 출처 예시: [도로교통공단 어린이보호구역 데이터 페이지]({SOURCE_URL})")
    else:
        accident_total = normalized_accidents["사고건수"].sum()
        pedestrian_total = normalized_accidents.loc[normalized_accidents["보행자사고"], "사고건수"].sum() if pedestrian_data_available else None
        deaths_total = normalized_accidents["사망자"].sum()
        injuries_total = normalized_accidents["부상자"].sum()
        facility_total = len(normalized_facilities) if normalized_facilities is not None else None

        cards = st.columns(4)
        cards[0].metric("전체 교통사고", format_int(accident_total) + "건")
        cards[1].metric("보행자 교통사고", format_int(pedestrian_total) + "건" if pedestrian_total is not None else "자료 없음")
        cards[2].metric("사망자 / 부상자", f"{format_int(deaths_total)}명 / {format_int(injuries_total)}명")
        cards[3].metric("교통안전시설", format_int(facility_total) + "개" if facility_total is not None else "자료 없음")

        st.caption(
            f"사고 데이터 {len(normalized_accidents):,}행 · 인코딩 {accident_encoding} · "
            f"중복 제거 {accidents_duplicates_removed:,}행"
        )
        if raw_facilities is not None:
            st.caption(
                f"안전시설 데이터 {len(normalized_facilities):,}행 · 인코딩 {facility_encoding} · "
                f"중복 제거 {facilities_duplicates_removed:,}행"
            )

        meanings = {}
        for key, meaning in {
            "lat": "사고 발생 위도", "lon": "사고 발생 경도", "date": "사고 날짜 또는 날짜·시간",
            "time": "사고 시간", "count": "사고 발생 건수", "type": "사고 유형",
            "pedestrian": "보행자 사고 여부", "region": "지역 또는 행정구역", "deaths": "사망자 수",
            "injuries": "부상자 수", "volume": "차량 통행량", "speed": "평균 통행속도",
        }.items():
            if acc_map.get(key):
                meanings[acc_map[key]] = meaning
        schema_frames = [schema_table(raw_accidents, meanings, "교통사고")]
        if raw_facilities is not None:
            facility_meanings = {}
            for key, meaning in {"lat": "시설 위도", "lon": "시설 경도", "type": "시설 종류", "region": "지역 또는 행정구역"}.items():
                if fac_map.get(key):
                    facility_meanings[fac_map[key]] = meaning
            schema_frames.append(schema_table(raw_facilities, facility_meanings, "교통안전시설"))
        st.markdown("#### 데이터 열 정보와 빈 값")
        st.dataframe(pd.concat(schema_frames, ignore_index=True), use_container_width=True, hide_index=True)

        missing_guidance = []
        if not acc_map.get("lat") or not acc_map.get("lon"):
            missing_guidance.append("사고 위도·경도 열이 연결되지 않아 사고 위치 지도를 만들 수 없습니다.")
        if not acc_map.get("date"):
            missing_guidance.append("사고 날짜 열이 연결되지 않아 시간대별 분석을 할 수 없습니다.")
        if not pedestrian_data_available:
            missing_guidance.append("보행자 여부 또는 사고 유형 열이 없어 보행자 사고를 구분할 수 없습니다.")
        if normalized_facilities is None:
            missing_guidance.append("안전시설 CSV가 없어 시설 분포와 사고 지역의 관계를 분석할 수 없습니다.")
        if missing_guidance:
            st.warning("분석 전 확인할 항목\n\n" + "\n".join([f"- {item}" for item in missing_guidance]))


# -----------------------------------------------------------------------------
# 탭 2: 교통사고 분석
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("📍 지역별·시간대별 교통사고 분석")
    if normalized_accidents is None or normalized_accidents.empty:
        st.info("교통사고 CSV를 먼저 업로드해 주세요.")
    else:
        if not acc_map.get("lat") or not acc_map.get("lon"):
            st.warning("위도와 경도 열을 연결하면 지도 분석을 사용할 수 있습니다.")
        else:
            map_accidents = normalized_accidents.dropna(subset=["위도", "경도"]).copy()
            if map_accidents.empty:
                st.warning("유효한 위도·경도 자료가 없습니다. 좌표 열 연결과 숫자 형식을 확인해 주세요.")
            else:
                map_accidents = map_accidents.merge(risk_table[["지역", "위험등급"]], on="지역", how="left")
                map_accidents["위험등급"] = map_accidents["위험등급"].fillna("보통")
                # 동일 좌표가 많으면 묶어서 표시해 지도를 읽기 쉽게 만듭니다.
                map_accidents["위도격자"] = map_accidents["위도"].round(4)
                map_accidents["경도격자"] = map_accidents["경도"].round(4)
                points = map_accidents.groupby(["위도격자", "경도격자", "지역", "위험등급"], as_index=False).agg(
                    사고건수=("사고건수", "sum")
                )
                fig_map = go.Figure()
                for level in ["높음", "보통", "낮음"]:
                    subset = points[points["위험등급"] == level]
                    if not subset.empty:
                        fig_map.add_trace(go.Scattermapbox(
                            lat=subset["위도격자"], lon=subset["경도격자"], mode="markers",
                            name=f"위험도 {level}",
                            marker={"size": (subset["사고건수"].clip(lower=1, upper=20) * 1.2 + 7).tolist(),
                                    "color": RISK_COLORS[level], "opacity": 0.75},
                            text=(subset["지역"] + " · 사고 " + subset["사고건수"].map(format_int) + "건").tolist(),
                            hoverinfo="text",
                        ))
                if normalized_facilities is not None:
                    map_facilities = normalized_facilities.dropna(subset=["위도", "경도"])
                    if not map_facilities.empty:
                        fig_map.add_trace(go.Scattermapbox(
                            lat=map_facilities["위도"], lon=map_facilities["경도"], mode="markers",
                            name="교통안전시설", marker={"size": 8, "color": "#2878B5", "symbol": "circle", "opacity": 0.8},
                            text=map_facilities["시설종류"].tolist(), hoverinfo="text",
                        ))
                center = {"lat": float(map_accidents["위도"].mean()), "lon": float(map_accidents["경도"].mean())}
                fig_map.update_layout(
                    mapbox={"style": "open-street-map", "center": center, "zoom": 12},
                    margin={"l": 0, "r": 0, "t": 10, "b": 0}, height=520,
                    legend={"orientation": "h", "yanchor": "bottom", "y": 0.01},
                )
                st.plotly_chart(fig_map, use_container_width=True)
                st.caption("붉은색은 위험도 높음, 주황색은 보통, 초록색은 낮음이며 파란색 점은 안전시설입니다. 지도는 업로드된 좌표 범위만 표시합니다.")

        st.markdown("#### 지역별 사고 발생 건수")
        if not risk_table.empty:
            region_chart = risk_table.sort_values("사고건수", ascending=False).head(20)
            fig_region = px.bar(
                region_chart, x="지역", y="사고건수", color="위험등급",
                color_discrete_map=RISK_COLORS, title="사고가 많이 발생한 지역 상위 20곳",
                labels={"지역": "지역", "사고건수": "사고 건수", "위험등급": "위험 등급"},
            )
            fig_region.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_region, use_container_width=True)

        st.markdown("#### 시간대별 사고 발생 추이")
        if normalized_accidents["사고일시"].notna().any():
            hourly = normalized_accidents.dropna(subset=["사고일시"]).copy()
            hourly["시간"] = hourly["사고일시"].dt.hour
            hourly_counts = hourly.groupby("시간")["사고건수"].sum().reindex(range(24), fill_value=0).rename_axis("시간").reset_index(name="사고건수")
            fig_hour = px.line(hourly_counts, x="시간", y="사고건수", markers=True,
                               title="0시부터 23시까지 시간대별 사고 건수",
                               labels={"시간": "사고 발생 시각", "사고건수": "사고 건수"})
            fig_hour.update_xaxes(dtick=1)
            st.plotly_chart(fig_hour, use_container_width=True)

            morning_count = hourly.loc[hourly["시간"].between(7, 9), "사고건수"].sum()
            evening_count = hourly.loc[hourly["시간"].between(15, 18), "사고건수"].sum()
            time_cols = st.columns(2)
            time_cols[0].metric("등교 시간 (07~09시)", format_int(morning_count) + "건")
            time_cols[1].metric("하교 시간 (15~18시)", format_int(evening_count) + "건")
            compare = pd.DataFrame({"시간대": ["등교 시간 (07~09시)", "하교 시간 (15~18시)"], "사고건수": [morning_count, evening_count]})
            fig_compare = px.bar(compare, x="시간대", y="사고건수", color="시간대", title="등교·하교 시간 사고 건수 비교")
            st.plotly_chart(fig_compare, use_container_width=True)
        else:
            st.info("날짜 또는 시간 열을 연결하면 시간대별 사고를 분석할 수 있습니다. 현재 날짜·시간 자료가 없거나 형식을 읽지 못했습니다.")

        st.markdown("#### 보행자 사고와 일반 사고")
        if pedestrian_data_available:
            pedestrian_count = normalized_accidents.loc[normalized_accidents["보행자사고"], "사고건수"].sum()
            general_count = normalized_accidents.loc[~normalized_accidents["보행자사고"], "사고건수"].sum()
            compare_ped = pd.DataFrame({"구분": ["보행자 사고", "일반 사고"], "사고건수": [pedestrian_count, general_count]})
            fig_ped = px.pie(compare_ped, names="구분", values="사고건수", hole=0.45, title="보행자 사고와 일반 사고 비율")
            st.plotly_chart(fig_ped, use_container_width=True)

            pedestrian_areas = risk_table.sort_values("보행자사고건수", ascending=False).head(15)
            fig_ped_area = px.bar(pedestrian_areas, x="지역", y="보행자사고건수", color="위험등급",
                                  color_discrete_map=RISK_COLORS, title="보행자 사고가 많은 지역",
                                  labels={"지역": "지역", "보행자사고건수": "보행자 사고 건수"})
            fig_ped_area.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_ped_area, use_container_width=True)

            pedestrian_rows = normalized_accidents[normalized_accidents["보행자사고"]]
            severity = pd.DataFrame({"구분": ["사망자", "부상자"],
                                     "인원": [pedestrian_rows["사망자"].sum(), pedestrian_rows["부상자"].sum()]})
            fig_severity = px.bar(severity, x="구분", y="인원", title="보행자 사고 사망자·부상자 현황")
            st.plotly_chart(fig_severity, use_container_width=True)
        else:
            st.info("보행자 여부 열이나 보행자 관련 사고 유형 열을 연결하면 보행자 사고 분석을 할 수 있습니다.")


# -----------------------------------------------------------------------------
# 탭 3: 교통안전시설 분석
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("🚸 교통안전시설 분포와 사고 지역 비교")
    if normalized_facilities is None or normalized_facilities.empty:
        st.info("안전시설 CSV를 업로드하고 위도·경도·시설 종류 열을 연결해 주세요.")
    else:
        facility_counts = normalized_facilities["시설종류"].value_counts().rename_axis("시설종류").reset_index(name="개수")
        fig_fac_type = px.bar(facility_counts, x="시설종류", y="개수", color="시설종류", title="시설 종류별 설치 개수",
                              labels={"시설종류": "시설 종류", "개수": "시설 개수"})
        st.plotly_chart(fig_fac_type, use_container_width=True)

        if not risk_table.empty:
            facility_area = risk_table.sort_values("사고건수", ascending=False).copy()
            facility_area["안전시설수표시"] = facility_area["안전시설수"].fillna(0)
            fig_relation = px.scatter(
                facility_area, x="안전시설수표시", y="사고건수", size="사고건수", color="위험등급",
                color_discrete_map=RISK_COLORS, hover_name="지역",
                hover_data={"보행자사고건수": True, "사망자": True, "부상자": True},
                title="지역별 안전시설 수와 교통사고 건수의 관계",
                labels={"안전시설수표시": "안전시설 개수", "사고건수": "교통사고 건수"},
            )
            st.plotly_chart(fig_relation, use_container_width=True)

            st.markdown("#### 사고가 많고 시설이 적은 지역")
            if "안전시설수" in risk_table.columns and risk_table["안전시설수"].notna().any():
                low_facility = risk_table.sort_values(["위험점수", "안전시설수"], ascending=[False, True]).head(10)
                st.dataframe(low_facility[["지역", "사고건수", "보행자사고건수", "안전시설수", "위험등급", "등급 판정 근거"]],
                             use_container_width=True, hide_index=True)
                st.caption("안전시설 수는 연결된 지역 단위에서 단순 집계한 값입니다. 시설 수가 적다고 곧바로 설치가 필요하다는 뜻은 아니며 현장 확인이 필요합니다.")
            else:
                st.warning("사고 지역과 시설 지역을 같은 기준으로 연결하지 못했습니다. 두 CSV에 공통된 지역명 열을 연결하거나 좌표 격자 기준으로 업로드해 주세요.")

        if not fac_map.get("lat") or not fac_map.get("lon"):
            st.warning("시설 위도·경도 열을 연결하지 않아 지도에서 시설 위치를 표시할 수 없습니다.")
        elif normalized_facilities.dropna(subset=["위도", "경도"]).empty:
            st.warning("유효한 시설 좌표가 없습니다. CSV의 좌표 형식을 확인해 주세요.")


# -----------------------------------------------------------------------------
# 탭 4: 위험도 분류 모델
# -----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("🤖 학교 주변 도로 위험도 분류")
    st.write("지역별 사고와 위험 요인을 점수화한 뒤 높음·보통·낮음 3단계로 구분합니다.")

    if risk_table.empty:
        st.info("위험도를 계산하려면 먼저 교통사고 데이터와 지역 구분에 필요한 열을 업로드해 주세요.")
    else:
        st.markdown("#### 위험도 산정 기준")
        st.write(
            "각 지표를 지역 간 백분위 순위(0~100점)로 바꾸고, 사용 가능한 지표에 가중치를 적용해 합산합니다. "
            "사고 빈도 35%, 보행자 사고 25%, 사고 심각도 25%, 안전시설 부족 10%, 차량 통행량 5%를 기본 가중치로 사용합니다. "
            "해당 자료가 없으면 그 지표의 가중치를 제외하고 나머지 가중치를 다시 비율화합니다. "
            "위험점수 70점 이상은 높음, 40점 이상 70점 미만은 보통, 40점 미만은 낮음입니다."
        )
        st.caption("사고 심각도 지표는 사망자 1명에 5점, 부상자 1명에 1점을 부여한 참고용 가중치입니다. 이는 공식 위험도 기준이 아닙니다.")
        if len(risk_table) < 3:
            st.warning("분석 지역이 3곳 미만입니다. 지역 간 비교가 제한되므로 위험 등급은 예비 결과로 해석해 주세요.")
        if normalized_facilities is None:
            st.warning("안전시설 데이터가 없어 시설 부족 요소는 위험 점수에서 제외했습니다.")
        if not pedestrian_data_available:
            st.warning("보행자 사고 자료가 없어 보행자 요소는 위험 점수에서 제외했습니다.")
        if not acc_map.get("deaths") and not acc_map.get("injuries"):
            st.warning("사망자·부상자 자료가 없어 사고 심각도 요소는 위험 점수에서 제외했습니다.")
        if not acc_map.get("volume"):
            st.warning("차량 통행량 자료가 없어 교통량 요소는 위험 점수에서 제외했습니다.")

        st.markdown("#### 지역별 위험도 결과")
        st.dataframe(
            risk_table[["지역", "사고건수", "보행자사고건수", "사망자", "부상자", "안전시설수", "위험점수", "위험등급", "등급 판정 근거"]],
            use_container_width=True, hide_index=True,
        )
        risk_chart = px.bar(risk_table.sort_values("위험점수", ascending=False), x="지역", y="위험점수", color="위험등급",
                            color_discrete_map=RISK_COLORS, title="지역별 위험 점수 (0~100점)",
                            labels={"지역": "지역", "위험점수": "위험 점수"})
        risk_chart.add_hline(y=70, line_dash="dash", annotation_text="높음 기준 70점")
        risk_chart.add_hline(y=40, line_dash="dot", annotation_text="보통 기준 40점")
        risk_chart.update_layout(xaxis_tickangle=-35, yaxis_range=[0, 100])
        st.plotly_chart(risk_chart, use_container_width=True)

        st.markdown("#### RandomForest 분류 모델")
        st.write(
            "실제 사고 위험 등급 정답 데이터가 제공되지 않았으므로, 위 점수 규칙으로 만든 등급을 학습용 임시 레이블로 사용합니다. "
            "따라서 아래 평가지표는 실제 사고를 예측하는 정확도가 아니라, RandomForest가 점수 규칙의 분류를 얼마나 재현하는지 보여줍니다."
        )

        feature_candidates = ["사고건수", "보행자사고건수", "사망자", "부상자", "안전시설수", "평균교통량", "평균속도"]
        feature_columns = [column for column in feature_candidates if column in risk_table.columns]
        model_data = risk_table[feature_columns].copy()
        model_data = model_data.replace([float("inf"), float("-inf")], float("nan"))
        model_data = model_data.fillna(0)
        target = risk_table["위험등급"].astype(str)
        class_counts = target.value_counts()
        enough_samples = (
            len(model_data) >= 30
            and len(class_counts) == 3
            and (class_counts >= 3).all()
        )

        if enough_samples:
            try:
                x_train, x_test, y_train, y_test = train_test_split(
                    model_data, target, test_size=0.25, random_state=42, stratify=target
                )
                model = RandomForestClassifier(
                    n_estimators=250, max_depth=6, min_samples_leaf=2,
                    class_weight="balanced", random_state=42,
                )
                model.fit(x_train, y_train)
                predictions = model.predict(x_test)
                accuracy = accuracy_score(y_test, predictions)

                metric_cols = st.columns(3)
                metric_cols[0].metric("규칙 등급 재현 정확도", f"{accuracy * 100:.1f}%")
                metric_cols[1].metric("학습 데이터", f"{len(x_train):,}개 지역")
                metric_cols[2].metric("평가 데이터", f"{len(x_test):,}개 지역")

                labels = [label for label in ["높음", "보통", "낮음"] if label in set(target)]
                matrix = confusion_matrix(y_test, predictions, labels=labels)
                fig_matrix = px.imshow(
                    matrix, text_auto=True, x=[f"예측 {label}" for label in labels],
                    y=[f"실제 {label}" for label in labels], color_continuous_scale="Blues",
                    title="혼동 행렬 (점수 규칙 등급과 모델 분류 비교)",
                    labels={"x": "모델 예측", "y": "점수 규칙 등급", "color": "개수"},
                )
                st.plotly_chart(fig_matrix, use_container_width=True)

                importance = pd.DataFrame({"분석 요소": feature_columns, "중요도": model.feature_importances_}).sort_values("중요도", ascending=False)
                fig_importance = px.bar(importance, x="중요도", y="분석 요소", orientation="h", title="RandomForest 변수 중요도")
                st.plotly_chart(fig_importance, use_container_width=True)
                st.success("모델 학습과 평가를 완료했습니다. 평가는 무작위로 분리한 평가 데이터에서 수행했습니다.")
            except Exception as error:
                st.warning(f"모델 학습 중 문제가 발생했습니다: {error}")
        else:
            st.info(
                f"모델 학습을 진행하지 않았습니다. 현재 지역 수는 {len(model_data)}개이며, "
                "학습에는 최소 30개 지역과 각 위험 등급별 3개 이상의 지역이 필요합니다. "
                "사고·시설 데이터를 더 확보한 뒤 다시 분석해 주세요."
            )


# -----------------------------------------------------------------------------
# 탭 5: 안전한 통학 환경 개선 방안
# -----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("💡 분석 결과를 바탕으로 한 개선 방안")
    if risk_table.empty:
        st.info("사고 데이터를 업로드하면 위험 지역과 개선 방향을 제안합니다.")
    else:
        high_risk = risk_table[risk_table["위험등급"] == "높음"].sort_values("위험점수", ascending=False)
        low_facility_areas = risk_table[risk_table["안전시설수"].notna()].sort_values(
            ["안전시설수", "사고건수"], ascending=[True, False]
        )

        st.markdown("#### 우선 확인할 위험 지역")
        if high_risk.empty:
            st.write("현재 설정한 기준(위험점수 70점 이상)에 해당하는 지역은 없습니다. 그렇더라도 낮음·보통 지역의 사고 예방 활동은 필요합니다.")
        else:
            st.dataframe(high_risk[["지역", "사고건수", "보행자사고건수", "사망자", "부상자", "위험점수", "등급 판정 근거"]],
                         use_container_width=True, hide_index=True)

        st.markdown("#### 시설이 적은 지역")
        if not low_facility_areas.empty:
            st.dataframe(low_facility_areas.head(10)[["지역", "사고건수", "안전시설수", "위험등급"]],
                         use_container_width=True, hide_index=True)
        else:
            st.write("시설 데이터가 없거나 지역 연결이 되지 않아 시설 부족 지역을 판정할 수 없습니다.")

        recommendations = []
        if not high_risk.empty:
            names = ", ".join(high_risk["지역"].head(3).astype(str).tolist())
            recommendations.append(f"위험점수가 높은 {names} 주변을 우선 현장 점검 대상으로 제안합니다.")
        if not low_facility_areas.empty:
            names = ", ".join(low_facility_areas["지역"].head(3).astype(str).tolist())
            recommendations.append(f"안전시설 수가 적게 집계된 {names} 지역은 횡단보도·보행 신호기·과속방지턱의 설치 상태와 필요성을 확인합니다.")
        if pedestrian_data_available:
            ped_high = risk_table.sort_values("보행자사고건수", ascending=False).iloc[0]
            recommendations.append(
                f"보행자 사고가 가장 많이 집계된 {ped_high['지역']} 주변은 보행자 시야 확보, 불법 주정차 관리, 횡단 대기 공간을 점검합니다."
            )
        if normalized_accidents["사고일시"].notna().any():
            times = normalized_accidents.dropna(subset=["사고일시"]).copy()
            times["시간"] = times["사고일시"].dt.hour
            morning = times.loc[times["시간"].between(7, 9), "사고건수"].sum()
            evening = times.loc[times["시간"].between(15, 18), "사고건수"].sum()
            busier = "등교 시간(07~09시)" if morning > evening else "하교 시간(15~18시)" if evening > morning else "등교·하교 시간 모두"
            recommendations.append(f"분석 기간에는 {busier}의 사고 건수가 상대적으로 많거나 비슷하므로, 해당 시간에 교문 주변 보행 지도와 차량 서행 안내를 강화합니다.")

        recommendations.extend([
            "학교·지자체·경찰과 함께 위험 지역의 횡단보도 시인성, 신호 주기, 조명, 보도 폭, 차량 속도를 현장에서 확인합니다.",
            "등하교 시간대 학교 주변 차량 진입·회차 동선을 점검하고, 불법 주정차로 가려지는 보행자 시야를 확보합니다.",
            "학생에게 사고 다발 지점과 안전한 횡단 방법을 안내하고, 보호자·운전자에게 학교 주변 서행을 지속적으로 홍보합니다.",
        ])
        for number, recommendation in enumerate(recommendations, start=1):
            st.markdown(f"**{number}.** {recommendation}")

        st.warning(
            "이 분석은 업로드된 사고·시설 자료의 범위 안에서 위험 신호를 찾는 참고 자료입니다. "
            "실제 보행 경로, 도로 폭, 신호 주기, 공사 구간, 차량 흐름을 모두 반영한 최적 통학 경로를 계산한 것은 아닙니다. "
            "횡단보도·신호등 신규 설치 여부는 반드시 현장 조사와 관계 기관 검토가 필요합니다."
        )


# -----------------------------------------------------------------------------
# 탭 6: 데이터 및 분석 결과 다운로드
# -----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("⬇️ 분석 결과 저장")
    if normalized_accidents is None:
        st.info("데이터를 업로드하면 정리된 데이터와 분석 결과를 CSV로 내려받을 수 있습니다.")
    else:
        st.write("다운로드 파일은 엑셀에서 한글이 깨지는 것을 줄이기 위해 UTF-8 BOM 인코딩으로 저장합니다.")
        cleaned_accidents = normalized_accidents.copy()
        cleaned_accidents["사고일시"] = cleaned_accidents["사고일시"].astype(str)
        st.download_button(
            "📥 정리된 교통사고 데이터 CSV 다운로드",
            data=cleaned_accidents.to_csv(index=False).encode("utf-8-sig"),
            file_name="정리된_교통사고_데이터.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if normalized_facilities is not None:
            st.download_button(
                "📥 정리된 교통안전시설 데이터 CSV 다운로드",
                data=normalized_facilities.to_csv(index=False).encode("utf-8-sig"),
                file_name="정리된_교통안전시설_데이터.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if not risk_table.empty:
            st.download_button(
                "📥 지역별 위험도 분석 결과 CSV 다운로드",
                data=risk_table.to_csv(index=False).encode("utf-8-sig"),
                file_name="지역별_교통사고_위험도_분석.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown("#### 현재 분석 결과 미리보기")
            st.dataframe(risk_table, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# 하단 안내
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    "안내: 이 앱은 학습·탐구 목적의 데이터 분석 도구입니다. 위험도 점수와 등급은 업로드된 데이터에 기반한 비교용 지표이며, "
    "공식 교통안전 진단이나 실제 사고 예측을 대신하지 않습니다. 데이터의 수집 기간·범위·정확성을 확인한 뒤 결과를 해석해 주세요."
)

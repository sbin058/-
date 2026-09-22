import streamlit as st
import pandas as pd
import plotly.express as px

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

st.set_page_config(page_title="영화 데이터 그래프 도감 2 - 분포와 관계", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)

    # 개봉일(여덟 자리 숫자) -> 날짜형
    df["openDt"] = pd.to_datetime(df["openDt"].astype(str), format="%Y%m%d", errors="coerce")

    # 장르: 세로막대(|) 기호로 여러 개 적힌 경우 첫 번째 장르만 사용
    df["genre"] = df["genre"].astype(str).str.split("|").str[0].str.strip()

    # 숫자 열 정리 (혹시 콤마가 섞여 문자열로 들어온 경우 대비)
    num_cols = ["first_scrn", "first_show", "first_week_audi", "total_audi", "days_in_top10"]
    for col in num_cols:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def insight_box(key: str):
    """그래프 아래에 '이 그래프로 알 수 있는 것'을 적는 자리"""
    st.text_area(
        "💡 이 그래프로 알 수 있는 것",
        placeholder="이 그래프를 보고 알 수 있는 내용을 한 문장으로 적어 보세요.",
        key=key,
    )


# ---------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------
st.title("🎬 영화 데이터 그래프 도감 2 - 분포와 관계")
st.caption(
    "1년간 박스오피스 10위권에 든 영화 가운데 그 기간에 개봉한 216편의 데이터를 활용합니다. "
    "(출처: KOBIS, greatsong/modudata)"
)

with st.spinner("데이터를 불러오는 중입니다..."):
    df = load_data()

with st.expander("📄 원본 데이터 미리보기"):
    st.dataframe(df, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# 1. 장르별 영화 편수 - 도넛 그래프
# ---------------------------------------------------------
st.header("1️⃣ 장르별 영화 편수")

genre_counts = df["genre"].value_counts().reset_index()
genre_counts.columns = ["genre", "count"]

fig_donut = px.pie(
    genre_counts,
    names="genre",
    values="count",
    hole=0.5,
    title="장르별 영화 편수",
)
fig_donut.update_traces(
    textinfo="label+percent",
    hovertemplate="장르: %{label}<br>편수: %{value}편<br>비율: %{percent}<extra></extra>",
)
st.plotly_chart(fig_donut, use_container_width=True)

insight_box("insight_1")

st.divider()

# ---------------------------------------------------------
# 2. 총 관객수 분포 - 히스토그램
# ---------------------------------------------------------
st.header("2️⃣ 총 관객수의 분포")

fig_hist = px.histogram(
    df,
    x="total_audi",
    nbins=30,
    title="영화별 총 관객수 분포",
    labels={"total_audi": "총 관객수"},
)
fig_hist.update_layout(yaxis_title="영화 편수")
st.plotly_chart(fig_hist, use_container_width=True)

insight_box("insight_2")

st.divider()

# ---------------------------------------------------------
# 3. 장르별 총 관객수 분포 - 박스플롯
# ---------------------------------------------------------
st.header("3️⃣ 장르별 총 관객수 분포")

fig_box = px.box(
    df,
    x="genre",
    y="total_audi",
    title="장르별 총 관객수 분포",
    labels={"genre": "장르", "total_audi": "총 관객수"},
    points="all",
)
st.plotly_chart(fig_box, use_container_width=True)

insight_box("insight_3")

st.divider()

# ---------------------------------------------------------
# 4. 개봉일 스크린수와 총 관객수의 관계 - 산점도
# ---------------------------------------------------------
st.header("4️⃣ 개봉일 스크린수와 총 관객수의 관계")

fig_scatter1 = px.scatter(
    df,
    x="first_scrn",
    y="total_audi",
    color="genre",
    hover_name="movieNm",
    title="개봉일 스크린수 vs 총 관객수",
    labels={"first_scrn": "개봉일 스크린수", "total_audi": "총 관객수"},
)
st.plotly_chart(fig_scatter1, use_container_width=True)

insight_box("insight_4")

st.divider()

# ---------------------------------------------------------
# 5. 개봉 첫 주 관객과 총 관객수의 관계 (10위권 유지 일수로 크기 표현)
# ---------------------------------------------------------
st.header("5️⃣ 개봉 첫 주 관객수와 총 관객수의 관계")

fig_scatter2 = px.scatter(
    df,
    x="first_week_audi",
    y="total_audi",
    size="days_in_top10",
    color="genre",
    hover_name="movieNm",
    title="개봉 첫 주 관객수 vs 총 관객수 (점 크기 = 10위권 유지 일수)",
    labels={"first_week_audi": "개봉 첫 주 관객수", "total_audi": "총 관객수"},
)
st.plotly_chart(fig_scatter2, use_container_width=True)

insight_box("insight_5")

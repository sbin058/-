import numpy as np
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
# 2. 장르 안의 영화 - 트리맵 (칸 크기 = 총 관객수)
# ---------------------------------------------------------
st.header("2️⃣ 장르 안에 들어 있는 영화 (트리맵)")

fig_treemap = px.treemap(
    df,
    path=[px.Constant("전체"), "genre", "movieNm"],
    values="total_audi",
    color="genre",
    title="장르 안의 영화 - 칸 크기는 총 관객수",
)
fig_treemap.update_traces(
    hovertemplate="%{label}<br>총 관객수: %{value:,.0f}명<extra></extra>"
)
st.plotly_chart(fig_treemap, use_container_width=True)

insight_box("insight_2")

st.divider()

# ---------------------------------------------------------
# 3. 총 관객수 분포 - 히스토그램 (+ 자동 계산 문구)
# ---------------------------------------------------------
st.header("3️⃣ 총 관객수의 분포")

fig_hist = px.histogram(
    df,
    x="total_audi",
    nbins=30,
    title="영화별 총 관객수 분포",
    labels={"total_audi": "총 관객수"},
)
fig_hist.update_layout(yaxis_title="영화 편수")
st.plotly_chart(fig_hist, use_container_width=True)

# 가장 영화가 몰려 있는 구간 계산
audi_values = df["total_audi"].dropna().values
counts, bin_edges = np.histogram(audi_values, bins=30)
max_idx = counts.argmax()
range_low, range_high = bin_edges[max_idx], bin_edges[max_idx + 1]
movies_in_range = int(counts[max_idx])

# 가장 관객이 많은 영화 계산
top_movie = df.loc[df["total_audi"].idxmax()]

st.info(
    f"📊 전체 216편 중 **{movies_in_range}편**이 총 관객수 "
    f"**{range_low:,.0f}명 ~ {range_high:,.0f}명** 구간에 몰려 있습니다. "
    f"가장 많은 관객을 동원한 영화는 **'{top_movie['movieNm']}'**"
    f"(총 관객 {top_movie['total_audi']:,.0f}명)입니다."
)

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
# 5. 장르별 총 관객수 분포 - 박스플롯 (영화 10편 이상인 장르만)
# ---------------------------------------------------------
st.header("5️⃣ 장르별 총 관객수 분포 (영화 10편 이상인 장르만)")

genres_ge10 = genre_counts.loc[genre_counts["count"] >= 10, "genre"]
df_box = df[df["genre"].isin(genres_ge10)]

fig_box = px.box(
    df_box,
    x="genre",
    y="total_audi",
    points="outliers",
    hover_data=["movieNm"],
    title="장르별 총 관객수 분포 (영화 10편 이상인 장르)",
    labels={"genre": "장르", "total_audi": "총 관객수"},
)
st.plotly_chart(fig_box, use_container_width=True)

insight_box("insight_5")

st.divider()

# ---------------------------------------------------------
# 6. 개봉일 스크린수 - 총 관객수 - 첫 주 관객수 관계 - 버블 그래프
# ---------------------------------------------------------
st.header("6️⃣ 개봉일 스크린수와 총 관객수의 관계 (버블 크기 = 첫 주 관객수)")

fig_bubble1 = px.scatter(
    df,
    x="first_scrn",
    y="total_audi",
    size="first_week_audi",
    color="genre",
    hover_name="movieNm",
    size_max=40,
    title="개봉일 스크린수 vs 총 관객수 (버블 크기: 개봉 첫 주 관객수)",
    labels={"first_scrn": "개봉일 스크린수", "total_audi": "총 관객수"},
)
st.plotly_chart(fig_bubble1, use_container_width=True)

insight_box("insight_6")

st.divider()

# ---------------------------------------------------------
# 7. 제작 국가 -> 장르 - 선버스트 (칸 크기 = 영화 편수)
# ---------------------------------------------------------
st.header("7️⃣ 제작 국가별 장르 구성 (선버스트)")

fig_sunburst = px.sunburst(
    df,
    path=["nation", "genre"],
    title="제작 국가 → 장르 - 칸 크기는 영화 편수",
)
fig_sunburst.update_traces(
    hovertemplate="%{label}<br>편수: %{value}편<extra></extra>"
)
st.plotly_chart(fig_sunburst, use_container_width=True)

insight_box("insight_7")

st.divider()

# ---------------------------------------------------------
# 8. 나만의 질문 - 10위권에 오래 머문 영화는 총 관객도 많은가 (산점도)
# ---------------------------------------------------------
st.header("8️⃣ 나만의 8번째 질문 — 만들어서 분석하기")

my_question = "영화는 어느 계절에 개봉될 때 총 관객수가 많은가?"
st.markdown(f"**내 질문:** {my_question}")
st.caption(
    "지금까지의 그래프는 genre·nation·first_scrn·first_week_audi·days_in_top10만 다뤘고 "
    "openDt(개봉일)는 아직 쓰이지 않았습니다. 개봉일에서 '계절'을 뽑아 총 관객수와 비교해 보았습니다."
)

season_map = {
    12: "겨울", 1: "겨울", 2: "겨울",
    3: "봄", 4: "봄", 5: "봄",
    6: "여름", 7: "여름", 8: "여름",
    9: "가을", 10: "가을", 11: "가을",
}
season_order = ["봄", "여름", "가을", "겨울"]

df["season"] = df["openDt"].dt.month.map(season_map)
df_season = df.dropna(subset=["season", "total_audi"])

fig_season = px.box(
    df_season,
    x="season",
    y="total_audi",
    points="outliers",
    hover_data=["movieNm"],
    category_orders={"season": season_order},
    title=my_question,
    labels={"season": "개봉 계절", "total_audi": "총 관객수"},
)
st.plotly_chart(fig_season, use_container_width=True)

insight_box("insight_8")

st.divider()

# ---------------------------------------------------------
# 9. 개봉 첫 주 관객과 총 관객수의 관계 (10위권 유지 일수로 크기 표현)
# ---------------------------------------------------------
st.header("9️⃣ 개봉 첫 주 관객수와 총 관객수의 관계")

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

insight_box("insight_9")

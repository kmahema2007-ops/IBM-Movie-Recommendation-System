import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION & RESPONSIVE LAYOUT ---
st.set_page_config(page_title="Movie Recommendation System", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM THEME & SPLIT SCREEN STYLING ---
# Injecting CSS for a half-white background and an animated movie clip panel wrapped in film tape borders
st.markdown("""
<style>
    /* Main Content Styling */
    .stApp {
        background-color: #f7f9fa; /* Half white / light grey aesthetic */
    }
    
    /* Film Tape Rolling Header Box */
    .film-strip-container {
        background: #111;
        border-top: 16px dashed #333;
        border-bottom: 16px dashed #333;
        padding: 20px 0;
        margin-bottom: 25px;
        overflow: hidden;
        width: 100%;
    }
    
    /* Horizontal marquee simulation for movie poster clips */
    .marquee {
        display: flex;
        width: 200%;
        animation: marquee 25s linear infinite;
    }
    .marquee img {
        height: 140px;
        margin: 0 10px;
        border-radius: 4px;
        border: 2px solid #fff;
    }
    @keyframes marquee {
        0% { transform: translateX(0%); }
        100% { transform: translateX(-50%); }
    }
    
    /* Custom Card Design */
    .movie-card {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        color: white;
    }
    .top-card { background-color: #1b4d3e; border-left: 5px solid #2ecc71; }
    .flop-card { background-color: #78281f; border-left: 5px solid #e74c3c; }
</style>
""", unsafe_allow_html=True)

# --- MOCK FILM STRIP HEADER PANEL ---
st.markdown("""
<div class="film-strip-container">
    <div class="marquee">
        <!-- Sample movie placeholders acting as your auto-rolling background clips -->
        <img src="https://unsplash.com" alt="Clip 1">
        <img src="https://unsplash.com" alt="Clip 2">
        <img src="https://unsplash.com" alt="Clip 3">
        <img src="https://unsplash.com" alt="Clip 4">
        <img src="https://unsplash.com" alt="Clip 5">
        <!-- Duplicate loop for continuous animation effect -->
        <img src="https://unsplash.com" alt="Clip 1">
        <img src="https://unsplash.com" alt="Clip 2">
        <img src="https://unsplash.com" alt="Clip 3">
        <img src="https://unsplash.com" alt="Clip 4">
        <img src="https://unsplash.com" alt="Clip 5">
    </div>
</div>
""", unsafe_allow_html=True)

st.title("🎬 MOVIE RECOMMENDATION SYSTEM")

# --- GENERATE DATASET (300 Movies) ---
@st.cache_data
def load_movie_data():
    np.random.seed(42)
    genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Romance']
    languages = ['English', 'Spanish', 'French', 'Hindi', 'Japanese']
    
    movie_pool = {
        'Action': ['Skyline', 'Vanguard', 'Apex', 'Grid', 'Bullet', 'Chaser', 'Overdrive'],
        'Comedy': ['Chuckles', 'The Prank', 'Giggles', 'High Jinks', 'Mixed Up', 'Out of Town'],
        'Drama': ['The Silhouette', 'Traces', 'Echoes', 'Riverbed', 'Shades of Gray', 'Stardust'],
        'Sci-Fi': ['Quantum Code', 'Nebula', 'Cybernetic', 'Vector One', 'Chronos', 'Deep Space'],
        'Horror': ['The Haunting', 'Red Room', 'Cursed', 'Nightfall', 'Whisper', 'Unseen'],
        'Romance': ['Autumn Love', 'Midnight Kiss', 'True Path', 'Heartstrings', 'Written in Stars']
    }

    data = []
    for i in range(1, 301):
        g = np.random.choice(genres)
        l = np.random.choice(languages)
        title_base = np.random.choice(movie_pool[g])
        title = f"{title_base} {np.random.randint(10, 99)}"
        rating = round(np.random.uniform(1.0, 10.0), 1)
        views = int(np.random.randint(500, 1000000))
        data.append([title, g, l, rating, views])
        
    return pd.DataFrame(data, columns=['Title', 'Genre', 'Language', 'Rating', 'Views'])

df = load_movie_data()

# --- SIDEBAR INPUT FILTERS ---
st.sidebar.header("🎯 Filter Options")
available_languages = df['Language'].unique()
available_genres = df['Genre'].unique()

selected_lang = st.sidebar.selectbox("Select Language", available_languages)
selected_genre = st.sidebar.selectbox("Select Genre", available_genres)

# --- FILTERED METRICS CALCULATIONS ---
filtered_df = df[(df['Language'] == selected_lang) & (df['Genre'] == selected_genre)]

# Ranking strategy: Sort primarily by Rating, then total Views
top_movies = filtered_df.sort_values(by=['Rating', 'Views'], ascending=False).head(3)
flop_movies = filtered_df.sort_values(by=['Rating', 'Views'], ascending=True).head(3)

# --- DISPLAY TOP vs FLOP TABLES ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 Top 3 Recommended Movies")
    if not top_movies.empty:
        for idx, row in top_movies.iterrows():
            st.markdown(f"""
            <div class='movie-card top-card'>
                <h4>🏆 {row['Title']}</h4>
                <p>⭐ <b>Rating:</b> {row['Rating']}/10 &nbsp;|&nbsp; 👁️ <b>Views:</b> {row['Views']:,}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("No matching titles found.")

with col2:
    st.subheader("📉 Flop Movies (Lowest Rated)")
    if not flop_movies.empty:
        for idx, row in flop_movies.iterrows():
            st.markdown(f"""
            <div class='movie-card flop-card'>
                <h4>⚠️ {row['Title']}</h4>
                <p>⭐ <b>Rating:</b> {row['Rating']}/10 &nbsp;|&nbsp; 👁️ <b>Views:</b> {row['Views']:,}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("No matching titles found.")

# --- 3D DATA VISUALIZATIONS SECTION ---
st.write("---")
st.subheader("📊 3D Global Dataset Analytics")
st.caption("Each visualization box provides deep interactive analysis on the complete 300 movie library.")

# Helper tool to convert 2D data vectors into 3D z-matrix meshgrids for charts
def generate_3d_mesh(dataframe, x_col, y_col):
    pivot = dataframe.groupby([x_col, y_col]).size().unstack(fill_value=0)
    return pivot.columns.tolist(), pivot.index.tolist(), pivot.values

# 1. 3D Bar Graph
with st.expander("📈 1. 3D Bar Graph - Genre vs Language Counts", expanded=True):
    langs, genrs, z_matrix = generate_3d_mesh(df, 'Genre', 'Language')
    fig_bar = go.Figure(data=[go.Surface(z=z_matrix, x=langs, y=genrs, colorscale='Viridis')])
    fig_bar.update_layout(title='3D Distribution Matrix', scene=dict(xaxis_title='Language', yaxis_title='Genre', zaxis_title='Movie Count'))
    st.plotly_chart(fig_bar, use_container_width=True)

# 2. 3D Histogram
with st.expander("📊 2. 3D Histogram - Rating & Views Groupings"):
    fig_hist = go.Figure(data=[go.Histogram2dContour(x=df['Rating'], y=df['Views'], colorscale='Blues', reversescale=True)])
    fig_hist.update_layout(title='Density Cluster Map', xaxis_title='Ratings', yaxis_title='Total Viewers')
    st.plotly_chart(fig_hist, use_container_width=True)

# 3. 3D Line Chart
with st.expander("📉 3. 3D Line Chart - Rating and Views Trajectory"):
    df_sorted = df.sort_values(by='Rating')
    fig_line = go.Figure(data=[go.Scatter3d(x=df_sorted['Rating'], y=df_sorted['Views'], z=df_sorted.index, mode='lines', line=dict(color='red', width=4))])
    fig_line.update_layout(title='Sequential Performance Path', scene=dict(xaxis_title='Rating', yaxis_title='Views', zaxis_title='Index ID'))
    st.plotly_chart(fig_line, use_container_width=True)

# 4. 3D Scatter Plot
with st.expander("✨ 4. 3D Scatter Plot - Volume Matrix Breakdown"):
    fig_scatter = px.scatter_3d(df, x='Rating', y='Views', z='Genre', color='Language', size='Rating', hover_name='Title', opacity=0.8)
    fig_scatter.update_layout(title='Multi-dimensional Metric Map')
    st.plotly_chart(fig_scatter, use_container_width=True)

# 5. 3D Pie / Donut Approximation Chart
with st.expander("🍕 5. 3D Pie Chart Projection - Genre Distribution Volume"):
    genre_counts = df['Genre'].value_counts()
    fig_pie = go.Figure(data=[go.Pie(labels=genre_counts.index, values=genre_counts.values, hole=.4)])
    fig_pie.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
    fig_pie.update_layout(title='Categorical Market Share')
    st.plotly_chart(fig_pie, use_container_width=True)

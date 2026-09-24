import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# 1. Page configuration
# -----------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation System")
st.write("Select a movie and get similar movie recommendations.")


# -----------------------------------
# 2. Load movies and ratings
# -----------------------------------

@st.cache_data
def load_data():

    df1 = pd.read_csv(
        "movies.dat",
        sep="::",
        names=["Movie_id", "Title", "Generes"],
        encoding="ISO-8859-1",
        engine="python"
    )

    df2 = pd.read_csv(
        "ratings.dat",
        sep="::",
        names=["User_id", "Movie_id", "rating", "Time_stamp"],
        encoding="ISO-8859-1",
        engine="python"
    )

    return df1, df2


df1, df2 = load_data()


# -----------------------------------
# 3. Create user-movie matrix
# -----------------------------------

@st.cache_data
def create_matrix(df2):

    user_movie_matrix = df2.pivot_table(
        index="User_id",
        columns="Movie_id",
        values="rating"
    )

    return user_movie_matrix.fillna(0)


user_movie_matrix_filled = create_matrix(df2)


# -----------------------------------
# 4. Calculate movie similarity
# -----------------------------------

@st.cache_data
def calculate_similarity(matrix):

    movie_similarity = cosine_similarity(matrix.T)

    return movie_similarity


movie_similarity = calculate_similarity(user_movie_matrix_filled)


# -----------------------------------
# 5. Recommendation function
# -----------------------------------

def recommend_movies(title, n=5):

    movie_id = df1[df1["Title"] == title]["Movie_id"].values[0]

    movie_position = user_movie_matrix_filled.columns.get_loc(movie_id)

    similarity_scores = movie_similarity[movie_position]

    sorted_positions = similarity_scores.argsort()[::-1]

    top_positions = sorted_positions[1:n+1]

    recommended_movie_ids = user_movie_matrix_filled.columns[top_positions]

    recommended_movies = df1[
        df1["Movie_id"].isin(recommended_movie_ids)
    ]

    return recommended_movies[["Movie_id", "Title"]]


# -----------------------------------
# 6. Movie selection
# -----------------------------------

movie_titles = df1["Title"].sort_values().tolist()

selected_movie = st.selectbox(
    "Choose a movie:",
    movie_titles
)


# -----------------------------------
# 7. Recommendation button
# -----------------------------------

if st.button("Recommend Movies"):

    recommendations = recommend_movies(
        selected_movie,
        n=5
    )

    st.subheader("🍿 Recommended Movies")

    for i, title in enumerate(recommendations["Title"], start=1):
        st.write(f"**{i}. {title}**")
"""
Simple Movie Recommendation System
Built for CodeOrbit Tech Artificial Intelligence Internship - Task 4

Approach:
- Content-Based Filtering
- Feature Representation: TF-IDF Vectorizer
- Similarity Metric: Cosine Similarity
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_dataset(file_name="movies.csv"):
    """
    Loads the movie dataset from the CSV file.
    Combines genre and description to form text features.
    """
    # Look for movies.csv in current directory or relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        file_name,
        os.path.join(script_dir, file_name),
        os.path.join(script_dir, "movies.csv"),
        os.path.join(os.getcwd(), file_name),
        os.path.join(os.getcwd(), "movie-recommendation-system", file_name)
    ]

    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break

    if not dataset_path:
        raise FileNotFoundError(f"Could not find '{file_name}'. Please ensure movies.csv is in the project folder.")

    df = pd.read_csv(dataset_path)

    # Fill any missing values with empty string
    df['genre'] = df['genre'].fillna('')
    df['description'] = df['description'].fillna('')

    # Combine genre and description into a single text feature for similarity calculation
    df['combined_features'] = df['genre'] + " " + df['description']
    return df


def calculate_similarity(df):
    """
    Computes the cosine similarity matrix between all movies using TF-IDF.
    """
    # Step 1: Initialize TF-IDF Vectorizer (removes common English stop words)
    tfidf = TfidfVectorizer(stop_words='english')

    # Step 2: Convert combined movie text features into a TF-IDF numerical matrix
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    # Step 3: Compute pairwise Cosine Similarity between all movie vectors
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim


def get_recommendations(movie_title, df, cosine_sim, top_n=5):
    """
    Returns the top N most similar movies for a given movie title.
    """
    # Case-insensitive title lookup
    matched_indices = df[df['title'].str.lower() == movie_title.strip().lower()].index

    if len(matched_indices) == 0:
        return None

    movie_index = matched_indices[0]

    # Get the similarity scores of all movies with the selected movie
    sim_scores = list(enumerate(cosine_sim[movie_index]))

    # Sort movies based on similarity score in descending order
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Exclude the selected movie itself (at index 0) and take top_n items
    top_indices_with_scores = sorted_scores[1:top_n + 1]

    # Collect recommended movie details
    recommendations = []
    for idx, score in top_indices_with_scores:
        recommendations.append({
            "title": df.loc[idx, 'title'],
            "genre": df.loc[idx, 'genre'],
            "similarity_score": round(score * 100, 2),
            "description": df.loc[idx, 'description']
        })

    return recommendations


def display_movie_list(df):
    """
    Displays the catalog of available movies in the dataset.
    """
    print("\n" + "=" * 55)
    print("           AVAILABLE MOVIES IN DATASET")
    print("=" * 55)
    for i, title in enumerate(df['title'], start=1):
        genre = df.loc[df['title'] == title, 'genre'].values[0]
        print(f"  [{i:2d}] {title:<25} ({genre})")
    print("=" * 55)


def main():
    print("=" * 60)
    print("  CODEORBIT TECH AI INTERNSHIP - TASK 4")
    print("  SIMPLE MOVIE RECOMMENDATION SYSTEM")
    print("=" * 60)
    print("Algorithm: Content-Based Filtering (TF-IDF + Cosine Similarity)")

    try:
        df = load_dataset("movies.csv")
    except Exception as e:
        print(f"\n[Error] {e}")
        return

    print(f"\n[Success] Loaded {len(df)} movies from dataset successfully.")

    # Compute similarity matrix once
    cosine_sim = calculate_similarity(df)

    while True:
        display_movie_list(df)
        print("\nEnter a movie name or number from the list above.")
        print("Type 'exit' to quit.")
        user_input = input("\nYour choice: ").strip()

        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using the Movie Recommendation System! Goodbye.\n")
            break

        # Check if user entered a number corresponding to movie index
        selected_title = None
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(df):
                selected_title = df.loc[idx, 'title']
            else:
                print(f"\n[Warning] Invalid movie number. Please enter a number between 1 and {len(df)}.")
                continue
        else:
            selected_title = user_input

        # Fetch recommendations
        recommendations = get_recommendations(selected_title, df, cosine_sim, top_n=5)

        if recommendations is None:
            print(f"\n[Error] Movie '{user_input}' was not found in the dataset.")
            print("Please make sure the title matches one of the movies listed above.")
            continue

        # Display Top 5 Recommendations
        print("\n" + "-" * 60)
        print(f" Top 5 Movies Recommended for: '{selected_title.upper()}'")
        print("-" * 60)
        for rank, rec in enumerate(recommendations, start=1):
            print(f" {rank}. {rec['title']} (Similarity: {rec['similarity_score']}%)")
            print(f"    Genre: {rec['genre']}")
            print(f"    Plot:  {rec['description']}")
            print()
        print("-" * 60)


if __name__ == "__main__":
    main()

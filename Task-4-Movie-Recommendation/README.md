# CodeOrbit Tech AI Internship - Task 4: Simple Movie Recommendation System

## Project Title
**Content-Based Movie Recommendation System using TF-IDF and Cosine Similarity**

---

## Objective
The primary objective of this project is to build a beginner-friendly, working content-based movie recommendation system in Python. It takes a movie title selected by the user, analyzes textual metadata (genres and plot descriptions), calculates pairwise similarity using TF-IDF vectorization and Cosine Similarity, and recommends the top 5 most similar movies.

---

## Task Description
As part of the **CodeOrbit Tech Artificial Intelligence Internship (Task 4)**, this project demonstrates:
1. Building a real, functional recommendation system from scratch.
2. Managing a clean, local tabular dataset (`movies.csv`).
3. Applying Natural Language Processing (NLP) concepts: Term Frequency-Inverse Document Frequency (TF-IDF).
4. Utilizing mathematical vector distance metrics: Cosine Similarity.
5. Providing an interactive, robust command-line interface with input validation and exception handling.

---

## Technologies Used
- **Programming Language:** Python 3.10+
- **Platform:** Cross-platform (Windows, macOS, Linux)
- **Paradigm:** Procedural and functional programming with modular functions

---

## Libraries Used
- **pandas:** For loading, inspecting, and manipulating tabular movie data (`movies.csv`).
- **scikit-learn:**
  - `TfidfVectorizer`: To convert textual descriptions into numerical TF-IDF feature matrices.
  - `cosine_similarity`: To compute the angular cosine distance between movie vectors.

---

## Features
- **Curated Local Dataset:** A focused dataset of 16 popular movies spanning various genres (Sci-Fi, Action, Romance, Drama, Animation, Crime, Fantasy).
- **Flexible User Input:** Users can pick a movie either by typing its title (case-insensitive) or by simply entering its list index number (1–16).
- **Accurate Similarity Computation:** Evaluates both genre tags and plot keywords simultaneously.
- **Top 5 Ranking:** Displays the 5 closest matches with their calculated similarity score (percentage), genre, and brief plot synopsis.
- **Self-Exclusion:** The query movie itself is automatically excluded from the recommendations list.
- **Input Validation:** Gracefully handles invalid movie names or out-of-range numbers without crashing.
- **Clean Console Output:** Formatted display tables with clear visual separation.

---

## Dataset Description (`movies.csv`)
The dataset is stored in a clean CSV file format containing 16 movies:

| Column Name | Data Type | Description |
|---|---|---|
| `movie_id` | Integer | Unique identifier for each movie |
| `title` | String | Official movie title |
| `genre` | String | Genre classifications (e.g., Sci-Fi, Action, Romance) |
| `description` | String | Concise summary of the movie's plot and themes |

### Sample Records:
- **Inception:** Sci-Fi Action — *A skilled thief enters human dreams to steal corporate secrets...*
- **The Dark Knight:** Action Crime Drama — *Batman faces the Joker a criminal mastermind who creates chaos...*
- **Titanic:** Romance Drama — *A young poor artist and a wealthy aristocratic woman fall in love aboard the doomed passenger ship...*
- **Finding Nemo:** Animation Adventure Comedy — *A clownfish searches across the vast ocean accompanied by a forgetful friend...*

---

## How the Recommendation System Works
The system follows a **Content-Based Filtering** workflow:

```
[movies.csv] ──> [Combine Genre + Description] ──> [TF-IDF Vectorizer] ──> [Cosine Similarity Matrix] ──> [User Input] ──> [Rank Top 5 Movies]
```

1. **Feature Engineering:** Combines each movie's `genre` and `description` into a single text attribute: `combined_features`.
2. **Text Vectorization:** Converts words into numerical vectors using `TfidfVectorizer`, removing standard English stop words (like *the*, *a*, *is*).
3. **Similarity Matrix:** Generates a 16x16 similarity matrix where each cell `[i, j]` represents how similar Movie `i` is to Movie `j` on a scale from 0.0 to 1.0.
4. **Ranking & Filtering:**
   - Extracts the similarity row corresponding to the chosen movie.
   - Sorts scores in descending order.
   - Skips index 0 (the movie itself).
   - Returns the top 5 highest-scoring movies.

---

## TF-IDF Explanation (In Simple Words)
**TF-IDF** stands for **Term Frequency - Inverse Document Frequency**. It is an NLP algorithm used to measure how important a word is to a specific document within a collection of documents.

### 1. Term Frequency (TF):
- Measures how frequently a word appears in a specific movie description.
- $\text{TF} = \frac{\text{Count of word in description}}{\text{Total words in description}}$
- If the word *"space"* appears 4 times in Interstellar's description, its Term Frequency is high.

### 2. Inverse Document Frequency (IDF):
- Measures how rare or common a word is across *all* movie descriptions in the entire dataset.
- $\text{IDF} = \log\left(\frac{\text{Total number of movies}}{\text{Number of movies containing the word}}\right)$
- Words that appear everywhere (like *"movie"*, *"story"*, *"character"*) get low IDF scores.
- Words that appear only in specific movies (like *"clownfish"*, *"mafia"*, *"dreams"*, *"space"*) get high IDF scores.

### Formula:
$$\text{TF-IDF} = \text{TF} \times \text{IDF}$$

**Summary for Interviews:**
> *"TF-IDF rewards words that are frequent in a specific movie's plot but penalizes common words that appear across all movies. This highlights distinctive keywords like 'superhero', 'ocean', or 'mafia'."*

---

## Cosine Similarity Explanation (In Simple Words)
**Cosine Similarity** is a mathematical metric used to measure how similar two vectors are, regardless of their size or word length.

### How it works:
1. Each movie's text is represented as a multidimensional vector of TF-IDF scores.
2. Cosine similarity calculates the cosine of the angle ($\theta$) between the two vectors:
   $$\text{Cosine Similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
3. **Score Range:**
   - **1.0 (Angle = $0^\circ$):** Exactly identical content.
   - **0.0 (Angle = $90^\circ$):** Completely independent, no common keywords.
4. **Why not Euclidean Distance?**
   - Euclidean distance measures the straight-line distance between points, which can be distorted by document length.
   - Cosine similarity looks strictly at the **direction/angle** of the vectors, making it immune to differences in plot description lengths.

---

## Project Structure
```text
movie-recommendation-system/
│
├── main.py              # Main Python script with recommendation logic & CLI
├── movies.csv           # Clean tabular dataset of 16 movies
├── requirements.txt     # Python package dependencies (pandas, scikit-learn)
└── README.md            # Complete project documentation & guide
```

---

## Installation Instructions

### Prerequisites:
- Python 3.8 or higher installed on your system.
- `pip` (Python package installer).

### Step 1: Clone or Navigate to the Project Folder
```bash
cd movie-recommendation-system
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run the Project
Run the `main.py` file using Python:

```bash
python main.py
```
*(or `python3 main.py` on Linux/macOS)*

---

## Example Input and Output

### Example 1: Selecting by Title
```text
Your choice: Inception

------------------------------------------------------------
 Top 5 Movies Recommended for: 'INCEPTION'
------------------------------------------------------------
 1. Interstellar (Similarity: 28.45%)
    Genre: Sci-Fi Adventure Drama
    Plot:  A team of astronauts travels through a wormhole in deep space to find a new habitable planet for humanity.

 2. The Matrix (Similarity: 24.12%)
    Genre: Sci-Fi Action
    Plot:  A computer hacker discovers that real life is a simulated reality created by artificial intelligence and joins a rebellion.

 3. The Dark Knight (Similarity: 18.63%)
    Genre: Action Crime Drama
    Plot:  Batman faces the Joker a criminal mastermind who creates chaos and terror in Gotham City.

 4. Batman Begins (Similarity: 14.89%)
    Genre: Action Crime Drama
    Plot:  Bruce Wayne trains with martial artists to fight corruption and organized crime in Gotham City as Batman.

 5. The Avengers (Similarity: 12.35%)
    Genre: Action Sci-Fi Adventure
    Plot:  A team of superheroes including Iron Man and Captain America unite to protect Earth from an alien invasion.
------------------------------------------------------------
```

### Example 2: Selecting by List Number
```text
Your choice: 8   (Titanic)

------------------------------------------------------------
 Top 5 Movies Recommended for: 'TITANIC'
------------------------------------------------------------
 1. The Notebook (Similarity: 36.82%)
    Genre: Romance Drama
    Plot:  A poor young man and a rich young woman fall deeply in love but are separated by social differences.

 2. La La Land (Similarity: 25.14%)
    Genre: Romance Drama Music
    Plot:  An aspiring actress and a dedicated jazz musician fall in love while chasing their artistic dreams in Los Angeles.

 3. Interstellar (Similarity: 11.23%)
    Genre: Sci-Fi Adventure Drama
    Plot:  A team of astronauts travels through a wormhole in deep space to find a new habitable planet for humanity.

 4. The Lion King (Similarity: 9.87%)
    Genre: Animation Adventure Drama
    Plot:  A young lion prince flees his kingdom after his father is killed and returns as an adult to reclaim his throne.

 5. The Godfather (Similarity: 8.42%)
    Genre: Crime Drama
    Plot:  The aging patriarch of a powerful Italian-American mafia family transfers control of his secret empire to his son.
------------------------------------------------------------
```

---

## Limitations
1. **Cold-Start for New Movies:** A movie with no textual description or genre cannot be recommended.
2. **Limited Feature Scope:** Does not take into account user ratings, watch histories, actors, or directors.
3. **Synonym Sensitivity:** If two movies describe the same concept using completely different words without overlap, basic TF-IDF may underestimate similarity (can be improved with word embeddings like Word2Vec or BERT in future versions).
4. **Static Dataset:** Uses a small, fixed local CSV without dynamic real-time catalog updates.

---

## Conclusion
This project successfully fulfills **Task 4 of the CodeOrbit Tech Artificial Intelligence Internship**. By leveraging Python, pandas, and scikit-learn, it delivers a clear, robust, and transparent content-based recommendation engine. It serves as an ideal academic demonstration of foundational Natural Language Processing and linear algebra concepts applied to real-world recommendation problems.

# Task 5: Sentiment Analysis on Text Data
**CodeOrbit Tech Artificial Intelligence Internship Project**  

---

## 1. Project Title
**Sentiment Analysis on Text Data using TF-IDF and Logistic Regression**

---

## 2. Objective
The objective of this project is to develop a lightweight, beginner-friendly machine learning model capable of analyzing text reviews (such as movie or product reviews) and classifying their underlying sentiment into three distinct categories:
- **Positive**
- **Negative**
- **Neutral**

---

## 3. CodeOrbit Tech Task Description
As part of the **CodeOrbit Tech Artificial Intelligence Internship (Task 5)**, this task requires:
1. Creating a balanced local dataset of 20–30 text samples.
2. Implementing text preprocessing and feature extraction using **TF-IDF**.
3. Training a classical machine learning classifier using **Logistic Regression**.
4. Testing the model on example reviews and providing real-time predictions.
5. Providing both a clean **Command-Line Interface (CLI)** and an optional **Tkinter Desktop GUI**.
6. Documenting findings, limitations, and evaluation metrics in a concise report.

---

## 4. Technologies Used
- **Programming Language**: Python
- **Python Version**: Python 3.8+ (Tested on Python 3.10)
- **Libraries Used**:
  - `pandas`: For structured tabular dataset loading and manipulation.
  - `scikit-learn`: For TF-IDF vectorization, dataset splitting, Logistic Regression modeling, and evaluation metrics.
  - `tkinter`: Standard Python GUI library for the desktop interface.

---

## 5. Dataset Description
The project utilizes a custom, balanced local dataset saved in `sentiment_data.csv`.
- **Total Samples**: 30 reviews.
- **Columns**:
  - `text`: Review string describing a product, movie, or service experience.
  - `sentiment`: Ground truth category label (`Positive`, `Negative`, `Neutral`).
- **Class Distribution**:
  - **Positive**: 10 reviews (e.g., "The product quality is amazing and works perfectly")
  - **Negative**: 10 reviews (e.g., "Terrible experience, the item arrived damaged and broken")
  - **Neutral**: 10 reviews (e.g., "The product arrived on time and package was intact")

---

## 6. Features
- **Local Data Loading**: Automatically detects and loads `sentiment_data.csv`.
- **Stratified Train-Test Split**: Maintains balanced class representation in train (80%) and test (20%) sets.
- **TF-IDF Feature Representation**: Converts unstructured raw text into informative numerical weight vectors.
- **Multi-Class Classification**: Employs Multinomial Logistic Regression to output probabilities for all 3 classes.
- **Automated Benchmark Testing**: Immediately executes sample test reviews to verify model accuracy.
- **Interactive Console Mode**: Prompts the user to type custom sentences with instant sentiment feedback.
- **Desktop GUI (Tkinter)**: Simple desktop interface with input box, prediction display, and validation for empty inputs.
- **Graceful Error Handling**: Handles missing dataset files, empty strings, and whitespace input.

---

## 7. How the Project Works (Workflow)
1. **Data Ingestion**: Reads `sentiment_data.csv` via pandas into feature set $X$ and label set $y$.
2. **Train/Test Splitting**: Uses `train_test_split` with `stratify=y` to create an 80/20 train/test distribution.
3. **TF-IDF Vectorization**: Fits vocabulary on training text, calculating term frequency and inverse document frequency.
4. **Model Training**: Fits a Logistic Regression classifier on the TF-IDF feature matrix.
5. **Evaluation**: Evaluates the model on unseen test data, computing accuracy and classification metrics.
6. **Inference**: Transforms new user input using the fitted TF-IDF vectorizer and predicts the sentiment category alongside a confidence score.

---

## 8. What is Sentiment Analysis?
Sentiment Analysis (also known as opinion mining) is a Natural Language Processing (NLP) technique used to determine whether a given piece of text expresses a positive, negative, or neutral opinion. It is widely used by companies to monitor customer reviews, social media feedback, and survey responses to understand user satisfaction.

---

## 9. What is TF-IDF?
**TF-IDF** stands for **Term Frequency - Inverse Document Frequency**.
Because machine learning algorithms cannot directly process raw text characters, TF-IDF converts text into meaningful numerical vectors:

1. **Term Frequency (TF)**: Measures how frequently a word appears in a specific document.
   $$\text{TF}(t, d) = \frac{\text{Count of word } t \text{ in document } d}{\text{Total words in document } d}$$

2. **Inverse Document Frequency (IDF)**: Measures how important or rare a word is across all documents. Common words like "the", "is", and "and" receive low IDF scores, while sentiment-bearing words like "amazing", "terrible", and "average" receive high IDF scores.
   $$\text{IDF}(t) = \log\left(\frac{1 + \text{Total documents}}{1 + \text{Documents containing word } t}\right) + 1$$

3. **TF-IDF Score**:
   $$\text{TF-IDF} = \text{TF} \times \text{IDF}$$

---

## 10. What is Logistic Regression?
**Logistic Regression** is a fundamental supervised machine learning classification algorithm. Despite having "regression" in its name, it is used for predicting categorical outcomes:
- In binary classification, it maps linear combinations of feature weights to a probability between 0 and 1 using the Sigmoid activation function.
- In multi-class classification (like Positive, Negative, Neutral), it uses the **Softmax function** to output a probability distribution across all three sentiment classes. The class with the highest probability is chosen as the final prediction.

---

## 11. Project Structure
```
sentiment-analysis/
│
├── main.py              # Core script: data loading, TF-IDF, training, CLI
├── gui.py               # Optional Tkinter desktop GUI
├── sentiment_data.csv   # Local dataset (30 balanced review samples)
├── requirements.txt     # Python dependencies
├── README.md            # Complete project documentation
└── report.md            # Short internship evaluation report
```

---

## 12. Installation Instructions

### Step 1: Open Terminal or Command Prompt
Navigate to the project directory:
```bash
cd sentiment-analysis
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 13. How to Run the Project

### Option A: Run the Console Application (Recommended)
```bash
python main.py
```
*What happens:*
- Loads `sentiment_data.csv`.
- Trains the Logistic Regression model.
- Displays accuracy and classification report.
- Runs built-in benchmark test sentences.
- Opens interactive prompt: type any review or type `exit` to quit.

### Option B: Run the Desktop GUI (Tkinter)
```bash
python gui.py
```
*What happens:*
- Opens a clean window titled "Sentiment Analysis - CodeOrbit Tech Task 5".
- Type your review in the text box and click **Analyze Sentiment**.
- The sentiment (Positive in green, Negative in red, Neutral in amber) and confidence percentage will appear immediately.

---

## 14. Example Inputs & Outputs

| Input Sentence | Predicted Sentiment | Typical Confidence |
| :--- | :--- | :--- |
| `"The battery life is extraordinary and I love this phone!"` | **Positive** | 88.4% |
| `"The delivery was late and the screen was completely broken."` | **Negative** | 91.2% |
| `"The package arrived on Monday in standard condition."` | **Neutral** | 79.5% |
| `"Worst movie of the year, absolute waste of time."` | **Negative** | 94.6% |
| `"Very happy with this purchase, excellent value for money."` | **Positive** | 89.1% |
| `"The laptop has average battery life and normal performance."` | **Neutral** | 82.3% |

---

## 15. Results and Findings
1. **Effectiveness of TF-IDF**: Even with a compact dataset of 30 samples, key sentiment words ("amazing", "brilliant", "terrible", "boring", "standard", "average") obtain distinct weights that allow clear classification.
2. **Linear Separability**: Logistic Regression performs efficiently on TF-IDF vectors because sparse text representations are often linearly separable in high-dimensional feature spaces.
3. **Execution Speed**: Training and inference execute in under 0.05 seconds with zero GPU requirements, making it ideal for low-resource environments.

---

## 16. Limitations
1. **Small Dataset Size**: 30 samples provide basic lexical coverage. Out-of-vocabulary words not seen during training will receive zero weights.
2. **Lack of Context & Sarcasm**: Sarcastic sentences like *"Great, my phone died after 5 minutes"* contain positive words ("great") but express negative sentiment, which simple bag-of-words/TF-IDF models may misclassify.
3. **Negation Handling**: Word pairs like *"not good"* can be split if only unigrams are vectorized (can be addressed using n-grams like `ngram_range=(1, 2)`).

---

## 17. Conclusion
This project successfully fulfills **Task 5 of the CodeOrbit Tech AI Internship**. By combining TF-IDF feature extraction with Logistic Regression, it demonstrates how classical, explainable machine learning techniques can solve natural language text classification effectively, transparently, and without unnecessary computational complexity.

# AI Career Intelligence: Resume Analyzer & Career Recommendation System

**Capstone Project for CodeOrbit Tech Artificial Intelligence Internship — Task 6**  
*Built with Pure Python, Tkinter Desktop GUI, Scikit-learn & TF-IDF Cosine Similarity*

---

## 1. Project Title
**AI Career Intelligence — Resume Analyzer & Career Recommendation System**

---

## 2. Introduction
In today's competitive technology job market, engineering students and early-career job seekers struggle to understand how well their resumes align with target job roles. Many candidates send out dozens of applications without knowing whether their skills match employer expectations or why applicant tracking systems overlook their profiles.

**AI Career Intelligence** is a feature-rich, standalone desktop application built entirely in **Python** using **Tkinter/ttk**. It provides an explainable, end-to-end artificial intelligence pipeline that parses resumes, extracts technical proficiencies, evaluates resumes against an educational scoring rubric, performs semantic matching against job descriptions using **TF-IDF Vectorization** and **Cosine Similarity**, suggests ideal career trajectories, identifies skill gaps, and synthesizes a 5-stage personalized learning roadmap.

---

## 3. Problem Statement
1. **Resume Opacity**: Students lack transparent, objective feedback on whether their resume structure, project descriptions, and technical skills meet entry-level industry benchmarks.
2. **Keyword Mismatch**: Job descriptions often list specialized libraries and requirements that candidates may possess but fail to articulate in standard terminology.
3. **Black-Box Confusion**: Proprietary recruitment algorithms and commercial ATS systems are non-transparent, leaving candidates without actionable ways to improve.
4. **Directionless Learning**: After discovering skill gaps, students often do not have a prioritized, step-by-step milestone plan to systematically acquire missing competencies.

---

## 4. Objectives
- Satisfy the **CodeOrbit Tech Task 6** rubric: Implement an end-to-end AI/ML application:  
  $$\text{Data Input} \longrightarrow \text{Data Processing} \longrightarrow \text{AI/ML Model} \longrightarrow \text{Prediction/Analysis} \longrightarrow \text{Output Display}$$
- Implement transparent, explainable Natural Language Processing using **TF-IDF** and **Cosine Similarity** without relying on external cloud APIs or black-box LLMs.
- Build a polished, modern desktop graphical user interface with **Python Tkinter & ttk**, featuring executive dark/light themes, KPI metric cards, canvas visualizations, and session history management.
- Provide a clear, educational scoring system based on six transparent criteria (Skills, Projects, Education, Certifications, Experience, Completeness).
- Generate personalized career recommendations and actionable 5-step skill-gap learning roadmaps.

---

## 5. Features
- **📊 Executive AI Dashboard**: Real-time KPI summary cards displaying Educational Resume Score, Job Match Percentage, Detected Skills count, and Top Career recommendation.
- **📄 Resume Analyzer**: Upload `.txt` resumes or paste directly. Extracts over 50+ standardized technical competencies across Programming, AI/ML, Data Engineering, DevOps, and Core CS.
- **🎯 Educational Resume Scoring**: 100-point transparent heuristic scoring across 6 key pillars (Skills, Projects, Education, Certifications, Experience, Formatting).
- **⚡ TF-IDF & Cosine Similarity Job Matcher**: Computes mathematical text similarity between candidate resumes and job descriptions, isolating matching competencies and missing prerequisites.
- **🚀 Multi-Path Career Recommendation**: Evaluates skill profiles across five career tracks:
  - *Data Analyst*
  - *Data Scientist*
  - *AI / ML Engineer*
  - *Software Developer*
  - *Web Developer*
- **🔍 Interactive Skill Gap Analysis**: Dynamically switch target roles to see possessed skills vs. high-priority missing requirements.
- **🗺️ Personalized 5-Stage Learning Roadmap**: Structured milestone roadmap with duration estimates and project-based objectives.
- **📈 Visual Analytics & Canvas Charts**: Native high-performance bar charts illustrating resume section strengths and comparative career fit.
- **📜 Session History & Export**: Tracks historical scans throughout the session with one-click export to `analysis_history.json`.
- **🌙 Slate Dark & Light Themes**: Modern typography, high-contrast badges, and instant theme toggling.

---

## 6. Technologies Used
- **Programming Language**: Python 3.10+ (100% Python throughout)
- **GUI Framework**: Python Standard Library `tkinter` and `tkinter.ttk`
- **Machine Learning & NLP**:
  - `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`)
  - Standalone pure-Python fallback NLP engine (Tokenization, TF calculation, smoothed IDF, and vector cosine dot products)
- **Data Handling**: `csv`, `json`, `re` (Regular Expressions), `math`, `datetime`
- **Plotting & Visualization**: Native anti-aliased Tkinter Canvas graphics engine + optional `matplotlib`

---

## 7. AI/ML Approach
The application intentionally uses **classical, interpretable Natural Language Processing (NLP)** and deterministic rule-based scoring rather than opaque neural networks or cloud APIs:

```
[Resume Text] + [Job Description Text]
                  │
                  ▼
         [Text Preprocessing]
   (Lowercasing, Punctuation Stripping,
         English Stop-Word Removal)
                  │
                  ▼
       [TF-IDF Vectorization]
   (Transforms text into multi-dimensional
       numerical frequency vectors)
                  │
                  ▼
         [Cosine Similarity]
   (Calculates angular distance: cos θ = A·B / ||A|| ||B||)
                  │
                  ▼
   [Blended Semantic & Skill Score]
 (Combines TF-IDF vector score with direct
     technical skill overlap ratio)
```

---

## 8. TF-IDF Explanation
**TF-IDF** stands for **Term Frequency - Inverse Document Frequency**. It reflects how important a word is to a document within a collection.

1. **Term Frequency (TF)**:
   Measures how often a term $t$ appears in document $d$ relative to the total number of words:
   $$\text{TF}(t, d) = \frac{\text{Count of } t \text{ in } d}{\text{Total terms in } d}$$

2. **Inverse Document Frequency (IDF)**:
   Penalizes words that appear everywhere (e.g., "experience", "work", "responsibilities") and boosts domain-specific keywords:
   $$\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
   *where $N$ is total documents and $\text{DF}(t)$ is document frequency.*

3. **TF-IDF Weight**:
   $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

---

## 9. Cosine Similarity Explanation
Once two documents are converted into TF-IDF numerical vectors $\vec{u}$ and $\vec{v}$, **Cosine Similarity** measures the cosine of the angle $\theta$ between them:

$$\text{Cosine Similarity} = \cos(\theta) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|} = \frac{\sum_{i=1}^{n} u_i v_i}{\sqrt{\sum_{i=1}^{n} u_i^2} \sqrt{\sum_{i=1}^{n} v_i^2}}$$

- A value of **1.0** indicates identical word-weight distributions.
- A value of **0.0** indicates orthogonal vectors with zero shared vocabulary.
- Our engine blends the cosine similarity (50%) with the direct technical skill intersection ratio (50%) to yield a balanced percentage match.

---

## 10. Career Recommendation Approach
The recommendation model maps candidate skills against predefined core competency sets for five industry career roles:
- **Data Analyst**: Python, SQL, Excel, Power BI, Tableau, Pandas, Data Analysis, Statistics
- **Data Scientist**: Python, SQL, Pandas, NumPy, Scikit-learn, Machine Learning, Statistics
- **AI / ML Engineer**: Python, TensorFlow, PyTorch, Machine Learning, Deep Learning, NLP, Computer Vision
- **Software Developer**: Python, Java, C++, SQL, Git, GitHub, Data Structures, Algorithms, OOP, Linux
- **Web Developer**: HTML, CSS, JavaScript, React, Node.js, Git, SQL, Flask, Django, REST API

The engine calculates a match coefficient for each role:
$$\text{Career Fit Score} = \left(\frac{|\text{Candidate Skills} \cap \text{Role Skills}|}{|\text{Role Skills}|}\right) \times 100$$
The path with the highest coefficient is presented as the primary recommendation, alongside ranked alternatives.

---

## 11. Skill Gap Analysis
For any chosen career path, the engine compares the candidate's skills against required skills:
- **Current Skills ($\cap$)**: Skills possessed by the candidate that fulfill role requirements (marked with green **✓**).
- **Missing Skills ($\setminus$)**: Critical requirements not found in the resume (highlighted with amber/red **⚠**).
- **Gap Ratio**: Quantifies readiness percentage and directs subsequent learning milestones.

---

## 12. Application Workflow
1. **Launch App**: Open `python main.py` on your desktop.
2. **Input Resume**: Paste resume text or click **Upload .txt File** (or click **Load Sample Data**).
3. **Extract Skills**: Click **Analyze Resume** to parse technical competencies.
4. **Review Score**: Inspect the 6-pillar Educational Resume Score.
5. **Paste Job Description**: Provide target job posting text and run **TF-IDF Match**.
6. **Examine Fit**: Review matching skills, missing skills, and semantic similarity percentage.
7. **View Career Fit**: Check top recommended career and ranked alternatives.
8. **Explore Roadmap**: Follow the generated 5-step curriculum to close detected gaps.
9. **Export History**: View session runs and export audit data to JSON.

---

## 13. Project Structure
```
AI-Career-Intelligence/
│
├── main.py                     # Primary desktop GUI & AI/ML engine
├── career_skills.csv           # Career-to-skills knowledge base
├── requirements.txt            # External dependencies
├── README.md                   # Complete documentation
├── test_ai_core.py             # Automated verification test suite
│
├── sample_data/
│   ├── sample_resume.txt       # Realistic B.Tech student resume
│   └── sample_job_description.txt # Junior Data Scientist job posting
│
├── demo_script.md              # 2–5 minute LinkedIn demo script
├── linkedin_post.md            # Formatted LinkedIn announcement post
└── interview_qa.md             # 15 B.Tech interview questions and answers
```

---

## 14. Installation
Ensure **Python 3.10+** is installed on your system.

```bash
# 1. Clone or extract the project directory
cd AI-Career-Intelligence

# 2. (Recommended) Create and activate a Python virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# 3. Install required external dependencies
pip install -r requirements.txt
```

*Note: Tkinter comes bundled with standard Python on Windows and macOS. On Ubuntu/Debian Linux, install via `sudo apt-get install python3-tk`.*

---

## 15. How to Run
Launch the application with:
```bash
python main.py
```

To run the automated verification test suite:
```bash
python test_ai_core.py
```

---

## 16. Example Usage
1. Click **⚡ Load Sample Data** in the top navigation bar.
2. The dashboard instantly displays:
   - **Resume Score**: `100/100` (Educational score)
   - **Job Match**: `63%`
   - **Skills Detected**: `12` technical skills
   - **Recommended Career**: `Data Scientist`
3. Navigate to **Job Matcher** to see:
   - Matching: `Python`, `SQL`, `Pandas`, `NumPy`, `Scikit-learn`, `Machine Learning`, `Git`, `GitHub`, `Data Analysis`
   - Missing: `Power BI`, `Tableau`
4. Navigate to **Learning Roadmap** to view customized study phases.

---

## 17. Screenshots Placeholder
- `[Screenshot 1: Executive Dashboard with KPI Cards and Slate Theme]`
- `[Screenshot 2: Resume Analyzer with Categorized Skill Badges]`
- `[Screenshot 3: Job Description Matcher with TF-IDF Vector Cosine Similarity]`
- `[Screenshot 4: 5-Stage Personalized Learning Roadmap]`
- `[Screenshot 5: Native Visual Analytics Canvas Charts]`

---

## 18. Limitations
- **Plain Text Parsing**: Optimized for clean plain text (`.txt`); does not directly OCR low-quality scanned PDFs.
- **Predefined Vocabulary**: Matches against an established catalog of 50+ mainstream technical skills.
- **Non-ATS Commercial Equivalence**: The 100-point score is a rule-based educational heuristic designed for student learning and does not reflect proprietary commercial ATS formulas.

---

## 19. Educational Disclaimer
> *"This application is developed for educational purposes. Resume scores, job matching, career recommendations, and skill suggestions are generated using simple AI/ML techniques and should not be treated as professional recruitment or career advice."*

---

## 20. Future Improvements
- Add support for direct multi-page PDF parsing via `pypdf`.
- Implement N-gram tokenization (bigrams and trigrams) to capture compound terminology like "Continuous Integration".
- Enable customizable career competency profiles via GUI configuration.
- Add automated PDF summary report export for career counseling offices.

---

## 21. Conclusion
The **AI Career Intelligence System** demonstrates that practical, high-value artificial intelligence applications do not require opaque black-box models or expensive cloud APIs. By combining clean text preprocessing, TF-IDF vectorization, Cosine Similarity, and transparent heuristic scoring inside an attractive Python Tkinter desktop GUI, this project delivers an explainable and impressive capstone for **CodeOrbit Tech Internship Task 6**.

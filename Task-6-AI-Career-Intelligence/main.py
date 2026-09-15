"""
AI Career Intelligence - Resume Analyzer & Career Recommendation System
Developed for CodeOrbit Tech Artificial Intelligence Internship Task 6.

Architecture:
Data Input (Resume & Job Description)
     ↓
Data Processing (Text Normalization, Tokenization, Skill Extraction)
     ↓
AI/ML Model (TF-IDF Vectorization, Cosine Similarity, Rule-based Career Scoring)
     ↓
Prediction / Analysis (Resume Score, Job Fit, Skill Gaps, Career Fit)
     ↓
Output Display (Modern Tkinter Desktop GUI with Charts, Roadmaps, & History)
"""

import os
import re
import math
import json
import csv
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Optional Scikit-learn and Matplotlib integration with seamless native fallbacks
SKLEARN_AVAILABLE = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# ==============================================================================
# 1. CORE AI & ML IMPLEMENTATION (Explainable & Fully Transparent)
# ==============================================================================

class PurePythonNLP:
    """
    Transparent, explainable TF-IDF Vectorizer and Cosine Similarity calculation.
    Ensures 100% reliability on any student laptop even without external libraries.
    """
    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
        "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's",
        "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
        "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
        "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not",
        "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
        "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
        "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
        "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
        "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
        "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
        "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
        "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
        "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
        "you've", "your", "yours", "yourself", "yourselves", "will", "shall", "work", "role"
    }

    @classmethod
    def tokenize(cls, text):
        """Lowercase, remove punctuation, and filter English stopwords."""
        clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        tokens = [token for token in clean.split() if len(token) > 2 and token not in cls.STOP_WORDS]
        return tokens

    @classmethod
    def compute_tfidf_similarity(cls, text1, text2):
        """
        Computes TF-IDF vectors for two documents and calculates their Cosine Similarity.
        Math: Cosine Similarity = (A . B) / (||A|| * ||B||)
        """
        if SKLEARN_AVAILABLE:
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform([text1, text2])
                sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                return float(sim)
            except Exception:
                pass

        # Native transparent mathematical implementation
        tokens1 = cls.tokenize(text1)
        tokens2 = cls.tokenize(text2)

        if not tokens1 or not tokens2:
            return 0.0

        vocab = sorted(list(set(tokens1 + tokens2)))
        docs = [tokens1, tokens2]
        num_docs = 2

        # 1. Compute Document Frequency (DF) for IDF calculation
        df = {}
        for term in vocab:
            count = sum(1 for doc in docs if term in doc)
            df[term] = count

        # 2. Compute TF-IDF vectors
        vectors = []
        for doc in docs:
            doc_len = len(doc)
            tf_counts = {}
            for term in doc:
                tf_counts[term] = tf_counts.get(term, 0) + 1

            vector = []
            for term in vocab:
                tf = tf_counts.get(term, 0) / doc_len
                # Smooth IDF formula: log((1 + N)/(1 + DF)) + 1
                idf = math.log((1 + num_docs) / (1 + df[term])) + 1.0
                vector.append(tf * idf)
            vectors.append(vector)

        # 3. Calculate Cosine Similarity
        v1 = vectors[0]
        v2 = vectors[1]
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return min(max(similarity, 0.0), 1.0)


class CareerIntelligenceEngine:
    """
    Central AI engine responsible for:
    1. Predefined technical skill catalog
    2. Regular expression skill extraction
    3. Educational resume scoring across 6 key pillars
    4. TF-IDF + Cosine similarity job matching
    5. Rule-based career path recommendation
    6. Skill gap & learning roadmap synthesis
    """

    # Comprehensive skill catalog categorized by technical domain
    TECHNICAL_SKILLS = {
        "Programming Languages": [
            "Python", "Java", "C++", "C", "C#", "SQL", "JavaScript", "TypeScript",
            "HTML", "CSS", "R", "Go", "Rust", "PHP"
        ],
        "Libraries & Frameworks": [
            "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Seaborn",
            "TensorFlow", "PyTorch", "Keras", "Flask", "Django", "React",
            "Node.js", "Express", "Tailwind"
        ],
        "AI, ML & Data": [
            "Machine Learning", "Deep Learning", "NLP", "Natural Language Processing",
            "Computer Vision", "Data Analysis", "Data Visualization", "Statistics",
            "Big Data", "Predictive Modeling", "Supervised Learning", "Unsupervised Learning"
        ],
        "Tools, Cloud & DevOps": [
            "Git", "GitHub", "Excel", "Power BI", "Tableau", "Docker",
            "Linux", "AWS", "Azure", "GCP", "PostgreSQL", "MySQL", "MongoDB",
            "VS Code", "Jupyter", "REST API"
        ],
        "Core Computer Science": [
            "Data Structures", "Algorithms", "OOP", "Object Oriented Programming",
            "Database Management", "System Design", "Problem Solving"
        ]
    }

    # Flattened list for quick lookup
    ALL_SKILLS = []
    for cat, sk_list in TECHNICAL_SKILLS.items():
        for sk in sk_list:
            if sk not in ALL_SKILLS:
                ALL_SKILLS.append(sk)

    DEFAULT_CAREER_MAP = {
        "Data Analyst": {
            "skills": ["Python", "SQL", "Excel", "Power BI", "Tableau", "Pandas", "Data Analysis", "Data Visualization", "Statistics"],
            "desc": "Analyzes business metrics, extracts insights from databases, designs dashboards, and tells stories with data."
        },
        "Data Scientist": {
            "skills": ["Python", "SQL", "Pandas", "NumPy", "Scikit-learn", "Machine Learning", "Statistics", "Data Visualization", "Data Analysis"],
            "desc": "Builds statistical and predictive machine learning models, performs exploratory analysis, and solves predictive challenges."
        },
        "AI / ML Engineer": {
            "skills": ["Python", "TensorFlow", "PyTorch", "Machine Learning", "Deep Learning", "NLP", "Scikit-learn", "Pandas", "NumPy", "Computer Vision"],
            "desc": "Designs, trains, and deploys deep neural networks, large NLP models, and computer vision systems for intelligent products."
        },
        "Software Developer": {
            "skills": ["Python", "Java", "C++", "SQL", "Git", "GitHub", "Data Structures", "Algorithms", "OOP", "Linux"],
            "desc": "Engineers scalable applications, implements clean modular architectures, designs algorithms, and manages code repositories."
        },
        "Web Developer": {
            "skills": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Git", "GitHub", "SQL", "Flask", "Django", "REST API"],
            "desc": "Creates modern responsive user interfaces, connects server-side backends, designs RESTful endpoints, and deploys web apps."
        }
    }

    def __init__(self, csv_path="career_skills.csv"):
        self.career_map = self.load_career_skills(csv_path)

    def load_career_skills(self, csv_path):
        """Loads career skill definitions from CSV if available, else uses defaults."""
        mapping = {}
        if os.path.exists(csv_path):
            try:
                with open(csv_path, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        career = row.get("career", "").strip()
                        skills = [s.strip() for s in row.get("required_skills", "").split(",") if s.strip()]
                        desc = row.get("description", "").strip()
                        if career and skills:
                            mapping[career] = {"skills": skills, "desc": desc}
                if mapping:
                    return mapping
            except Exception:
                pass
        return self.DEFAULT_CAREER_MAP

    def extract_skills(self, text):
        """
        Extracts technical skills using regex word-boundary matching.
        Preserves special characters in names like C++, C#, Node.js.
        """
        detected = []
        lower_text = text.lower()

        for skill in self.ALL_SKILLS:
            # Special regex for skills with symbols
            if skill == "C++":
                pattern = r'(?:\b|(?<=[^a-zA-Z0-9]))c\+\+(?:\b|(?=[^a-zA-Z0-9]))'
            elif skill == "C#":
                pattern = r'(?:\b|(?<=[^a-zA-Z0-9]))c\#(?:\b|(?=[^a-zA-Z0-9]))'
            elif skill == "C":
                pattern = r'\b[cC]\b'
            elif skill == "R":
                pattern = r'\b[rR]\b'
            elif skill == ".NET":
                pattern = r'\.net\b'
            elif skill == "Node.js":
                pattern = r'\bnode(?:\.js)?\b'
            else:
                escaped = re.escape(skill.lower())
                pattern = r'\b' + escaped + r'\b'

            if re.search(pattern, lower_text):
                detected.append(skill)

        # Remove redundant single-letter matches if C++ or C# were the actual mentions
        if "C++" in detected and "C" in detected:
            if not re.search(r'\bc language\b|\bprogramming in c\b', lower_text):
                detected.remove("C")

        return sorted(list(set(detected)))

    def calculate_educational_resume_score(self, resume_text, detected_skills):
        """
        Calculates an explainable Educational Resume Score (0-100) based on 6 criteria:
        1. Skills (25 points max)
        2. Projects (20 points max)
        3. Education (15 points max)
        4. Certifications (15 points max)
        5. Experience/Internship (15 points max)
        6. Completeness & Formatting (10 points max)
        """
        text_lower = resume_text.lower()

        # 1. Skills Score (max 25)
        # Scaled based on the variety of recognized technical skills
        num_skills = len(detected_skills)
        if num_skills >= 10:
            skills_score = 25
        elif num_skills >= 7:
            skills_score = 21
        elif num_skills >= 5:
            skills_score = 17
        elif num_skills >= 3:
            skills_score = 12
        elif num_skills >= 1:
            skills_score = 7
        else:
            skills_score = 0

        # 2. Projects Score (max 20)
        proj_keywords = ["project", "developed", "built", "implemented", "system", "application", "model", "github", "pipeline"]
        has_proj_section = bool(re.search(r'\b(projects|academic projects|key projects)\b', text_lower))
        proj_matches = sum(1 for kw in proj_keywords if kw in text_lower)
        projects_score = 0
        if has_proj_section:
            projects_score += 10
        projects_score += min(10, proj_matches * 1.5)
        projects_score = min(20, int(round(projects_score)))

        # 3. Education Score (max 15)
        edu_keywords = ["b.tech", "bachelor", "degree", "university", "institute", "college", "engineering", "cgpa", "gpa", "computer science"]
        edu_matches = sum(1 for kw in edu_keywords if kw in text_lower)
        if edu_matches >= 4:
            education_score = 15
        elif edu_matches >= 2:
            education_score = 12
        elif edu_matches >= 1:
            education_score = 8
        else:
            education_score = 0

        # 4. Certifications Score (max 15)
        cert_keywords = ["certification", "certificate", "certified", "coursera", "udemy", "nptel", "edx", "completion", "achievement"]
        cert_matches = sum(1 for kw in cert_keywords if kw in text_lower)
        if cert_matches >= 3:
            certifications_score = 15
        elif cert_matches >= 2:
            certifications_score = 12
        elif cert_matches >= 1:
            certifications_score = 8
        else:
            certifications_score = 0

        # 5. Experience / Internship Score (max 15)
        exp_keywords = ["intern", "internship", "trainee", "experience", "work experience", "responsibilities", "contributed", "role"]
        exp_matches = sum(1 for kw in exp_keywords if kw in text_lower)
        if exp_matches >= 3:
            experience_score = 15
        elif exp_matches >= 2:
            experience_score = 12
        elif exp_matches >= 1:
            experience_score = 8
        else:
            experience_score = 0

        # 6. Completeness & Formatting (max 10)
        completeness_score = 0
        if re.search(r'[\w\.-]+@[\w\.-]+\.\w+', resume_text): # Email
            completeness_score += 3
        if re.search(r'(\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}', resume_text) or "phone" in text_lower:
            completeness_score += 2
        if "linkedin" in text_lower or "github" in text_lower:
            completeness_score += 2
        word_count = len(resume_text.split())
        if word_count >= 150: # Adequate length
            completeness_score += 3
        elif word_count >= 80:
            completeness_score += 1
        completeness_score = min(10, completeness_score)

        total_score = skills_score + projects_score + education_score + certifications_score + experience_score + completeness_score

        breakdown = {
            "skills": skills_score,
            "projects": projects_score,
            "education": education_score,
            "certifications": certifications_score,
            "experience": experience_score,
            "completeness": completeness_score,
            "total": total_score
        }
        return breakdown

    def match_job_description(self, resume_text, job_desc_text):
        """
        Performs TF-IDF Cosine Similarity and skill overlap analysis.
        Returns: match_percentage, matching_skills, missing_skills, suggested_skills
        """
        if not resume_text.strip() or not job_desc_text.strip():
            return {
                "similarity_score": 0.0,
                "match_percentage": 0,
                "matching_skills": [],
                "missing_skills": [],
                "suggested_skills": []
            }

        # 1. Cosine similarity using TF-IDF
        raw_sim = PurePythonNLP.compute_tfidf_similarity(resume_text, job_desc_text)
        
        # 2. Extract skills from both documents
        resume_skills = set(self.extract_skills(resume_text))
        job_skills = set(self.extract_skills(job_desc_text))

        matching_skills = sorted(list(resume_skills.intersection(job_skills)))
        missing_skills = sorted(list(job_skills.difference(resume_skills)))

        # 3. Blended match percentage: 55% TF-IDF Cosine Similarity + 45% Direct Skill Overlap
        if job_skills:
            skill_ratio = len(matching_skills) / len(job_skills)
        else:
            skill_ratio = raw_sim

        # Realistic calibration for student resumes (avoiding 0% or extreme extremes)
        blended = (raw_sim * 0.50) + (skill_ratio * 0.50)
        match_percentage = int(round(blended * 100))
        match_percentage = min(98, max(5, match_percentage)) if resume_skills else min(40, int(round(raw_sim * 100)))

        # Suggested skills are the priority missing skills from the job posting
        suggested = missing_skills[:4]

        return {
            "similarity_score": round(raw_sim, 3),
            "match_percentage": match_percentage,
            "matching_skills": matching_skills,
            "missing_skills": missing_skills,
            "suggested_skills": suggested
        }

    def recommend_career(self, detected_skills):
        """
        Rule-based recommendation evaluating skill match across defined career categories.
        Returns ranked list of careers with scores and top recommended career.
        """
        user_skills_set = set(detected_skills)
        ranked = []

        for career_name, data in self.career_map.items():
            req_skills = data["skills"]
            req_set = set(req_skills)
            matched = user_skills_set.intersection(req_set)
            
            # Match score as percentage of required skills
            if req_skills:
                score = (len(matched) / len(req_skills)) * 100
            else:
                score = 0.0

            ranked.append({
                "career": career_name,
                "score": round(score, 1),
                "matched_skills": sorted(list(matched)),
                "missing_skills": sorted(list(req_set.difference(user_skills_set))),
                "required_skills": req_skills,
                "description": data["desc"]
            })

        # Sort descending by match score
        ranked.sort(key=lambda x: x["score"], reverse=True)
        top_rec = ranked[0] if ranked else None

        return {
            "top_career": top_rec["career"] if top_rec and top_rec["score"] > 0 else "Software Developer",
            "top_score": top_rec["score"] if top_rec else 0,
            "ranked_careers": ranked
        }

    def analyze_skill_gap(self, detected_skills, target_career):
        """
        Performs detailed gap analysis between detected skills and chosen target career.
        """
        target_info = self.career_map.get(target_career, self.DEFAULT_CAREER_MAP.get(target_career, {}))
        required = target_info.get("skills", [])
        
        user_set = set(detected_skills)
        req_set = set(required)

        current = sorted(list(user_set.intersection(req_set)))
        missing = sorted(list(req_set.difference(user_set)))

        match_pct = int(round((len(current) / len(required) * 100))) if required else 0

        return {
            "career": target_career,
            "description": target_info.get("desc", ""),
            "required_skills": required,
            "current_skills": current,
            "missing_skills": missing,
            "match_percentage": match_pct
        }

    def generate_learning_roadmap(self, target_career, missing_skills):
        """
        Generates an actionable, personalized 5-step learning roadmap based on missing skills.
        """
        steps = []
        
        # Step 1: Foundational core
        if missing_skills:
            step1_skills = missing_skills[:2]
            steps.append({
                "step": 1,
                "title": f"Master Foundations: {', '.join(step1_skills)}",
                "duration": "2 - 3 Weeks",
                "description": f"Focus on syntax, core principles, hands-on tutorials, and official documentation for {', '.join(step1_skills)}."
            })
        else:
            steps.append({
                "step": 1,
                "title": "Review Core Fundamentals",
                "duration": "1 - 2 Weeks",
                "description": "Strengthen fundamental concepts, coding speed, and architectural best practices."
            })

        # Step 2: Advanced tools or next missing skill
        if len(missing_skills) > 2:
            step2_skills = missing_skills[2:4]
            steps.append({
                "step": 2,
                "title": f"Expand Toolset: {', '.join(step2_skills)}",
                "duration": "3 - 4 Weeks",
                "description": f"Learn practical workflows, real-world libraries, and integration patterns for {', '.join(step2_skills)}."
            })
        else:
            steps.append({
                "step": 2,
                "title": "Deepen Domain Libraries & Tooling",
                "duration": "2 - 3 Weeks",
                "description": "Explore production-grade libraries, debugging workflows, and optimization techniques."
            })

        # Step 3: Hands-on Capstone Projects
        steps.append({
            "step": 3,
            "title": f"Build 2 End-to-End {target_career} Projects",
            "duration": "4 - 5 Weeks",
            "description": f"Develop complete, practical applications integrating real-world datasets, APIs, and clean modular code."
        })

        # Step 4: Open Source & Portfolio
        steps.append({
            "step": 4,
            "title": "Publish Clean Repositories to GitHub",
            "duration": "1 - 2 Weeks",
            "description": "Write comprehensive README files, document problem statements and architecture diagrams, and push commits."
        })

        # Step 5: Interview Prep & Mock Assessments
        steps.append({
            "step": 5,
            "title": "Technical Interview Preparation & Networking",
            "duration": "2 - 3 Weeks",
            "description": "Practice domain-specific questions, explain project decisions clearly, and engage with professional tech communities."
        })

        return steps


# ==============================================================================
# 2. MODERN DESKTOP GUI IMPLEMENTATION (Tkinter + ttk)
# ==============================================================================

class ModernTheme:
    """Design System styling constants: Slate Modern Palette."""
    # Dark Theme
    BG_DARK = "#0F172A"       # Deep slate navy
    PANEL_DARK = "#1E293B"    # Slate card background
    HEADER_DARK = "#111827"   # Top bar dark
    SIDEBAR_DARK = "#090D16"  # Left sidebar dark
    TEXT_MAIN_DARK = "#F8FAFC"# Bright white text
    TEXT_MUTED_DARK = "#94A3B8"# Muted gray text
    BORDER_DARK = "#334155"   # Card borders
    ACCENT_CYAN = "#0EA5E9"   # Modern vibrant cyan
    ACCENT_HOVER = "#38BDF8"  # Cyan hover
    SUCCESS_GREEN = "#10B981" # Green badges
    WARNING_AMBER = "#F59E0B" # Amber indicators
    DANGER_RED = "#EF4444"    # Missing skills red
    CARD_HOVER = "#273549"

    # Light Theme
    BG_LIGHT = "#F8FAFC"
    PANEL_LIGHT = "#FFFFFF"
    HEADER_LIGHT = "#FFFFFF"
    SIDEBAR_LIGHT = "#F1F5F9"
    TEXT_MAIN_LIGHT = "#0F172A"
    TEXT_MUTED_LIGHT = "#64748B"
    BORDER_LIGHT = "#E2E8F0"


class AICareerIntelligenceApp(tk.Tk):
    """
    Main Application Window for AI Career Intelligence.
    Features:
    - Sleek left navigation sidebar
    - Header with live model status and sample data loader
    - 9 full-featured interactive views
    - Responsive layout with canvas scrolling
    - Dynamic custom-drawn charts
    - Session analysis history
    """

    def __init__(self):
        super().__init__()

        self.title("AI Career Intelligence — Resume Analyzer & Career Recommendation System")
        self.geometry("1200x820")
        self.minsize(1050, 720)

        # Application state
        self.engine = CareerIntelligenceEngine()
        self.is_dark_theme = True
        self.active_tab = "dashboard"
        self.session_history = []
        self.analysis_cache = None

        # Sample data presets
        self.sample_resume = self.load_sample_file("sample_data/sample_resume.txt")
        self.sample_job_desc = self.load_sample_file("sample_data/sample_job_description.txt")

        # Setup GUI architecture
        self.init_styles()
        self.build_ui_layout()
        self.show_view("dashboard")

        # Auto-run an initial sample analysis so dashboard opens with rich live data
        self.load_sample_data(auto_analyze=True)

    def load_sample_file(self, rel_path):
        """Safely loads text from file or returns standard fallback."""
        if os.path.exists(rel_path):
            try:
                with open(rel_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
        return ""

    def init_styles(self):
        """Initializes ttk theme and custom styles."""
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

    def get_colors(self):
        """Returns active theme color palette."""
        if self.is_dark_theme:
            return {
                "bg": ModernTheme.BG_DARK,
                "panel": ModernTheme.PANEL_DARK,
                "header": ModernTheme.HEADER_DARK,
                "sidebar": ModernTheme.SIDEBAR_DARK,
                "text_main": ModernTheme.TEXT_MAIN_DARK,
                "text_muted": ModernTheme.TEXT_MUTED_DARK,
                "border": ModernTheme.BORDER_DARK,
                "accent": ModernTheme.ACCENT_CYAN,
                "accent_hover": ModernTheme.ACCENT_HOVER,
                "success": ModernTheme.SUCCESS_GREEN,
                "warning": ModernTheme.WARNING_AMBER,
                "danger": ModernTheme.DANGER_RED,
                "card_hover": ModernTheme.CARD_HOVER,
                "input_bg": "#0B1120",
                "active_nav": "#1E293B"
            }
        else:
            return {
                "bg": ModernTheme.BG_LIGHT,
                "panel": ModernTheme.PANEL_LIGHT,
                "header": ModernTheme.HEADER_LIGHT,
                "sidebar": ModernTheme.SIDEBAR_LIGHT,
                "text_main": ModernTheme.TEXT_MAIN_LIGHT,
                "text_muted": ModernTheme.TEXT_MUTED_LIGHT,
                "border": ModernTheme.BORDER_LIGHT,
                "accent": "#0284C7",
                "accent_hover": "#0369A1",
                "success": "#059669",
                "warning": "#D97706",
                "danger": "#DC2626",
                "card_hover": "#F1F5F9",
                "input_bg": "#FFFFFF",
                "active_nav": "#E2E8F0"
            }

    def build_ui_layout(self):
        """Constructs the root window layout: Sidebar + Header + Content Area + Status Bar."""
        c = self.get_colors()
        self.configure(bg=c["bg"])

        # 1. Main container
        self.root_frame = tk.Frame(self, bg=c["bg"])
        self.root_frame.pack(fill="both", expand=True)

        # 2. Left Sidebar
        self.sidebar_frame = tk.Frame(self.root_frame, bg=c["sidebar"], width=240)
        self.sidebar_frame.pack(side="left", fill="y")
        self.sidebar_frame.pack_propagate(False)
        self.build_sidebar()

        # 3. Right Container (Header + Dynamic Content Area + Footer)
        self.main_container = tk.Frame(self.root_frame, bg=c["bg"])
        self.main_container.pack(side="right", fill="both", expand=True)

        self.build_header()

        # 4. Dynamic Content Area with smooth switching
        self.content_viewport = tk.Frame(self.main_container, bg=c["bg"])
        self.content_viewport.pack(side="top", fill="both", expand=True, padx=20, pady=(15, 10))

        # 5. Footer Status Bar
        self.build_footer()

    def build_sidebar(self):
        """Constructs modern sidebar navigation with brand identity and nav items."""
        c = self.get_colors()

        # Brand header
        brand_frame = tk.Frame(self.sidebar_frame, bg=c["sidebar"], padx=16, pady=20)
        brand_frame.pack(fill="x")

        title_lbl = tk.Label(
            brand_frame,
            text="AI CAREER INTEL",
            font=("Segoe UI", 13, "bold"),
            fg=c["accent"],
            bg=c["sidebar"]
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            brand_frame,
            text="CodeOrbit Tech Task 6",
            font=("Segoe UI", 8),
            fg=c["text_muted"],
            bg=c["sidebar"]
        )
        sub_lbl.pack(anchor="w", pady=(2, 0))

        # Divider line
        div = tk.Frame(self.sidebar_frame, bg=c["border"], height=1)
        div.pack(fill="x", padx=16, pady=(0, 15))

        # Navigation buttons dictionary
        self.nav_buttons = {}
        nav_items = [
            ("dashboard", "📊  Overview Dashboard"),
            ("resume_analyzer", "📄  Resume Analyzer"),
            ("resume_score", "🎯  Educational Score"),
            ("job_matcher", "⚡  Job Matcher (TF-IDF)"),
            ("career_rec", "🚀  Career Recommendations"),
            ("skill_gap", "🔍  Skill Gap Analysis"),
            ("roadmap", "🗺️  Learning Roadmap"),
            ("analytics", "📈  Analytics & Charts"),
            ("history", "📜  Analysis History"),
            ("about", "ℹ️  About & AI Workflow")
        ]

        for tab_id, label_text in nav_items:
            btn = tk.Button(
                self.sidebar_frame,
                text=label_text,
                font=("Segoe UI", 10),
                fg=c["text_main"],
                bg=c["sidebar"],
                activebackground=c["active_nav"],
                activeforeground=c["accent"],
                bd=0,
                padx=16,
                pady=10,
                anchor="w",
                cursor="hand2",
                command=lambda tid=tab_id: self.show_view(tid)
            )
            btn.pack(fill="x", pady=2)
            self.nav_buttons[tab_id] = btn

        # Bottom branding in sidebar
        bottom_frame = tk.Frame(self.sidebar_frame, bg=c["sidebar"], padx=16, pady=16)
        bottom_frame.pack(side="bottom", fill="x")

        tag_box = tk.Label(
            bottom_frame,
            text="B.Tech Final Year\nStudent Showcase",
            font=("Segoe UI", 8),
            fg=c["text_muted"],
            bg=c["panel"],
            padx=10,
            pady=8,
            relief="solid",
            bd=1
        )
        tag_box.pack(fill="x")

    def build_header(self):
        """Top bar with title, quick sample loader, and theme toggle."""
        c = self.get_colors()

        self.header_frame = tk.Frame(self.main_container, bg=c["header"], height=60, padx=20)
        self.header_frame.pack(side="top", fill="x")

        # Active page title
        self.header_title = tk.Label(
            self.header_frame,
            text="Executive AI Dashboard",
            font=("Segoe UI", 15, "bold"),
            fg=c["text_main"],
            bg=c["header"],
            pady=16
        )
        self.header_title.pack(side="left")

        # Right-side action controls
        right_ctrls = tk.Frame(self.header_frame, bg=c["header"])
        right_ctrls.pack(side="right", pady=12)

        # Status badge
        self.status_pill = tk.Label(
            right_ctrls,
            text="● AI Engine Ready",
            font=("Segoe UI", 9, "bold"),
            fg=c["success"],
            bg=c["panel"],
            padx=10,
            pady=5,
            bd=1,
            relief="solid"
        )
        self.status_pill.pack(side="left", padx=(0, 10))

        # Quick Load Sample Data Button
        sample_btn = tk.Button(
            right_ctrls,
            text="⚡ Load Sample Data",
            font=("Segoe UI", 9, "bold"),
            bg=c["accent"],
            fg="#FFFFFF",
            activebackground=c["accent_hover"],
            activeforeground="#FFFFFF",
            bd=0,
            padx=12,
            pady=5,
            cursor="hand2",
            command=self.load_sample_data
        )
        sample_btn.pack(side="left", padx=(0, 10))

        # Reset Button
        reset_btn = tk.Button(
            right_ctrls,
            text="🔄 Reset",
            font=("Segoe UI", 9),
            bg=c["panel"],
            fg=c["text_main"],
            bd=1,
            padx=10,
            pady=5,
            cursor="hand2",
            command=self.reset_inputs
        )
        reset_btn.pack(side="left", padx=(0, 10))

        # Theme Toggle
        theme_txt = "☀️ Light" if self.is_dark_theme else "🌙 Dark"
        self.theme_btn = tk.Button(
            right_ctrls,
            text=theme_txt,
            font=("Segoe UI", 9),
            bg=c["panel"],
            fg=c["text_main"],
            bd=1,
            padx=10,
            pady=5,
            cursor="hand2",
            command=self.toggle_theme
        )
        self.theme_btn.pack(side="left")

    def build_footer(self):
        """Bottom status bar with internship attribution and educational disclaimer."""
        c = self.get_colors()
        self.footer_frame = tk.Frame(self.main_container, bg=c["header"], height=32, padx=20)
        self.footer_frame.pack(side="bottom", fill="x")

        disclaimer_text = "Educational Purpose: Uses TF-IDF & Cosine Similarity. Not commercial ATS or hiring advice."
        disclaimer_lbl = tk.Label(
            self.footer_frame,
            text=disclaimer_text,
            font=("Segoe UI", 8),
            fg=c["text_muted"],
            bg=c["header"],
            pady=6
        )
        disclaimer_lbl.pack(side="left")

        attr_lbl = tk.Label(
            self.footer_frame,
            text="CodeOrbit Tech AI Internship | Python + Tkinter",
            font=("Segoe UI", 8),
            fg=c["accent"],
            bg=c["header"]
        )
        attr_lbl.pack(side="right")

    def toggle_theme(self):
        """Toggles between Slate Dark and Professional Light themes."""
        self.is_dark_theme = not self.is_dark_theme
        # Rebuild interface
        for widget in self.root_frame.winfo_children():
            widget.destroy()
        self.root_frame.destroy()
        self.build_ui_layout()
        self.show_view(self.active_tab)

    def show_view(self, view_id):
        """Switches the active content view smoothly."""
        self.active_tab = view_id
        c = self.get_colors()

        # Update sidebar active highlighting
        for tid, btn in self.nav_buttons.items():
            if tid == view_id:
                btn.configure(bg=c["active_nav"], fg=c["accent"], font=("Segoe UI", 10, "bold"))
            else:
                btn.configure(bg=c["sidebar"], fg=c["text_main"], font=("Segoe UI", 10))

        # Clear existing view
        for child in self.content_viewport.winfo_children():
            child.destroy()

        # Map views
        view_map = {
            "dashboard": (self.render_dashboard_view, "Executive AI Dashboard"),
            "resume_analyzer": (self.render_resume_analyzer_view, "Resume Analyzer & Skill Extraction"),
            "resume_score": (self.render_resume_score_view, "Educational Resume Score Breakdown"),
            "job_matcher": (self.render_job_matcher_view, "Job Description Matcher (TF-IDF & Cosine)"),
            "career_rec": (self.render_career_recommendation_view, "Career Path Recommendations"),
            "skill_gap": (self.render_skill_gap_view, "Target Career Skill Gap Analysis"),
            "roadmap": (self.render_roadmap_view, "Personalized AI Learning Roadmap"),
            "analytics": (self.render_analytics_view, "Visual Analytics & Metric Distribution"),
            "history": (self.render_history_view, "Session Analysis History"),
            "about": (self.render_about_view, "AI System Architecture & Internship Info")
        }

        render_func, title_text = view_map.get(view_id, (self.render_dashboard_view, "Dashboard"))
        self.header_title.configure(text=title_text)
        render_func()

    # --------------------------------------------------------------------------
    # DATA & ANALYSIS MANAGEMENT
    # --------------------------------------------------------------------------

    def perform_full_analysis(self, resume_text, job_desc_text):
        """
        Executes end-to-end AI/ML pipeline:
        1. Skill Extraction
        2. Educational Resume Scoring
        3. TF-IDF Cosine Similarity Job Matching
        4. Career Path Recommendation
        5. Skill Gap Analysis
        6. Personalized Learning Roadmap
        """
        if not resume_text.strip():
            return None

        # 1. Skill Extraction
        detected_skills = self.engine.extract_skills(resume_text)

        # 2. Educational Resume Score
        resume_score = self.engine.calculate_educational_resume_score(resume_text, detected_skills)

        # 3. Job Matcher
        job_match = self.engine.match_job_description(resume_text, job_desc_text)

        # 4. Career Recommendation
        career_rec = self.engine.recommend_career(detected_skills)
        top_career = career_rec["top_career"]

        # 5. Skill Gap Analysis
        skill_gap = self.engine.analyze_skill_gap(detected_skills, top_career)

        # 6. Learning Roadmap
        roadmap = self.engine.generate_learning_roadmap(top_career, skill_gap["missing_skills"])

        analysis = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "resume_text": resume_text,
            "job_desc_text": job_desc_text,
            "detected_skills": detected_skills,
            "resume_score": resume_score,
            "job_match": job_match,
            "career_rec": career_rec,
            "top_career": top_career,
            "skill_gap": skill_gap,
            "roadmap": roadmap
        }

        self.analysis_cache = analysis
        self.session_history.append(analysis)

        # Update status pill
        c = self.get_colors()
        self.status_pill.configure(text=f"● Analyzed ({len(detected_skills)} Skills Detected)", fg=c["success"])

        return analysis

    def load_sample_data(self, auto_analyze=True):
        """Populates realistic B.Tech sample resume and job description."""
        # Use bundled sample files or built-in comprehensive student dataset
        if not self.sample_resume:
            self.sample_resume = (
                "Aarav Sharma | aarav.sharma@email.com | +91 98765 43210 | Bangalore, India\n"
                "LinkedIn: linkedin.com/in/aarav-sharma | GitHub: github.com/aarav-tech\n\n"
                "EDUCATION\n"
                "B.Tech in Computer Science & Engineering | Apex Institute of Technology (2021-2025)\n"
                "CGPA: 8.6 / 10.0 | Relevant Coursework: Data Structures, Algorithms, DBMS, OOP, Machine Learning, Statistics\n\n"
                "TECHNICAL SKILLS\n"
                "- Programming: Python, SQL, C++, Java\n"
                "- Data & AI: Pandas, NumPy, Scikit-learn, Machine Learning, Data Analysis, Data Visualization, Matplotlib\n"
                "- Tools & Systems: Git, GitHub, Linux, VS Code, Jupyter Notebook\n\n"
                "PROJECTS\n"
                "1. Customer Churn Prediction System (Scikit-learn, Pandas, Python)\n"
                "- Developed supervised classification pipeline predicting customer attrition with 88% accuracy.\n"
                "- Engineered features from 15,000 transaction records using Pandas and NumPy.\n"
                "2. Sales Performance Analysis Dashboard (Python, SQL, Matplotlib)\n"
                "- Formulated complex SQL queries to extract regional revenue trends and visual reports.\n\n"
                "WORK EXPERIENCE / INTERNSHIPS\n"
                "Artificial Intelligence Intern | CodeOrbit Tech (June 2024 - Aug 2024)\n"
                "- Built data cleaning scripts and automated evaluation pipelines with Scikit-learn.\n"
                "- Collaborated on GitHub version control and participated in Agile sprints.\n\n"
                "CERTIFICATIONS\n"
                "- Coursera: Machine Learning Specialization (DeepLearning.AI)\n"
                "- NPTEL: Data Science with Python (Elite + Silver)"
            )

        if not self.sample_job_desc:
            self.sample_job_desc = (
                "Role: Junior Data Scientist / Data Analyst\n"
                "Company: NexaTech Analytics | Location: Bangalore / Hybrid\n\n"
                "Requirements:\n"
                "- Bachelor's degree in Computer Science, Data Science, or related field (B.Tech / B.E.).\n"
                "- Strong proficiency in Python and SQL for data manipulation.\n"
                "- Hands-on experience with Pandas, NumPy, and Scikit-learn for machine learning.\n"
                "- Experience with Power BI or Tableau for business intelligence dashboards.\n"
                "- Solid understanding of Statistics, exploratory data analysis, and predictive modeling.\n"
                "- Familiarity with Git and GitHub collaborative code repositories."
            )

        if auto_analyze:
            self.perform_full_analysis(self.sample_resume, self.sample_job_desc)

    def reset_inputs(self):
        """Clears current active analysis."""
        confirm = messagebox.askyesno("Reset Analysis", "Are you sure you want to reset the current analysis inputs?")
        if confirm:
            self.sample_resume = ""
            self.sample_job_desc = ""
            self.analysis_cache = None
            c = self.get_colors()
            self.status_pill.configure(text="● Ready (Cleared)", fg=c["text_muted"])
            self.show_view(self.active_tab)

    # --------------------------------------------------------------------------
    # VIEW 1: EXECUTIVE DASHBOARD
    # --------------------------------------------------------------------------

    def render_dashboard_view(self):
        """Renders the executive AI dashboard with KPI cards, career recommendation, and quick actions."""
        c = self.get_colors()
        data = self.analysis_cache

        # Scrollable container
        canvas = tk.Canvas(self.content_viewport, bg=c["bg"], bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.content_viewport, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=c["bg"])

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=880)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Welcome banner card
        welcome_frame = tk.Frame(scrollable_frame, bg=c["panel"], bd=1, relief="solid", padx=20, pady=16)
        welcome_frame.pack(fill="x", pady=(0, 15))

        w_title = tk.Label(
            welcome_frame,
            text="Welcome to AI Career Intelligence 👋",
            font=("Segoe UI", 14, "bold"),
            fg=c["text_main"],
            bg=c["panel"]
        )
        w_title.pack(anchor="w")

        w_sub = tk.Label(
            welcome_frame,
            text="End-to-end AI system: Resume Parser → TF-IDF Job Matcher → Career Path Recommender → Skill Gap Roadmap.",
            font=("Segoe UI", 9),
            fg=c["text_muted"],
            bg=c["panel"]
        )
        w_sub.pack(anchor="w", pady=(4, 0))

        # Metric KPI cards row
        kpi_row = tk.Frame(scrollable_frame, bg=c["bg"])
        kpi_row.pack(fill="x", pady=(0, 15))

        resume_score_val = f"{data['resume_score']['total']}/100" if data else "— / 100"
        job_match_val = f"{data['job_match']['match_percentage']}%" if data else "— %"
        skills_detected_val = f"{len(data['detected_skills'])}" if data else "—"
        missing_skills_val = f"{len(data['skill_gap']['missing_skills'])}" if data else "—"
        career_rec_val = data['top_career'] if data else "Run Analysis"

        cards_data = [
            ("Resume Score", resume_score_val, "Educational metric", c["accent"]),
            ("Job Match", job_match_val, "TF-IDF + Cosine Sim", c["success"]),
            ("Skills Detected", skills_detected_val, "Technical catalog", c["warning"]),
            ("Target Missing", missing_skills_val, "Skills to learn", c["danger"])
        ]

        for i, (title, val, subtitle, accent_color) in enumerate(cards_data):
            card = tk.Frame(kpi_row, bg=c["panel"], bd=1, relief="solid", padx=15, pady=15, width=205)
            card.pack(side="left", fill="both", expand=True, padx=(0 if i == 0 else 10, 0))

            lbl_t = tk.Label(card, text=title, font=("Segoe UI", 9), fg=c["text_muted"], bg=c["panel"])
            lbl_t.pack(anchor="w")

            lbl_v = tk.Label(card, text=val, font=("Segoe UI", 20, "bold"), fg=accent_color, bg=c["panel"])
            lbl_v.pack(anchor="w", pady=(4, 2))

            lbl_s = tk.Label(card, text=subtitle, font=("Segoe UI", 8), fg=c["text_muted"], bg=c["panel"])
            lbl_s.pack(anchor="w")

        # Two-column dashboard split: Left (Career Recommendation Highlight) + Right (Detected Skills Overview)
        mid_row = tk.Frame(scrollable_frame, bg=c["bg"])
        mid_row.pack(fill="x", pady=(0, 15))

        # Career Highlight Card (Left)
        career_box = tk.Frame(mid_row, bg=c["panel"], bd=1, relief="solid", padx=20, pady=18)
        career_box.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(career_box, text="PRIMARY CAREER RECOMMENDATION", font=("Segoe UI", 9, "bold"), fg=c["accent"], bg=c["panel"]).pack(anchor="w")
        tk.Label(career_box, text=career_rec_val, font=("Segoe UI", 18, "bold"), fg=c["text_main"], bg=c["panel"]).pack(anchor="w", pady=(6, 4))

        rec_desc = data["skill_gap"]["description"] if data else "Run resume analysis to identify your optimal career trajectory."
        tk.Label(career_box, text=rec_desc, font=("Segoe UI", 9), fg=c["text_muted"], bg=c["panel"], wraplength=380, justify="left").pack(anchor="w", pady=(0, 10))

        # Match bar for top career
        if data:
            top_fit = data["skill_gap"]["match_percentage"]
            bar_lbl = tk.Label(career_box, text=f"Career Fit Index: {top_fit}%", font=("Segoe UI", 9, "bold"), fg=c["success"], bg=c["panel"])
            bar_lbl.pack(anchor="w", pady=(0, 4))
            self.draw_inline_progress_bar(career_box, top_fit, c["success"], c["border"], width=360, height=12)

        # Quick Navigation link
        view_career_btn = tk.Button(
            career_box,
            text="Explore Career Path →",
            font=("Segoe UI", 9, "bold"),
            fg=c["accent"],
            bg=c["panel"],
            bd=0,
            cursor="hand2",
            command=lambda: self.show_view("career_rec")
        )
        view_career_btn.pack(anchor="w", pady=(12, 0))

        # Skill Summary Card (Right)
        skill_box = tk.Frame(mid_row, bg=c["panel"], bd=1, relief="solid", padx=20, pady=18)
        skill_box.pack(side="right", fill="both", expand=True)

        tk.Label(skill_box, text="DETECTED TECHNICAL SKILLS", font=("Segoe UI", 9, "bold"), fg=c["accent"], bg=c["panel"]).pack(anchor="w")

        if data and data["detected_skills"]:
            skills_display_frame = tk.Frame(skill_box, bg=c["panel"])
            skills_display_frame.pack(fill="both", expand=True, pady=10)

            # Display first 10 skills as clean badge items
            for sk in data["detected_skills"][:10]:
                badge = tk.Label(
                    skills_display_frame,
                    text=f"✓ {sk}",
                    font=("Segoe UI", 8, "bold"),
                    fg=c["success"],
                    bg=c["input_bg"],
                    padx=8,
                    pady=4,
                    relief="solid",
                    bd=1
                )
                badge.pack(side="left", padx=3, pady=3)

            more_count = len(data["detected_skills"]) - 10
            if more_count > 0:
                more_lbl = tk.Label(
                    skills_display_frame,
                    text=f"+{more_count} more",
                    font=("Segoe UI", 8),
                    fg=c["text_muted"],
                    bg=c["panel"]
                )
                more_lbl.pack(side="left", padx=5)
        else:
            tk.Label(skill_box, text="No skills detected yet. Upload or paste a resume to begin.", font=("Segoe UI", 9), fg=c["text_muted"], bg=c["panel"]).pack(anchor="w", pady=15)

        view_skills_btn = tk.Button(
            skill_box,
            text="Inspect All Skills →",
            font=("Segoe UI", 9, "bold"),
            fg=c["accent"],
            bg=c["panel"],
            bd=0,
            cursor="hand2",
            command=lambda: self.show_view("resume_analyzer")
        )
        view_skills_btn.pack(anchor="w", pady=(5, 0))

        # Quick Action Buttons Panel
        actions_panel = tk.Frame(scrollable_frame, bg=c["panel"], bd=1, relief="solid", padx=20, pady=15)
        actions_panel.pack(fill="x", pady=(0, 20))

        tk.Label(actions_panel, text="QUICK ACTIONS", font=("Segoe UI", 9, "bold"), fg=c["text_muted"], bg=c["panel"]).pack(anchor="w", pady=(0, 10))

        btn_row = tk.Frame(actions_panel, bg=c["panel"])
        btn_row.pack(fill="x")

        a1 = tk.Button(btn_row, text="📄 Analyze New Resume", font=("Segoe UI", 9), bg=c["input_bg"], fg=c["text_main"], bd=1, padx=12, pady=6, cursor="hand2", command=lambda: self.show_view("resume_analyzer"))
        a1.pack(side="left", padx=(0, 10))

        a2 = tk.Button(btn_row, text="⚡ Compare Job Description", font=("Segoe UI", 9), bg=c["input_bg"], fg=c["text_main"], bd=1, padx=12, pady=6, cursor="hand2", command=lambda: self.show_view("job_matcher"))
        a2.pack(side="left", padx=(0, 10))

        a3 = tk.Button(btn_row, text="🗺️ View Learning Roadmap", font=("Segoe UI", 9), bg=c["input_bg"], fg=c["text_main"], bd=1, padx=12, pady=6, cursor="hand2", command=lambda: self.show_view("roadmap"))
        a3.pack(side="left", padx=(0, 10))

        a4 = tk.Button(btn_row, text="📈 View Analytics Charts", font=("Segoe UI", 9), bg=c["input_bg"], fg=c["text_main"], bd=1, padx=12, pady=6, cursor="hand2", command=lambda: self.show_view("analytics"))
        a4.pack(side="left")

    def draw_inline_progress_bar(self, parent, percentage, fill_color, bg_color, width=300, height=10):
        """Draws a crisp anti-aliased progress bar on a Tkinter canvas."""
        canvas = tk.Canvas(parent, width=width, height=height, bg=bg_color, bd=0, highlightthickness=0)
        canvas.pack(anchor="w")
        fill_width = int((percentage / 100.0) * width)
        if fill_width > 0:
            canvas.create_rectangle(0, 0, fill_width, height, fill=fill_color, outline="")

    # --------------------------------------------------------------------------
    # VIEW 2: RESUME ANALYZER
    # --------------------------------------------------------------------------

    def render_resume_analyzer_view(self):
        """Resume text input, file upload, and skill extraction breakdown."""
        c = self.get_colors()

        main_box = tk.Frame(self.content_viewport, bg=c["bg"])
        main_box.pack(fill="both", expand=True)

        # Left Column: Input Text Area & Upload
        left_frame = tk.Frame(main_box, bg=c["panel"], bd=1, relief="solid", padx=16, pady=16)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 12))

        header_row = tk.Frame(left_frame, bg=c["panel"])
        header_row.pack(fill="x", pady=(0, 8))

        tk.Label(header_row, text="Resume Text Content", font=("Segoe UI", 11, "bold"), fg=c["text_main"], bg=c["panel"]).pack(side="left")

        # Upload file button
        upload_btn = tk.Button(
            header_row,
            text="📁 Upload .txt File",
            font=("Segoe UI", 8),
            bg=c["input_bg"],
            fg=c["text_main"],
            bd=1,
            padx=8,
            pady=3,
            cursor="hand2",
            command=self.upload_resume_file
        )
        upload_btn.pack(side="right")

        # Text input area
        self.resume_text_area = tk.Text(
            left_frame,
            font=("Consolas", 9),
            bg=c["input_bg"],
            fg=c["text_main"],
            insertbackground=c["text_main"],
            bd=1,
            relief="solid",
            wrap="word",
            height=20
        )
        self.resume_text_area.pack(fill="both", expand=True, pady=(0, 10))

        # Insert current text if available
        current_text = self.analysis_cache["resume_text"] if self.analysis_cache else self.sample_resume
        if current_text:
            self.resume_text_area.insert("1.0", current_text)

        # Action Analyze Button
        analyze_btn = tk.Button(
            left_frame,
            text="🚀 Analyze Resume & Extract Skills",
            font=("Segoe UI", 10, "bold"),
            bg=c["accent"],
            fg="#FFFFFF",
            activebackground=c["accent_hover"],
            activeforeground="#FFFFFF",
            bd=0,
            pady=8,
            cursor="hand2",
            command=self.trigger_resume_analysis
        )
        analyze_btn.pack(fill="x")

        # Right Column: Detected Skills Display
        right_frame = tk.Frame(main_box, bg=c["panel"], bd=1, relief="solid", padx=16, pady=16, width=380)
        right_frame.pack(side="right", fill="both")
        right_frame.pack_propagate(False)

        tk.Label(right_frame, text="Detected Technical Skills", font=("Segoe UI", 11, "bold"), fg=c["text_main"], bg=c["panel"]).pack(anchor="w")

        skills = self.analysis_cache["detected_skills"] if self.analysis_cache else []

        tk.Label(
            right_frame,
            text=f"Total Identified: {len(skills)} skills",
            font=("Segoe UI", 9),
            fg=c["text_muted"],
            bg=c["panel"]
        ).pack(anchor="w", pady=(2, 10))

        # Skills scrollable list
        sk_canvas = tk.Canvas(right_frame, bg=c["input_bg"], bd=1, relief="solid", highlightthickness=0)
        sk_scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=sk_canvas.yview)
        sk_content = tk.Frame(sk_canvas, bg=c["input_bg"], padx=10, pady=10)

        sk_content.bind("<Configure>", lambda e: sk_canvas.configure(scrollregion=sk_canvas.bbox("all")))
        sk_canvas.create_window((0, 0), window=sk_content, anchor="nw", width=330)
        sk_canvas.configure(yscrollcommand=sk_scrollbar.set)

        sk_canvas.pack(side="left", fill="both", expand=True)
        sk_scrollbar.pack(side="right", fill="y")

        if skills:
            # Group detected skills by category
            for cat, cat_skills in self.engine.TECHNICAL_SKILLS.items():
                matched_in_cat = [s for s in cat_skills if s in skills]
                if matched_in_cat:
                    cat_header = tk.Label(sk_content, text=cat, font=("Segoe UI", 9, "bold"), fg=c["accent"], bg=c["input_bg"])
                    cat_header.pack(anchor="w", pady=(8, 4))
                    for sk in matched_in_cat:
                        badge = tk.Label(
                            sk_content,
                            text=f"✓  {sk}",
                            font=("Segoe UI", 8),
                            fg=c["success"],
                            bg=c["panel"],
                            padx=6,
                            pady=3,
                            relief="solid",
                            bd=1
                        )
                        badge.pack(anchor="w", padx=2, pady=2)
        else:
            tk.Label(
                sk_content,
                text="No skills extracted yet.\nEnter or upload resume text and click 'Analyze'.",
                font=("Segoe UI", 9),
                fg=c["text_muted"],
                bg=c["input_bg"]
            ).pack(anchor="w", pady=10)

    def upload_resume_file(self):
        """Allows user to pick a text file from disk."""
        path = filedialog.askopenfilename(
            title="Select Resume Text File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.resume_text_area.delete("1.0", tk.END)
                self.resume_text_area.insert("1.0", content)
                messagebox.showinfo("File Uploaded", f"Successfully loaded: {os.path.basename(path)}")
            except Exception as ex:
                messagebox.showerror("Error", f"Failed to read file: {str(ex)}")

    def trigger_resume_analysis(self):
        """Executes full analysis using the text area contents."""
        resume_content = self.resume_text_area.get("1.0", tk.END).strip()
        if not resume_content:
            messagebox.showwarning("Empty Resume", "Please enter or paste your resume text before analyzing.")
            return

        job_desc = self.analysis_cache["job_desc_text"] if self.analysis_cache else self.sample_job_desc
        self.perform_full_analysis(resume_content, job_desc)
        self.show_view("resume_analyzer")
        messagebox.showinfo("Analysis Complete", f"Successfully extracted {len(self.analysis_cache['detected_skills'])} technical skills!")

    # --------------------------------------------------------------------------
    # VIEW 3: RESUME SCORE
    # --------------------------------------------------------------------------

    def render_resume_score_view(self):
        """Educational resume scoring breakdown across 6 transparent criteria."""
        c = self.get_colors()
        data = self.analysis_cache

        main_box = tk.Frame(self.content_viewport, bg=c["bg"])
        main_box.pack(fill="both", expand=True)

        if not data:
            tk.Label(main_box, text="Please run a resume analysis first to calculate your score.", font=("Segoe UI", 12), fg=c["text_muted"], bg=c["bg"]).pack(pady=40)
            return

        score_info = data["resume_score"]
        total = score_info["total"]

        # Top Overall Score Banner
        top_banner = tk.Frame(main_box, bg=c["panel"], bd=1, relief="solid", padx=20, pady=16)
        top_banner.pack(fill="x", pady=(0, 15))

        score_left = tk.Frame(top_banner, bg=c["panel"])
        score_left.pack(side="left")

        tk.Label(score_left, text="EDUCATIONAL RESUME SCORE", font=("Segoe UI", 9, "bold"), fg=c["accent"], bg=c["panel"]).pack(anchor="w")
        tk.Label(score_left, text=f"{total} / 100", font=("Segoe UI", 26, "bold"), fg=c["success"] if total >= 75 else c["warning"], bg=c["panel"]).pack(anchor="w", pady=2)

        assessment = "Strong Competency Profile" if total >= 80 else ("Moderate Foundation — Room for Projects/Certifications" if total >= 60 else "Early Stage Resume — Missing Key Sections")
        tk.Label(score_left, text=assessment, font=("Segoe UI", 9), fg=c["text_muted"], bg=c["panel"]).pack(anchor="w")

        # Disclaimer badge (Left as instructed)
        disc_frame = tk.Frame(top_banner, bg=c["input_bg"], bd=1, relief="solid", padx=12, pady=10)
        disc_frame.pack(side="right")
        tk.Label(disc_frame, text="ℹ️ Educational Benchmark Notice", font=("Segoe UI", 8, "bold"), fg=c["text_main"], bg=c["input_bg"]).pack(anchor="w")
        tk.Label(
            disc_frame,
            text="Rule-based heuristic scoring created for student learning.\nDoes not represent a commercial recruiter ATS score.",
            font=("Segoe UI", 8),
            fg=c["text_muted"],
            bg=c["input_bg"]
        ).pack(anchor="w")

        # 6 Pillars Breakdown Grid
        pillars_frame = tk.Frame(main_box, bg=c["bg"])
        pillars_frame.pack(fill="both", expand=True)

        criteria = [
            ("Technical Skills", score_info["skills"], 25, "Recognized technical skills, tools & libraries density", c["accent"]),
            ("Academic Projects", score_info["projects"], 20, "Project headings, GitHub links & action verbs", c["success"]),
            ("Education Credential", score_info["education"], 15, "B.Tech/degree coursework, CGPA & institute info", c["accent"]),
            ("Certifications", score_info["certifications"], 15, "Coursera, NPTEL, Udemy & technical certifications", c["warning"]),
            ("Work Experience / Internships", score_info["experience"], 15, "Internships, industry roles & practical experience", c["accent"]),
            ("Completeness & Contact", score_info["completeness"], 10, "Email, phone number, LinkedIn/GitHub links", c["success"])
        ]

        for i, (title, score, max_s, desc, color) in enumerate(criteria):
            card = tk.Frame(pillars_frame, bg=c["panel"], bd=1, relief="solid", padx=16, pady=12)
            row = i // 2
            col = i % 2
            card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            pillars_frame.grid_columnconfigure(col, weight=1)

            h_row = tk.Frame(card, bg=c["panel"])
            h_row.pack(fill="x")
            tk.Label(h_row, text=title, font=("Segoe UI", 10, "bold"), fg=c["text_main"], bg=c["panel"]).pack(side="left")
            tk.Label(h_row, text=f"{score} / {max_s} pts", font=("Segoe UI", 10, "bold"), fg=color, bg=c["panel"]).pack(side="right")

            pct = int((score / max_s) * 100)
            self.draw_inline_progress_bar(card, pct, color, c["border"], width=380, height=8)

            tk.Label(card, text=desc, font=("Segoe UI", 8), fg=c["text_muted"], bg=c["panel"]).pack(anchor="w", pady=(6, 0))

    # --------------------------------------------------------------------------
    # VIEW 4: JOB MATCHER (TF-IDF + COSINE SIMILARITY)
    # --------------------------------------------------------------------------

    def render_job_matcher_view(self):
        """Job description input and NLP TF-IDF Cosine Similarity calculation."""
        c = self.get_colors()

        main_box = tk.Frame(self.content_viewport, bg=c["bg"])
        main_box.pack(fill="both", expand=True)

        # Left Column: Job Description Text Input
        left_frame = tk.Frame(main_box, bg=c["panel"], bd=1, relief="solid", padx=16, pady=16)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 12))

        tk.Label(left_frame, text="Target Job Description", font=("Segoe UI", 11, "bold"), fg=c["text_main"], bg=c["panel"]).pack(anchor="w")
        tk.Label(left_frame, text="Paste job posting text below to compare against your resume using TF-IDF & Cosine Similarity.", font=("Segoe UI", 8), fg=c["text_muted"], bg=c["panel"]).pack(anchor="w", pady=(0, 8))

        self.job_text_area = tk.Text(
            left_frame,
            font=("Consolas", 9),
            bg=c["input_bg"],
            fg=c["text_main"],
            insertbackground=c["text_main"],
            bd=1,
            relief="solid",
            wrap="word",
            height=18
        )
        self.job_text_area.pack(fill="both", expand=True, pady=(0, 10))

        # Insert current job desc if available
        curr_job = self.analysis_cache["job_desc_text"] if self.analysis_cache else self.sample_job_desc
        if curr_job:
            self.job_text_area.insert("1.0", curr_job)

        match_btn = tk.Button(
            left_frame,
            text="⚡ Run TF-IDF & Cosine Similarity Match",
            font=("Segoe UI", 10, "bold"),
            bg=c["accent"],
            fg="#FFFFFF",
            activebackground=c["accent_hover"],
            activeforeground="#FFFFFF",
            bd=0,
            pady=8,
            cursor="hand2",
            command=self.trigger_job_matching
        )
        match_btn.pack(fill="x")

        # Right Column: Match Result & Overlap Breakdown
        right_frame = tk.Frame(main_box, bg=c["panel"], bd=1, relief="solid", padx=16, pady=16, width=400)
        right_frame.pack(side="right", fill="both")
        right_frame.pack_propagate(False)

        data = self.analysis_cache
        job_match = data["job_match"] if data else None

        tk.Label(right_frame, text="AI Match Results", font=("Segoe UI", 11, "bold"), fg=c["text_main"], bg=c["panel"]).pack(anchor="w")

        if job_match:
            match_pct = job_match["match_percentage"]
            sim_score = job_match["similarity_score"]

            score_banner = tk.Frame(right_frame, bg=c["input_bg"], bd=1, relief="solid", padx=14, pady=12)
            score_banner.pack(fill="x", pady=10)

            tk.Label(score_banner, text="JOB MATCH FIT", font=("Segoe UI", 8, "bold"), fg=c["text_muted"], bg=c["input_bg"]).pack(anchor="w")
            tk.Label(score_banner, text=f"{match_pct}%", font=("Segoe UI", 24, "bold"), fg=c["success"] if match_pct >= 75 else c["warning"], bg=c["input_bg"]).pack(anchor="w")
            tk.Label(score_banner, text=f"TF-IDF Cosine Similarity Vector: {sim_score}", font=("Segoe UI", 8), fg=c["text_muted"], bg=c["input_bg"]).pack(anchor="w", pady=(2, 0))

            # Matching Skills Section
            tk.Label(right_frame, text="Matching Skills (Present in both):", font=("Segoe UI", 9, "bold"), fg=c["success"], bg=c["panel"]).pack(anchor="w", pady=(10, 4))
            m_frame = tk.Frame(right_frame, bg=c["panel"])
            m_frame.pack(fill="x")
            if job_match["matching_skills"]:
                for s in job_match["matching_skills"]:
                    tk.Label(m_frame, text=f"✓ {s}", font=("Segoe UI", 8, "bold"), fg=c["success"], bg=c["input_bg"], padx=6, pady=2, bd=1, relief="solid").pack(side="left", padx=2, pady=2)
            else:
                tk.Label(m_frame, text="None detected", font=("Segoe UI", 8), fg=c["text_muted"], bg=c["panel"]).pack(anchor="w")

            # Missing Skills Section
            tk.Label(right_frame, text="Missing Skills (Found in Job Posting):", font=("Segoe UI", 9, "bold"), fg=c["danger"], bg=c["panel"]).pack(anchor="w", pady=(12, 4))
            miss_frame = tk.Frame(right_frame, bg=c["panel"])
            miss_frame.pack(fill="x")
            if job_match["missing_skills"]:
                for s in job_match["missing_skills"]:
                    tk.Label(miss_frame, text=f"⚠ {s}", font=("Segoe UI", 8, "bold"), fg=c["danger"], bg=c["input_bg"], padx=6, pady=2, bd=1, relief="solid").pack(side="left", padx=2, pady=2)
            else:
                tk.Label(miss_frame, text="None! All requested skills matched.", font=("Segoe UI", 8), fg=c["success"], bg=c["panel"]).pack(anchor="w")

            # Suggested learning priorities
            tk.Label(right_frame, text="Suggested Action to Reach 90%+:", font=("Segoe UI", 9, "bold"), fg=c["accent"], bg=c["panel"]).pack(anchor="w", pady=(14, 4))
            sugg_text = ", ".join(job_match["suggested_skills"]) if job_match["suggested_skills"] else "Add measurable project outcomes."
            tk.Label(right_frame, text=f"Focus on acquiring: {sugg_text}", font=("Segoe UI", 8), fg=c["text_muted"], bg=c["panel"], wraplength=360, justify="left").pack(anchor="w")

        else:
            tk.Label(right_frame, text="Enter a job description and click 'Run TF-IDF Match'.", font=("Segoe UI", 9), fg=c["text_muted"], bg=c["panel"]).pack(anchor="w", pady=15)

    def trigger_job_matching(self):
        """Triggers TF-IDF Cosine Similarity calculation with new job description."""
        job_content = self.job_text_area.get("1.0", tk.END).strip()
        if not job_content:
            messagebox.showwarning("Missing Input", "Please provide a target job description.")
            return

        resume_text = self.analysis_cache["resume_text"] if self.analysis_cache else self.sample_resume
        self.perform_full_analysis(resume_text, job_content)
        self.show_view("job_matcher")
        messagebox.showinfo("Job Match Complete", f"Match Score: {self.analysis_cache['job_match']['match_percentage']}%")

    # --------------------------------------------------------------------------
    # VIEW 5: CAREER RECOMMENDATION
    # --------------------------------------------------------------------------

    def render_career_recommendation_view(self):
        """Evaluates and ranks career paths based on detected skills."""
        c = self.get_colors()
        data = self.analysis_cache

        main_box = tk.Frame(self.content_viewport, bg=c["bg"])
        main_box.pack(fill="both", expand=True)

        if not data:
            tk.Label(main_box, text="Please run a resume analysis first to view career recommendations.", font=("Segoe UI", 12), fg=c["text_muted"], bg=c["bg"]).pack(pady=40)
            return

        ranked = data["career_rec"]["ranked_careers"]
        top_career = data["career_rec"]["top_career"]

        # Top Recommendation Highlight
        top_card = tk.Frame(main_box, bg=c["panel"], bd=1, relief="solid", padx=20, pady=16)
        top_card.pack(fill="x", pady=(0, 15))

        tk.Label(top_card, text="TOP RECOMMENDED CAREER PATH", font=("Segoe UI", 9, "bold"), fg=c["accent"], bg=c["panel"]).pack(anchor="w")
        tk.Label(top_card, text=f"🚀 {top_career}", font=("Segoe UI", 20, "bold"), fg=c["text_main"], bg=c["panel"]).pack(anchor="w", pady=2)
        tk.Label(top_card, text=data["skill_gap"]["description"], font=("Segoe UI", 9), fg=c["text_muted"], bg=c["panel"]).pack(anchor="w", pady=(0, 10))

        # Ranked Career List
        tk.Label(main_box, text="All Career Pathways Ranked by Current Skill Competency:", font=("Segoe UI", 10, "bold"), fg=c["text_main"], bg=c["bg"]).pack(anchor="w", pady=(0, 8))

        scrollable_canvas = tk.Canvas(main_box, bg=c["bg"], bd=0, highlightthickness=0)
        s_bar = ttk.Scrollbar(main_box, orient="vertical", command=scrollable_canvas.yview)
        list_frame = tk.Frame(scrollable_canvas, bg=c["bg"])

        list_frame.bind("<Configure>", lambda e: scrollable_canvas.configure(scrollregion=scrollable_canvas.bbox("all")))
        scrollable_canvas.create_window((0, 0), window=list_frame, anchor="nw", width=880)
        scrollable_canvas.configure(yscrollcommand=s_bar.set)

        scrollable_canvas.pack(side="left", fill="both", expand=True)
        s_bar.pack(side="right", fill="y")

        for item in ranked:
            c_name = item["career"]
            score = item["score"]
            is_top = (c_name == top_career)

            card = tk.Frame(list_frame, bg=c["panel"], bd=1, relief="solid", padx=16, pady=12)
            card.pack(fill="x", pady=5)

            header = tk.Frame(card, bg=c["panel"])
            header.pack(fill="x")

            title_txt = f"{c_name} (Recommended)" if is_top else c_name
            tk.Label(header, text=title_txt, font=("Segoe UI", 11, "bold"), fg=c["accent"] if is_top else c["text_main"], bg=c["panel"]).pack(side="left")
            tk.Label(header, text=f"{score}% Match", font=("Segoe UI", 11, "bold"), fg=c["success"] if score >= 70 else (c["warning"] if score >= 40 else c["danger"]), bg=c["panel"]).pack(side="right")

            self.draw_inline_progress_bar(card, score, c["success"] if score >= 70 else c["warning"], c["border"], width=840, height=8)

            desc_row = tk.Frame(card, bg=c["panel"])
            desc_row.pack(fill="x", pady=(6, 0))

            matched_str = ", ".join(item["matched_skills"]) if item["matched_skills"] else "None"
            missing_str = ", ".join(item["missing_skills"]) if item["missing_skills"] else "None"

            tk.Label(desc_row, text=f"Possessed: {matched_str}", font=("Segoe UI", 8), fg=c["success"], bg=c["panel"]).pack(anchor="w")
            tk.Label(desc_row, text=f"Missing: {missing_str}", font=("Segoe UI", 8), fg=c["text_muted"], bg=c["panel"]).pack(anchor="w")

    # --------------------------------------------------------------------------
    # VIEW 6: SKILL GAP ANALYSIS
    # --------------------------------------------------------------------------

    def render_skill_gap_view(self):
        """Compares user skills against target career with interactive dropdown."""
        c = self.get_colors()
        data = self.analysis_cache

        main_box = tk.Frame(self.content_viewport, bg=c["bg"])
        main_box.pack(fill="both", expand=True)

        if not data:
            tk.Label(main_box, text="Please run a resume analysis first to view skill gaps.", font=("Segoe UI", 12), fg=c["text_muted"], bg=c["bg"]).pack(pady=40)
            return

        # Target Career Selection Row
        sel_row = tk.Frame(main_box, bg=c["panel"], bd=1, relief="solid", padx=16, pady=12)
        sel_row.pack(fill="x", pady=(0, 15))

        tk.Label(sel_row, text="Target Career Path:", font=("Segoe UI", 10, "bold"), fg=c["text_main"], bg=c["panel"]).pack(side="left")

        career_options = list(self.engine.career_map.keys())
        self.selected_target_career = tk.StringVar(value=data["top_career"])

        dropdown = ttk.Combobox(
            sel_row,
            textvariable=self.selected_target_career,
            values=career_options,
            state="readonly",
            width=25,
            font=("Segoe UI", 9)
        )
        dropdown.pack(side="left", padx=10)
        dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_skill_gap_target())

        gap_info = self.engine.analyze_skill_gap(data["detected_skills"], self.selected_target_career.get())

        # Match metric badge
        match_lbl = tk.Label(
            sel_row,
            text=f"Current Match: {gap_info['match_percentage']}%",
            font=("Segoe UI", 10, "bold"),
            fg=c["success"] if gap_info["match_percentage"] >= 70 else c["warning"],
            bg=c["input_bg"],
            padx=10,
            pady=4,
            relief="solid",
            bd=1
        )
        match_lbl.pack(side="right")

        # Two columns: Current Skills vs Missing Skills
        cols_frame = tk.Frame(main_box, bg=c["bg"])
        cols_frame.pack(fill="both", expand=True)

        # Left Column: Current Skills
        curr_box = tk.Frame(cols_frame, bg=c["panel"], bd=1, relief="solid", padx=16, pady=16)
        curr_box.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(curr_box, text=f"Current Skills Possessed ({len(gap_info['current_skills'])})", font=("Segoe UI", 11, "bold"), fg=c["success"], bg=c["panel"]).pack(anchor="w")
        tk.Label(curr_box, text="Skills detected in your resume that fulfill target requirements:", font=("Segoe UI", 8), fg=c["text_muted"], bg=c["panel"]).pack(anchor="w", pady=(0, 10))

        if gap_info["current_skills"]:
            for sk in gap_info["current_skills"]:
                b = tk.Label(curr_box, text=f"✓  {sk}", font=("Segoe UI", 9), fg=c["success"], bg=c["input_bg"], padx=8, pady=4, bd=1, relief="solid")
                b.pack(anchor="w", pady=2)
        else:
            tk.Label(curr_box, text="None detected for this path.", font=("Segoe UI", 9), fg=c["text_muted"], bg=c["panel"]).pack(anchor="w")

        # Right Column: Missing Skills
        miss_box = tk.Frame(cols_frame, bg=c["panel"], bd=1, relief="solid", padx=16, pady=16)
        miss_box.pack(side="right", fill="both", expand=True)

        tk.Label(miss_box, text=f"Missing / Recommended Skills ({len(gap_info['missing_skills'])})", font=("Segoe UI", 11, "bold"), fg=c["danger"], bg=c["panel"]).pack(anchor="w")
        tk.Label(miss_box, text="High-impact skills to acquire to become fully job-ready:", font=("Segoe UI", 8), fg=c["text_muted"], bg=c["panel"]).pack(anchor="w", pady=(0, 10))

        if gap_info["missing_skills"]:
            for sk in gap_info["missing_skills"]:
                b = tk.Label(miss_box, text=f"⚠  {sk}", font=("Segoe UI", 9), fg=c["danger"], bg=c["input_bg"], padx=8, pady=4, bd=1, relief="solid")
                b.pack(anchor="w", pady=2)
        else:
            tk.Label(miss_box, text="Congratulations! You possess all primary required skills.", font=("Segoe UI", 9), fg=c["success"], bg=c["panel"]).pack(anchor="w")

    def update_skill_gap_target(self):
        """Refreshes the skill gap view upon dropdown selection."""
        self.show_view("skill_gap")

    # --------------------------------------------------------------------------
    # VIEW 7: LEARNING ROADMAP
    # --------------------------------------------------------------------------

    def render_roadmap_view(self):
        """Actionable 5-stage personalized learning roadmap based on missing skills."""
        c = self.get_colors()
        data = self.analysis_cache

        main_box = tk.Frame(self.content_viewport, bg=c["bg"])
        main_box.pack(fill="both", expand=True)

        if not data:
            tk.Label(main_box, text="Please run an analysis first to generate a learning roadmap.", font=("Segoe UI", 12), fg=c["text_muted"], bg=c["bg"]).pack(pady=40)
            return

        roadmap_steps = data["roadmap"]

        # Top summary banner
        top_card = tk.Frame(main_box, bg=c["panel"], bd=1, relief="solid", padx=20, pady=14)
        top_card.pack(fill="x", pady=(0, 12))

        tk.Label(top_card, text="YOUR PERSONALIZED AI LEARNING ROADMAP", font=("Segoe UI", 10, "bold"), fg=c["accent"], bg=c["panel"]).pack(anchor="w")
        tk.Label(top_card, text=f"Tailored for: {data['top_career']} Role Readiness", font=("Segoe UI", 13, "bold"), fg=c["text_main"], bg=c["panel"]).pack(anchor="w", pady=2)
        tk.Label(top_card, text="Structured 5-stage educational guide to close detected skill gaps through structured project development.", font=("Segoe UI", 8), fg=c["text_muted"], bg=c["panel"]).pack(anchor="w")

        # Scrollable steps
        canvas = tk.Canvas(main_box, bg=c["bg"], bd=0, highlightthickness=0)
        sbar = ttk.Scrollbar(main_box, orient="vertical", command=canvas.yview)
        steps_frame = tk.Frame(canvas, bg=c["bg"])

        steps_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=steps_frame, anchor="nw", width=880)
        canvas.configure(yscrollcommand=sbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        sbar.pack(side="right", fill="y")

        for s in roadmap_steps:
            card = tk.Frame(steps_frame, bg=c["panel"], bd=1, relief="solid", padx=16, pady=12)
            card.pack(fill="x", pady=5)

            h = tk.Frame(card, bg=c["panel"])
            h.pack(fill="x")

            step_tag = tk.Label(h, text=f"STEP {s['step']}", font=("Segoe UI", 8, "bold"), fg="#FFFFFF", bg=c["accent"], padx=8, pady=2)
            step_tag.pack(side="left")

            title_lbl = tk.Label(h, text=f"  {s['title']}", font=("Segoe UI", 10, "bold"), fg=c["text_main"], bg=c["panel"])
            title_lbl.pack(side="left")

            time_lbl = tk.Label(h, text=f"⏱️ {s['duration']}", font=("Segoe UI", 8), fg=c["text_muted"], bg=c["panel"])
            time_lbl.pack(side="right")

            desc_lbl = tk.Label(card, text=s["description"], font=("Segoe UI", 8), fg=c["text_muted"], bg=c["panel"], wraplength=830, justify="left")
            desc_lbl.pack(anchor="w", pady=(6, 0))

    # --------------------------------------------------------------------------
    # VIEW 8: ANALYTICS & VISUALIZATIONS
    # --------------------------------------------------------------------------

    def render_analytics_view(self):
        """Displays data visualizers comparing resume sections, career fits, and skill categories."""
        c = self.get_colors()
        data = self.analysis_cache

        main_box = tk.Frame(self.content_viewport, bg=c["bg"])
        main_box.pack(fill="both", expand=True)

        if not data:
            tk.Label(main_box, text="Please run an analysis first to generate analytics charts.", font=("Segoe UI", 12), fg=c["text_muted"], bg=c["bg"]).pack(pady=40)
            return

        # Canvas-based high-performance charts (Native, 100% reliable across all systems)
        scrollable_canvas = tk.Canvas(main_box, bg=c["bg"], bd=0, highlightthickness=0)
        s_bar = ttk.Scrollbar(main_box, orient="vertical", command=scrollable_canvas.yview)
        charts_container = tk.Frame(scrollable_canvas, bg=c["bg"])

        charts_container.bind("<Configure>", lambda e: scrollable_canvas.configure(scrollregion=scrollable_canvas.bbox("all")))
        scrollable_canvas.create_window((0, 0), window=charts_container, anchor="nw", width=880)
        scrollable_canvas.configure(yscrollcommand=s_bar.set)

        scrollable_canvas.pack(side="left", fill="both", expand=True)
        s_bar.pack(side="right", fill="y")

        # Chart 1: Resume Scoring Breakdown
        c1_card = tk.Frame(charts_container, bg=c["panel"], bd=1, relief="solid", padx=16, pady=16)
        c1_card.pack(fill="x", pady=(0, 15))

        tk.Label(c1_card, text="Resume Evaluation Section Breakdown (Actual vs Max)", font=("Segoe UI", 11, "bold"), fg=c["text_main"], bg=c["panel"]).pack(anchor="w", pady=(0, 10))

        score_items = [
            ("Technical Skills", data["resume_score"]["skills"], 25),
            ("Projects", data["resume_score"]["projects"], 20),
            ("Education", data["resume_score"]["education"], 15),
            ("Certifications", data["resume_score"]["certifications"], 15),
            ("Experience / Internships", data["resume_score"]["experience"], 15),
            ("Completeness", data["resume_score"]["completeness"], 10)
        ]

        chart1_canvas = tk.Canvas(c1_card, width=830, height=160, bg=c["input_bg"], bd=1, relief="solid", highlightthickness=0)
        chart1_canvas.pack(pady=5)

        # Draw horizontal bars
        for idx, (label, score, max_val) in enumerate(score_items):
            y = 15 + idx * 24
            chart1_canvas.create_text(10, y, text=label, anchor="w", fill=c["text_muted"], font=("Segoe UI", 8))
            
            # Bar background
            bar_start_x = 160
            max_bar_width = 520
            chart1_canvas.create_rectangle(bar_start_x, y - 6, bar_start_x + max_bar_width, y + 6, fill=c["border"], outline="")
            
            # Bar fill
            fill_w = int((score / max_val) * max_bar_width)
            chart1_canvas.create_rectangle(bar_start_x, y - 6, bar_start_x + fill_w, y + 6, fill=c["accent"], outline="")
            
            # Text label
            chart1_canvas.create_text(bar_start_x + max_bar_width + 15, y, text=f"{score}/{max_val}", anchor="w", fill=c["text_main"], font=("Segoe UI", 8, "bold"))

        # Chart 2: Career Path Competency Comparison
        c2_card = tk.Frame(charts_container, bg=c["panel"], bd=1, relief="solid", padx=16, pady=16)
        c2_card.pack(fill="x", pady=(0, 15))

        tk.Label(c2_card, text="Career Path Competency Fit Comparison (%)", font=("Segoe UI", 11, "bold"), fg=c["text_main"], bg=c["panel"]).pack(anchor="w", pady=(0, 10))

        chart2_canvas = tk.Canvas(c2_card, width=830, height=140, bg=c["input_bg"], bd=1, relief="solid", highlightthickness=0)
        chart2_canvas.pack(pady=5)

        ranked = data["career_rec"]["ranked_careers"]
        for idx, item in enumerate(ranked):
            y = 18 + idx * 24
            chart2_canvas.create_text(10, y, text=item["career"], anchor="w", fill=c["text_muted"], font=("Segoe UI", 8))
            
            bar_start_x = 160
            max_bar_width = 520
            chart2_canvas.create_rectangle(bar_start_x, y - 6, bar_start_x + max_bar_width, y + 6, fill=c["border"], outline="")
            
            fill_w = int((item["score"] / 100.0) * max_bar_width)
            bar_col = c["success"] if item["score"] >= 70 else (c["warning"] if item["score"] >= 40 else c["danger"])
            chart2_canvas.create_rectangle(bar_start_x, y - 6, bar_start_x + fill_w, y + 6, fill=bar_col, outline="")
            chart2_canvas.create_text(bar_start_x + max_bar_width + 15, y, text=f"{item['score']}%", anchor="w", fill=c["text_main"], font=("Segoe UI", 8, "bold"))

    # --------------------------------------------------------------------------
    # VIEW 9: SESSION HISTORY
    # --------------------------------------------------------------------------

    def render_history_view(self):
        """Displays analysis runs recorded during current application session."""
        c = self.get_colors()

        main_box = tk.Frame(self.content_viewport, bg=c["bg"])
        main_box.pack(fill="both", expand=True)

        header_frame = tk.Frame(main_box, bg=c["bg"])
        header_frame.pack(fill="x", pady=(0, 10))

        tk.Label(header_frame, text="Session Analysis History", font=("Segoe UI", 11, "bold"), fg=c["text_main"], bg=c["bg"]).pack(side="left")

        # Save to JSON button
        save_btn = tk.Button(
            header_frame,
            text="💾 Export History to JSON",
            font=("Segoe UI", 8),
            bg=c["panel"],
            fg=c["text_main"],
            bd=1,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self.export_history_json
        )
        save_btn.pack(side="right")

        if not self.session_history:
            tk.Label(main_box, text="No analyses have been recorded in this session yet.", font=("Segoe UI", 10), fg=c["text_muted"], bg=c["bg"]).pack(pady=40)
            return

        # Table container
        table_frame = tk.Frame(main_box, bg=c["panel"], bd=1, relief="solid", padx=10, pady=10)
        table_frame.pack(fill="both", expand=True)

        # Table headers
        th = tk.Frame(table_frame, bg=c["input_bg"], pady=6)
        th.pack(fill="x")

        tk.Label(th, text="Timestamp", font=("Segoe UI", 8, "bold"), fg=c["text_muted"], bg=c["input_bg"], width=20, anchor="w").pack(side="left", padx=5)
        tk.Label(th, text="Resume Score", font=("Segoe UI", 8, "bold"), fg=c["text_muted"], bg=c["input_bg"], width=15, anchor="w").pack(side="left")
        tk.Label(th, text="Job Match", font=("Segoe UI", 8, "bold"), fg=c["text_muted"], bg=c["input_bg"], width=15, anchor="w").pack(side="left")
        tk.Label(th, text="Recommended Career", font=("Segoe UI", 8, "bold"), fg=c["text_muted"], bg=c["input_bg"], width=25, anchor="w").pack(side="left")
        tk.Label(th, text="Skills Detected", font=("Segoe UI", 8, "bold"), fg=c["text_muted"], bg=c["input_bg"], width=15, anchor="w").pack(side="left")

        # Table rows
        for idx, item in enumerate(reversed(self.session_history)):
            row = tk.Frame(table_frame, bg=c["panel"] if idx % 2 == 0 else c["input_bg"], pady=6)
            row.pack(fill="x")

            tk.Label(row, text=item["timestamp"], font=("Segoe UI", 8), fg=c["text_main"], bg=row["bg"], width=20, anchor="w").pack(side="left", padx=5)
            tk.Label(row, text=f"{item['resume_score']['total']}/100", font=("Segoe UI", 8, "bold"), fg=c["accent"], bg=row["bg"], width=15, anchor="w").pack(side="left")
            tk.Label(row, text=f"{item['job_match']['match_percentage']}%", font=("Segoe UI", 8, "bold"), fg=c["success"], bg=row["bg"], width=15, anchor="w").pack(side="left")
            tk.Label(row, text=item["top_career"], font=("Segoe UI", 8), fg=c["text_main"], bg=row["bg"], width=25, anchor="w").pack(side="left")
            tk.Label(row, text=f"{len(item['detected_skills'])} skills", font=("Segoe UI", 8), fg=c["text_muted"], bg=row["bg"], width=15, anchor="w").pack(side="left")

    def export_history_json(self):
        """Exports session history into a local JSON file."""
        if not self.session_history:
            messagebox.showwarning("Empty History", "No analysis records to export.")
            return

        export_data = []
        for h in self.session_history:
            export_data.append({
                "timestamp": h["timestamp"],
                "resume_score": h["resume_score"]["total"],
                "job_match_pct": h["job_match"]["match_percentage"],
                "cosine_similarity": h["job_match"]["similarity_score"],
                "top_career": h["top_career"],
                "detected_skills": h["detected_skills"],
                "missing_skills": h["skill_gap"]["missing_skills"]
            })

        out_path = "analysis_history.json"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)
            messagebox.showinfo("Export Successful", f"Saved {len(export_data)} records to: {os.path.abspath(out_path)}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Could not write file: {str(e)}")

    # --------------------------------------------------------------------------
    # VIEW 10: ABOUT & AI WORKFLOW
    # --------------------------------------------------------------------------

    def render_about_view(self):
        """Displays project credentials, CodeOrbit Task 6 architecture, and ML explanation."""
        c = self.get_colors()

        main_box = tk.Frame(self.content_viewport, bg=c["panel"], bd=1, relief="solid", padx=20, pady=20)
        main_box.pack(fill="both", expand=True)

        tk.Label(main_box, text="AI Career Intelligence System", font=("Segoe UI", 14, "bold"), fg=c["accent"], bg=c["panel"]).pack(anchor="w")
        tk.Label(main_box, text="CodeOrbit Tech Artificial Intelligence Internship — Task 6 Capstone", font=("Segoe UI", 9, "bold"), fg=c["text_main"], bg=c["panel"]).pack(anchor="w", pady=(2, 10))

        # Workflow diagram representation
        wf_box = tk.Frame(main_box, bg=c["input_bg"], bd=1, relief="solid", padx=15, pady=12)
        wf_box.pack(fill="x", pady=10)

        tk.Label(wf_box, text="PROJECT ARCHITECTURE & PIPELINE:", font=("Segoe UI", 8, "bold"), fg=c["accent"], bg=c["input_bg"]).pack(anchor="w")
        pipeline_txt = "Data Input (Resume & Job Text) ➔ Data Processing (Regex Skill Extractor) ➔ AI/ML Model (TF-IDF & Cosine Similarity) ➔ Analysis / Recommendation ➔ Modern Desktop Display"
        tk.Label(wf_box, text=pipeline_txt, font=("Segoe UI", 9), fg=c["text_main"], bg=c["input_bg"]).pack(anchor="w", pady=(4, 0))

        # Algorithm Explanations
        algo_frame = tk.Frame(main_box, bg=c["panel"])
        algo_frame.pack(fill="both", expand=True, pady=10)

        tk.Label(algo_frame, text="1. TF-IDF (Term Frequency - Inverse Document Frequency):", font=("Segoe UI", 9, "bold"), fg=c["text_main"], bg=c["panel"]).pack(anchor="w")
        t_desc = "Measures the statistical significance of words across the resume and job description. High frequency in one document combined with specificity across documents yields high vector weights."
        tk.Label(algo_frame, text=t_desc, font=("Segoe UI", 8), fg=c["text_muted"], bg=c["panel"], wraplength=830, justify="left").pack(anchor="w", pady=(2, 6))

        tk.Label(algo_frame, text="2. Cosine Similarity:", font=("Segoe UI", 9, "bold"), fg=c["text_main"], bg=c["panel"]).pack(anchor="w")
        c_desc = "Computes the angular distance between the multi-dimensional TF-IDF vectors of the resume and job description. A value of 1.0 represents identical semantic distribution, while 0.0 represents no overlap."
        tk.Label(algo_frame, text=c_desc, font=("Segoe UI", 8), fg=c["text_muted"], bg=c["panel"], wraplength=830, justify="left").pack(anchor="w", pady=(2, 6))

        tk.Label(algo_frame, text="3. Educational Scoring & Career Recommendation:", font=("Segoe UI", 9, "bold"), fg=c["text_main"], bg=c["panel"]).pack(anchor="w")
        r_desc = "Uses deterministic rule-based evaluation mapped against standard technical competencies (Data Analyst, Data Scientist, AI/ML Engineer, Software Developer, Web Developer) for transparent career guidance."
        tk.Label(algo_frame, text=r_desc, font=("Segoe UI", 8), fg=c["text_muted"], bg=c["panel"], wraplength=830, justify="left").pack(anchor="w", pady=(2, 10))

        # Official Disclaimer Box
        d_box = tk.Frame(main_box, bg=c["input_bg"], bd=1, relief="solid", padx=12, pady=10)
        d_box.pack(fill="x")
        tk.Label(d_box, text="OFFICIAL EDUCATIONAL DISCLAIMER", font=("Segoe UI", 8, "bold"), fg=c["warning"], bg=c["input_bg"]).pack(anchor="w")
        disc_full = (
            "This application is developed for educational purposes. Resume scores, job matching, "
            "career recommendations, and skill suggestions are generated using simple AI/ML techniques "
            "and should not be treated as professional recruitment or career advice."
        )
        tk.Label(d_box, text=disc_full, font=("Segoe UI", 8), fg=c["text_muted"], bg=c["input_bg"], wraplength=820, justify="left").pack(anchor="w", pady=(2, 0))


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Launches the AI Career Intelligence Desktop Application."""
    app = AICareerIntelligenceApp()
    app.mainloop()


if __name__ == "__main__":
    main()

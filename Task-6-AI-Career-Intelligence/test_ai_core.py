"""
Unit & Integration Verification for AI Career Intelligence Core AI/ML Engine.
Tests TF-IDF, Cosine Similarity, Skill Extraction, Scoring, and Recommendations.
"""

import sys
from main import PurePythonNLP, CareerIntelligenceEngine

def test_nlp_engine():
    print("[1/5] Testing TF-IDF & Cosine Similarity Engine...")
    doc1 = "Python developer with machine learning, SQL, and data analysis experience."
    doc2 = "Looking for a Python software engineer experienced in machine learning and SQL databases."
    doc3 = "Veterinary physician specializing in feline surgery and animal healthcare."

    sim_related = PurePythonNLP.compute_tfidf_similarity(doc1, doc2)
    sim_unrelated = PurePythonNLP.compute_tfidf_similarity(doc1, doc3)

    print(f"  Similarity (Related Docs):   {sim_related:.4f}")
    print(f"  Similarity (Unrelated Docs): {sim_unrelated:.4f}")

    assert sim_related > 0.25, "Related documents should have significant TF-IDF similarity"
    assert sim_unrelated < 0.15, "Unrelated documents should have near-zero similarity"
    print("  ✓ TF-IDF & Cosine Similarity Passed!")

def test_skill_extraction():
    print("\n[2/5] Testing Regex Skill Extractor...")
    engine = CareerIntelligenceEngine()
    sample_text = """
    Aarav is an AI engineer with strong skills in Python, C++, SQL, Pandas, NumPy,
    Scikit-learn, and Machine Learning. He uses Git and GitHub for version control.
    """
    skills = engine.extract_skills(sample_text)
    print(f"  Extracted Skills ({len(skills)}): {skills}")

    expected = {"Python", "C++", "SQL", "Pandas", "NumPy", "Scikit-learn", "Machine Learning", "Git", "GitHub"}
    assert expected.issubset(set(skills)), f"Expected skills missing: {expected - set(skills)}"
    print("  ✓ Skill Extraction Passed!")

def test_educational_scoring():
    print("\n[3/5] Testing Educational Resume Scoring (6 Pillars)...")
    engine = CareerIntelligenceEngine()
    with open("sample_data/sample_resume.txt", "r", encoding="utf-8") as f:
        resume_text = f.read()

    skills = engine.extract_skills(resume_text)
    score_breakdown = engine.calculate_educational_resume_score(resume_text, skills)
    print(f"  Score Breakdown: {score_breakdown}")
    print(f"  Total Score: {score_breakdown['total']}/100")

    assert 60 <= score_breakdown['total'] <= 100, "Sample student resume should score between 60 and 100"
    assert score_breakdown['skills'] > 0
    assert score_breakdown['projects'] > 0
    assert score_breakdown['education'] > 0
    print("  ✓ Educational Scoring Passed!")

def test_job_matching_and_recommendation():
    print("\n[4/5] Testing Job Matcher & Career Recommendation...")
    engine = CareerIntelligenceEngine()
    with open("sample_data/sample_resume.txt", "r", encoding="utf-8") as f:
        resume_text = f.read()
    with open("sample_data/sample_job_description.txt", "r", encoding="utf-8") as f:
        job_text = f.read()

    match_result = engine.match_job_description(resume_text, job_text)
    print(f"  Job Match: {match_result['match_percentage']}%")
    print(f"  Matching Skills: {match_result['matching_skills']}")
    print(f"  Missing Skills:  {match_result['missing_skills']}")

    assert match_result['match_percentage'] > 50, "Sample resume and job description should have >50% match"
    assert len(match_result['matching_skills']) >= 4, "Should match key common skills (Python, SQL, etc.)"

    skills = engine.extract_skills(resume_text)
    career_rec = engine.recommend_career(skills)
    print(f"  Top Recommended Career: {career_rec['top_career']} (Score: {career_rec['top_score']}%)")

    gap_analysis = engine.analyze_skill_gap(skills, career_rec['top_career'])
    print(f"  Skill Gap Analysis for {gap_analysis['career']}: Current={gap_analysis['current_skills']}, Missing={gap_analysis['missing_skills']}")

    roadmap = engine.generate_learning_roadmap(career_rec['top_career'], gap_analysis['missing_skills'])
    print(f"  Roadmap Steps Generated: {len(roadmap)}")
    assert len(roadmap) == 5, "Should generate exactly 5 roadmap milestones"

    print("  ✓ Job Matching & Career Recommendation Passed!")

def main():
    print("==================================================")
    print("  RUNNING AI CAREER INTELLIGENCE VERIFICATION SUITE")
    print("==================================================")
    try:
        test_nlp_engine()
        test_skill_extraction()
        test_educational_scoring()
        test_job_matching_and_recommendation()
        print("\n==================================================")
        print("  ALL CORE AI/ML VERIFICATION TESTS PASSED (5/5) ✓")
        print("==================================================")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

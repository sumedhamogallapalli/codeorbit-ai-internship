"""
Task 5: Sentiment Analysis - Desktop GUI
CodeOrbit Tech Artificial Intelligence Internship
Framework: Python Tkinter (Standard Library)

Provides a clean desktop interface to input reviews,
predict sentiment (Positive, Negative, Neutral), and view confidence.
"""

import tkinter as tk
from tkinter import messagebox
from main import load_dataset, train_sentiment_model, predict_sentiment


class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analysis - CodeOrbit Tech Task 5")
        self.root.geometry("560x520")
        self.root.resizable(False, False)
        self.root.configure(bg="#F4F6F9")

        # Train model on startup
        try:
            df = load_dataset("sentiment_data.csv")
            self.vectorizer, self.model, self.accuracy, _ = train_sentiment_model(df)
            self.model_status = f"Model Ready (Test Accuracy: {self.accuracy * 100:.1f}%)"
        except Exception as e:
            self.vectorizer = None
            self.model = None
            self.model_status = f"Model load error: {e}"

        self.create_widgets()

    def create_widgets(self):
        # Header Title
        header_frame = tk.Frame(self.root, bg="#1E293B", pady=16)
        header_frame.pack(fill="x")

        title_label = tk.Label(
            header_frame,
            text="Sentiment Analysis on Text Data",
            font=("Helvetica", 16, "bold"),
            fg="#FFFFFF",
            bg="#1E293B",
        )
        title_label.pack()

        subtitle_label = tk.Label(
            header_frame,
            text="CodeOrbit Tech AI Internship | TF-IDF + Logistic Regression",
            font=("Helvetica", 10),
            fg="#94A3B8",
            bg="#1E293B",
        )
        subtitle_label.pack(pady=(4, 0))

        # Status / Accuracy Bar
        status_label = tk.Label(
            self.root,
            text=self.model_status,
            font=("Helvetica", 9),
            fg="#16A34A" if "Ready" in self.model_status else "#DC2626",
            bg="#F4F6F9",
        )
        status_label.pack(pady=(12, 4))

        # Main Input Container
        content_frame = tk.Frame(self.root, bg="#FFFFFF", padx=20, pady=20, relief="groove", bd=1)
        content_frame.pack(padx=24, pady=10, fill="both", expand=True)

        input_prompt = tk.Label(
            content_frame,
            text="Enter Review / Text to Analyze:",
            font=("Helvetica", 11, "bold"),
            fg="#1E293B",
            bg="#FFFFFF",
            anchor="w",
        )
        input_prompt.pack(fill="x")

        # Text Area
        self.text_input = tk.Text(
            content_frame,
            height=5,
            font=("Helvetica", 11),
            wrap="word",
            relief="solid",
            bd=1,
            padx=8,
            pady=8,
        )
        self.text_input.pack(fill="x", pady=(8, 12))

        # Button Row
        btn_frame = tk.Frame(content_frame, bg="#FFFFFF")
        btn_frame.pack(fill="x", pady=4)

        analyze_btn = tk.Button(
            btn_frame,
            text="Analyze Sentiment",
            font=("Helvetica", 11, "bold"),
            bg="#2563EB",
            fg="#FFFFFF",
            activebackground="#1D4ED8",
            activeforeground="#FFFFFF",
            padx=16,
            pady=8,
            cursor="hand2",
            relief="flat",
            command=self.handle_analyze,
        )
        analyze_btn.pack(side="left", padx=(0, 8))

        clear_btn = tk.Button(
            btn_frame,
            text="Clear",
            font=("Helvetica", 11),
            bg="#E2E8F0",
            fg="#334155",
            activebackground="#CBD5E1",
            padx=14,
            pady=8,
            cursor="hand2",
            relief="flat",
            command=self.handle_clear,
        )
        clear_btn.pack(side="left")

        # Prediction Display Box
        self.result_frame = tk.Frame(content_frame, bg="#F8FAFC", pady=16, padx=12, relief="solid", bd=1)
        self.result_frame.pack(fill="x", pady=(16, 0))

        self.result_label = tk.Label(
            self.result_frame,
            text="Prediction will appear here",
            font=("Helvetica", 13, "bold"),
            fg="#64748B",
            bg="#F8FAFC",
        )
        self.result_label.pack()

        self.confidence_label = tk.Label(
            self.result_frame,
            text="",
            font=("Helvetica", 10),
            fg="#64748B",
            bg="#F8FAFC",
        )
        self.confidence_label.pack(pady=(4, 0))

    def handle_analyze(self):
        if not self.model or not self.vectorizer:
            messagebox.showerror("Error", "Model is not trained. Check dataset.")
            return

        user_text = self.text_input.get("1.0", tk.END).strip()

        # Handle empty input properly
        if not user_text:
            messagebox.showwarning("Empty Input", "Please enter some text before analyzing.")
            self.text_input.focus_set()
            return

        sentiment, conf = predict_sentiment(user_text, self.vectorizer, self.model)

        color_map = {
            "Positive": "#16A34A",  # Green
            "Negative": "#DC2626",  # Red
            "Neutral": "#D97706",   # Amber
        }

        color = color_map.get(sentiment, "#1E293B")
        self.result_label.config(text=f"Sentiment: {sentiment.upper()}", fg=color)
        self.confidence_label.config(text=f"Confidence: {conf:.1f}%")

    def handle_clear(self):
        self.text_input.delete("1.0", tk.END)
        self.result_label.config(text="Prediction will appear here", fg="#64748B")
        self.confidence_label.config(text="")


def main():
    root = tk.Tk()
    app = SentimentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

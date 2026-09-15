# CodeOrbit Tech AI Internship - Task 1: Rule-Based Chatbot

A simple, beginner-friendly console-based chatbot built using Python standard library. Developed as part of the Artificial Intelligence Internship at **CodeOrbit Tech**.

---

## 📌 Project Title
**Task 1: Rule-Based Chatbot in Python**

## 🎯 Objective
The primary objective of this project is to build an interactive console-based chatbot using fundamental programming concepts such as:
- Conditional decision making (`if-elif-else`)
- Keyword pattern matching and string normalization
- Handling user greetings, technical inquiries, FAQs, and courtesies
- Providing safe fallback responses for unrecognized user inputs

---

## 🛠️ Technologies Used
- **Programming Language**: Python 3.8+
- **Modules**: Built-in `datetime` (no external dependencies required)
- **Environment**: Terminal / Command Prompt / IDE (VS Code, PyCharm, or IDLE)

---

## ✨ Features
1. **Interactive Console Interface**: Clean and intuitive command-line prompt for real-time conversation.
2. **Greeting Detection**: Understands various greetings like `hello`, `hi`, `hey`, `good morning`, `good afternoon`, and `good evening`.
3. **Identity & Context Awareness**: Answers questions regarding its identity (`OrbitBot`) and its role in the CodeOrbit Tech AI Internship.
4. **Knowledge Base**: Explains foundational concepts including *What is AI?*, *What is Python?*, and *What is a Rule-Based Chatbot?*.
5. **Help & Utilities**: Provides a helpful `help` command, current date & time, and lighthearted programming jokes.
6. **Graceful Fallback Handling**: Responds politely with helpful guidance whenever an input does not match known rules.
7. **Clean Exit Command**: Users can exit gracefully at any time using `bye`, `exit`, or `quit`.

---

## 🧠 How It Works
1. **Input Normalization**: The bot takes user input, removes extraneous spaces using `.strip()`, and converts the text to lowercase using `.lower()`.
2. **Pattern & Keyword Matching**: It evaluates the normalized string against predefined keywords using Python's `in` operator and `any()` lists.
3. **Rule Hierarchy**:
   - Priority 1: Greetings & Pleasantries
   - Priority 2: Identity & Purpose
   - Priority 3: CodeOrbit Tech & Internship Context
   - Priority 4: Technical & Educational Explanations
   - Priority 5: Utilities (Time, Date, Joke, Help)
   - Priority 6: Session Exit
   - Priority 7: Default Fallback Response
4. **Execution Loop**: A `while True` loop keeps the conversation going until an exit signal is received.

---

## 📂 Project Structure
```text
codeorbit-ai-internship-task-1/
│
├── chatbot.py           # Main Python source code
├── requirements.txt     # Dependency information (uses Python standard library)
├── README.md            # Comprehensive project documentation
└── submission_guide.md  # Video script, LinkedIn post & interview Q&A
```

---

## 🚀 Installation & Setup Steps

### 1. Prerequisites
Ensure you have Python 3.8 or higher installed on your computer. You can verify this by running:
```bash
python --version
# or
python3 --version
```

### 2. Clone or Download the Repository
```bash
git clone https://github.com/<your-username>/codeorbit-task1-rule-based-chatbot.git
cd codeorbit-task1-rule-based-chatbot
```

### 3. (Optional) Virtual Environment
Since the project relies solely on the Python standard library, no virtual environment or external packages are strictly needed. However, you can create one if desired:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

---

## ▶️ How to Run the Project
Execute the script using Python:
```bash
python chatbot.py
# or
python3 chatbot.py
```

---

## 💬 Sample Conversation / Example Output

```text
============================================================
   CODEORBIT TECH AI INTERNSHIP - TASK 1
            RULE-BASED CHATBOT (PYTHON)
============================================================
Welcome! I am OrbitBot.
Type your message below and press Enter.
Type 'help' for a list of sample questions, or 'bye' to exit.
------------------------------------------------------------

You: Hello
Bot: Hello there! Welcome. How can I assist you today?

You: Who are you?
Bot: I am OrbitBot, a simple rule-based AI chatbot developed in Python for the CodeOrbit Tech AI Internship (Task 1).

You: What is AI?
Bot: Artificial Intelligence (AI) is the simulation of human intelligence in machines programmed to think, learn, and solve problems like humans.

You: What is CodeOrbit Tech?
Bot: CodeOrbit Tech is a technology learning and internship platform where students gain practical exposure in domains like Artificial Intelligence and Software Development.

You: Tell me a joke
Bot: Why do programmers prefer dark mode? Because light attracts bugs!

You: Tell me the time
Bot: Today is September 12, 2026, and the current time is 04:39 PM.

You: What is the weather on Mars?
Bot: I'm sorry, I didn't quite understand that. Since I am a rule-based bot, please try asking in simpler terms, or type 'help' to see questions I can answer!

You: Bye
Bot: Goodbye! Thank you for chatting with me. Have a wonderful day!

[Chat session ended. Thank you!]
```

---

## 📦 GitHub Repository Submission
- **Repository Name**: `codeorbit-task1-rule-based-chatbot`
- **Topic Tags**: `codeorbit-tech`, `ai-internship`, `python`, `rule-based-chatbot`, `btech-project`

---

## 👤 Author
- **Intern**: Final-Year B.Tech Student (Artificial Intelligence Internship)
- **Organization**: CodeOrbit Tech
- **Task**: Task 1 - Rule-Based Chatbot

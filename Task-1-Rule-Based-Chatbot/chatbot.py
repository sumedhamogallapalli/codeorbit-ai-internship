"""
=============================================================================
Project Title: Task 1 - Rule-Based Chatbot
Organization: CodeOrbit Tech (AI Internship)
Author: Final-Year B.Tech Student
Description: A simple console-based rule-based chatbot implemented in Python
             using if-else pattern matching, keyword detection, and fallback handling.
=============================================================================
"""

import datetime


def get_bot_response(user_input: str) -> str:
    """
    Decides the bot's response using simple rule-based if-elif-else logic
    and keyword matching.

    Parameters:
        user_input (str): The raw text entered by the user.

    Returns:
        str: The chatbot's reply.
    """
    # Step 1: Pre-process the user input
    # Convert to lowercase and strip whitespace for consistent matching
    cleaned_input = user_input.strip().lower()

    # If the user pressed enter without typing anything
    if not cleaned_input:
        return "Please say something! You can type 'help' to see what I can do."

    # ---------------------------------------------------------
    # RULE 1: Greetings
    # Check if the input contains common greeting keywords
    # ---------------------------------------------------------
    if any(greet in cleaned_input for greet in ["hello", "hi", "hey", "hola", "greetings"]):
        return "Hello there! Welcome. How can I assist you today?"

    elif "good morning" in cleaned_input:
        return "Good morning! Hope you have a productive and great day ahead."

    elif "good afternoon" in cleaned_input:
        return "Good afternoon! How is your day going so far?"

    elif "good evening" in cleaned_input:
        return "Good evening! How can I help you before the day wraps up?"

    # ---------------------------------------------------------
    # RULE 2: Bot Identity & Purpose
    # Answers questions about who the bot is
    # ---------------------------------------------------------
    elif any(phrase in cleaned_input for phrase in ["who are you", "what is your name", "what are you"]):
        return ("I am OrbitBot, a simple rule-based AI chatbot developed in Python "
                "for the CodeOrbit Tech AI Internship (Task 1).")

    elif any(phrase in cleaned_input for phrase in ["what can you do", "your capabilities", "how do you work"]):
        return ("I use keyword matching and conditional if-else rules to answer common questions, "
                "share tech definitions, tell the current time, and converse politely.")

    # ---------------------------------------------------------
    # RULE 3: Well-being & Courtesies
    # Handles pleasantries
    # ---------------------------------------------------------
    elif any(phrase in cleaned_input for phrase in ["how are you", "how are you doing", "how do you do"]):
        return "I am running smoothly, thank you for asking! How are you doing today?"

    elif any(phrase in cleaned_input for phrase in ["i am fine", "i am good", "doing well", "great"]):
        return "Glad to hear that! What would you like to explore today?"

    elif any(phrase in cleaned_input for phrase in ["thank you", "thanks", "thank u"]):
        return "You're very welcome! Feel free to ask if you need anything else."

    # ---------------------------------------------------------
    # RULE 4: CodeOrbit Tech & Internship Task Info
    # Provides context on the internship organization
    # ---------------------------------------------------------
    elif "codeorbit" in cleaned_input or "codeorbit tech" in cleaned_input:
        return ("CodeOrbit Tech is a technology learning and internship platform where students "
                "gain practical exposure in domains like Artificial Intelligence and Software Development.")

    elif "internship" in cleaned_input or "task 1" in cleaned_input:
        return ("This project is Task 1 of the CodeOrbit Tech AI Internship: "
                "Building a beginner-friendly Rule-Based Chatbot in Python.")

    # ---------------------------------------------------------
    # RULE 5: Technical Questions (AI, Python, Rule-Based Chatbot)
    # Explains core concepts in simple student-friendly terms
    # ---------------------------------------------------------
    elif "what is ai" in cleaned_input or "artificial intelligence" in cleaned_input:
        return ("Artificial Intelligence (AI) is the simulation of human intelligence in machines "
                "programmed to think, learn, and solve problems like humans.")

    elif "what is python" in cleaned_input or "why python" in cleaned_input:
        return ("Python is a popular high-level, interpreted programming language known for its "
                "clean syntax, readability, and vast ecosystem in AI and Data Science.")

    elif any(phrase in cleaned_input for phrase in ["rule based", "rule-based", "what is a rule based chatbot"]):
        return ("A rule-based chatbot operates on predetermined rules and pattern matching (if-else logic). "
                "It responds to inputs that match specific keywords or patterns mapped in its code.")

    # ---------------------------------------------------------
    # RULE 6: Utilities (Time & Date, Jokes, Help)
    # Quick handy responses
    # ---------------------------------------------------------
    elif "time" in cleaned_input or "date" in cleaned_input:
        now = datetime.datetime.now()
        current_time = now.strftime("%I:%M %p")
        current_date = now.strftime("%B %d, %Y")
        return f"Today is {current_date}, and the current time is {current_time}."

    elif "joke" in cleaned_input or "tell me a joke" in cleaned_input:
        return ("Why do programmers prefer dark mode? Because light attracts bugs!")

    elif "help" in cleaned_input or "menu" in cleaned_input or "commands" in cleaned_input:
        return (
            "Here are some things you can ask me:\n"
            "  - Greetings: 'hello', 'good morning', 'how are you'\n"
            "  - Identity: 'who are you', 'what can you do'\n"
            "  - Tech questions: 'what is AI?', 'what is Python?', 'what is a rule-based chatbot?'\n"
            "  - Organization: 'what is CodeOrbit Tech?', 'tell me about the internship'\n"
            "  - Extras: 'tell me the time', 'tell me a joke'\n"
            "  - Exit: type 'bye', 'exit', or 'quit' to end the chat."
        )

    # ---------------------------------------------------------
    # RULE 7: Exit Intent
    # Handled inside get_bot_response if called directly
    # ---------------------------------------------------------
    elif any(farewell in cleaned_input for farewell in ["bye", "goodbye", "exit", "quit", "see you"]):
        return "Goodbye! Thank you for chatting with me. Have a wonderful day!"

    # ---------------------------------------------------------
    # RULE 8: Fallback Response (Default)
    # Executed when user input does not match any recognized rule
    # ---------------------------------------------------------
    else:
        return ("I'm sorry, I didn't quite understand that. "
                "Since I am a rule-based bot, please try asking in simpler terms, "
                "or type 'help' to see questions I can answer!")


def main():
    """
    Main function to run the console-based chatbot application.
    """
    print("=" * 60)
    print("   CODEORBIT TECH AI INTERNSHIP - TASK 1")
    print("            RULE-BASED CHATBOT (PYTHON)")
    print("=" * 60)
    print("Welcome! I am OrbitBot.")
    print("Type your message below and press Enter.")
    print("Type 'help' for a list of sample questions, or 'bye' to exit.")
    print("-" * 60)

    # Main conversation loop
    while True:
        try:
            user_message = input("\nYou: ")
            
            # Check for exit condition
            if user_message.strip().lower() in ["bye", "goodbye", "exit", "quit"]:
                bot_reply = get_bot_response(user_message)
                print(f"Bot: {bot_reply}")
                print("\n[Chat session ended. Thank you!]")
                break

            # Get and display bot response
            bot_reply = get_bot_response(user_message)
            print(f"Bot: {bot_reply}")

        except (KeyboardInterrupt, EOFError):
            print("\n\n[Session terminated by user. Goodbye!]")
            break


if __name__ == "__main__":
    main()

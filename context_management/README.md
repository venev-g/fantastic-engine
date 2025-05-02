
# 🧠 Gemini Chatbot with Entity Memory

A memory-augmented chatbot built with **Streamlit**, **LangChain**, and **Google's Gemini**, designed to remember and organize user-specific information like name, preferences, goals, and more — enabling truly personalized conversations.

[Open in Streamlit](https://gemini-chatbot.streamlit.app/)

---

## 🚀 Features

### ✅ Natural Chat Interface

* Built with [Streamlit](https://streamlit.io/) for an intuitive, conversational UI.
* Chat history is retained across interactions in the same session.

### 🧬 Expanded Entity Categories

Extracts and organizes user details into 3 high-level categories:

#### 1. Personal Information

* `name`: User's name
* `age`: User's age
* `location`: Where the user lives
* `occupation`: Job/profession

#### 2. Preferences & Interests

* `preferences_likes`: Things the user likes
* `preferences_dislikes`: Things the user dislikes
* `preferences_hobbies`: User's hobbies
* `preferences_media`: Books, movies, shows, etc.
* `preferences_food`: Favorite foods and drinks

#### 3. Personal Context

* `context_future_plans`: Plans the user has for the future
* `context_problems`: Challenges the user is facing
* `context_goals`: User’s aspirations and goals

---

## 🧠 Structured Memory Organization

* Information is stored using **categorized keys** (e.g., `preferences_likes`, `context_goals`) in `st.session_state.entity_memory`.
* Enables **easier retrieval**, **clean debugging**, and **targeted responses**.

---

## 📤 Enhanced Entity Extraction

* Utilizes a detailed structured prompt with Gemini to extract and **categorize** user information.
* Ensures consistency and accuracy of stored data.
* Automatically ignores irrelevant or missing values using `null` or empty lists.

---

## 🧾 Improved System Prompt Generation

* Dynamically generates a personalized **system prompt** from stored memory.
* Allows Gemini to refer to previous knowledge **naturally** and **contextually**, not repetitively.
* Example prompt fragment:

  > The user has mentioned hobbies: painting, hiking. They are from Mumbai and want to become a data scientist.

---

## 🔍 Better Debugging Experience

* Memory is displayed in a collapsible `Debug: View Categorized Memory` section.
* Data is grouped into:

  * 📝 Personal Information
  * 💙 Preferences & Interests
  * 🔍 Personal Context
* Helpful during development and testing to see what the model has remembered.

---

## 💡 Why Categorization Matters

This chatbot's categorization system helps it:

* **Remember nuanced details** about users
* **Understand distinctions** (e.g., a “goal” vs. a “hobby”)
* **Respond more personally** using relevant context
* **Build a richer, evolving user profile**

---


## 🧪 Example Conversation

![example usage](/context_management/demo.gif)



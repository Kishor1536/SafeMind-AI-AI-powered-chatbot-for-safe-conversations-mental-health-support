---
# 🧠 Enhanced Character.AI Chatbot with Suicide Prevention & RAG Recommendations

This project is an **AI-powered chatbot** that integrates **Character.AI**, **Groq LLM**, and **LangChain RAG** to provide:

✅ **Safe conversations** with smart **message censoring** (removes TOS-violating self-harm content while preserving emotional meaning)
✅ **Suicide ideation detection** with **immediate crisis support**
✅ **Recommendations for mental health professionals** (retrieved using **RAG with FastEmbed + ChromaDB**)
✅ **Crisis helplines (India-focused)** to provide direct human support

---

## ⚡ Features

* **Suicidal ideation detection** using keyword + regex patterns
* **Message censoring** powered by Groq LLM (with fallback keyword replacement)
* **Doctor recommendations (RAG)**:

  * Uses **FastEmbed embeddings** + **ChromaDB vectorstore**
  * Retrieves relevant counselor/doctor profiles from `dataset.json`
* **Crisis Mode**:

  * Displays verified **suicide prevention helplines (India)**
  * Lets the user choose between:

    1. 📞 Call a crisis helpline
    2. 👨‍⚕ Get professional doctor recommendations
    3. ↩ Continue normal chat
* **Character.AI integration** for natural conversation

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **Character.AI API (PyCharacterAI)**
* **Groq LLM** (`llama-3.3-70b-versatile`)
* **LangChain + ChromaDB** (for RAG)
* **FastEmbed** (for embeddings)
* **dotenv** (for API keys management)

---

## 📂 Project Structure

```
.
├── dataset.json              # Doctor/counselor dataset
├── bot.py                    # Main chatbot code
├── .env                      # Store your API keys here
└── README.md                 # Project documentation
```

---

## 🔑 Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/mental-health-chatbot.git
cd mental-health-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add API keys

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Update `token` and `character_id` inside `bot.py` with your **Character.AI** credentials.

---

## ▶️ Run the Bot

```bash
python bot.py
```

---

## 📞 Crisis Helplines (India)

The chatbot includes **verified Indian helplines** such as:

* **AASRA Mumbai** – 022 2754 6669
* **Jeevan Aastha Helpline** – 1800 233 3330
* **Vandrevala Foundation** – 9999 666 555
* **1Life Crisis Support** – 78930 78930
* **iCALL Helpline** – 022 2556 3291

⚠️ **Disclaimer**: This chatbot is **not a replacement for professional help**. If you or someone you know is in crisis, please **call a helpline immediately**.

---

## 🧑‍💻 Author

Built with ❤️ for safe AI conversations and mental health support.
---

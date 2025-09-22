🧠 Enhanced Character.AI Chatbot with Suicide Prevention & RAG Recommendations

This project is an AI-powered chatbot that integrates Character.AI, Groq LLM, and LangChain RAG to provide:

✅ Safe conversations with smart message censoring (removes TOS-violating self-harm content while preserving emotional meaning)
✅ Suicide ideation detection with immediate crisis support
✅ Recommendations for mental health professionals (retrieved using RAG with FastEmbed + ChromaDB)
✅ Crisis helplines (India-focused) to provide direct human support

⚡ Features

Suicidal ideation detection using keyword + regex patterns

Message censoring powered by Groq LLM (with fallback keyword replacement)

Doctor recommendations (RAG):

Uses FastEmbed embeddings + ChromaDB vectorstore

Retrieves relevant counselor/doctor profiles from dataset.json

Crisis Mode:

Displays verified suicide prevention helplines (India)

Lets the user choose between:

📞 Call a crisis helpline

👨‍⚕ Get professional doctor recommendations

↩ Continue normal chat

Character.AI integration for natural conversation

🛠️ Tech Stack

Python 3.10+

Character.AI API (PyCharacterAI)

Groq LLM (llama-3.3-70b-versatile)

LangChain + ChromaDB (for RAG)

FastEmbed (for embeddings)

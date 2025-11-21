# AI-Document-Q-A

# ⚡ Chat with PDF (Groq LPU Version)

A high-performance **RAG (Retrieval-Augmented Generation)** application that allows you to **chat with PDF documents in real time**.

---

## 🔑 Key Features

- ⚡ **Inference:** Uses **Groq API (Llama 3 70B)** for *near-instant responses*  
- 🧠 **Embeddings:** Utilizes **HuggingFace (all-MiniLM-L6-v2)** locally on CPU — *no external embedding costs*  
- 🔐 **Privacy:** Document vectors are stored **in memory (FAISS)** — *no external vector DB*  

---

## 🛠️ Local Installation & Setup

### 1️⃣ Prerequisites

- Python **3.9 or higher**
- A **Groq API Key** → *(Get it for free here)*

---

### 2️⃣ Installation

Clone this repository and install dependencies:

```bash
# Install dependencies
pip install -r requirements.txt

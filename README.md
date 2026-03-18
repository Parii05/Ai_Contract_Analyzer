# 📄 ClauseAI — AI-Powered Contract Intelligence Platform

ClauseAI is an AI-powered platform that helps you understand complex legal contracts in a **simple, structured, and actionable way**.

Whether you are a student, founder, or working professional, ClauseAI breaks down contracts into clear insights, highlights risks, and suggests what you should do next — **without needing legal expertise**.

---

## Key Features

### 1) Multi-Agent Contract Analysis

ClauseAI uses multiple AI agents, each focusing on a specific type of risk:

* **Legal Risk** — unclear or risky clauses
* **Compliance Risk** — policy and regulatory issues
* **Financial Risk** — hidden costs or penalties
* **Operational Risk** — execution and obligation issues

Each agent generates **structured summaries and clause-level insights**.

---

### 2)LangGraph-Based Orchestration

ClauseAI uses a structured workflow system to:

* Control execution flow step-by-step
* Coordinate multiple AI agents
* Ensure reliable and scalable analysis

---

### 3)Pinecone Vector Database Integration

ClauseAI stores contract data using semantic embeddings, which allows it to:

* Enable fast similarity search
* Power Retrieval-Augmented Generation (RAG)
* Support long-term contract memory

---

### 4)Interactive Visualization Dashboard

Instead of reading long outputs, you get a **clean and visual dashboard**:

* Agent-wise Risk Comparison
*  Compliance & Health Scores
*  Important Clause Tables
*  Priority Action Panels
*  Risk Distribution Charts

---

### 5)RAG-Based Question Answering

ClauseAI allows you to ask questions like:

> “What are the penalties in this contract?”

It ensures:

* Semantic search using Pinecone
* Answers grounded strictly in contract clauses
* No hallucinated responses
* Context-aware explanations

---

### 6)Customizable & Downloadable Reports

* Generate structured reports
* Adjust tone (business / legal / technical)
* Easy to download and share

---

### 7)Negotiation AI

ClauseAI doesn’t just analyze — it helps you act:

* Converts detected risks into negotiation suggestions
* Helps revise contract terms before signing

---

### 8)Smart Action Plan

After analysis, ClauseAI provides a **step-by-step guide**:

* What to fix
* What to question
* What to negotiate

All explained in a **simple and practical way**.

---

## 🏗️ System Architecture

```text
User → Streamlit UI
        ↓
Text Extraction
        ↓
LangGraph Orchestration
        ↓
Parallel AI Agents (Groq LLM)
        ↓
Structured JSON Output
        ↓
Pinecone Vector Storage
        ↓
RAG Query Engine
        ↓
Visualization + Reports
```

---

## 🛠️ Tech Stack (Simplified)

* **Frontend:** Streamlit
* **LLM:** Groq-powered models
* **Orchestration:** LangGraph
* **Vector Database:** Pinecone
* **Core Concept:** RAG (Retrieval-Augmented Generation)

---


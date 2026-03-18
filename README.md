# 📄 ClauseAI — AI-Powered Contract Intelligence Platform

ClauseAI is an AI-powered that helps you understand complex legal contracts in a simple , structured , and and actionable way. Whether you are a student , founder, or working professional. This website ClauseAI breaks down contracts into clear insight, highlight risks , and even suggest what you should do next.

---

## 🚀 Key Features

### 1) Multi-Agent Contract Analysis
ClauseAI uses multiple AI agents, each focusing on a specific type of risk
  - Legal Risk- unclear or risky clauses
  - Compliance Risk- policy issues
  - Financial Risk-hidden costs or penalties
  - Operational Risk - execution and obligation issues
- Each agent produces structured summaries and clause-level insights.

### 2) LangGraph-Based Orchestration
ClauseAI uses a structured workflow system
- Control execution flow step-by-step
- Coordinate multiple AI agents
- Ensure reliable and scalable analysis

### 3) Pinecone Vector Database Integration
ClauseAI stores contract data using semantic embeddings, allowing it to
- Enables fast similarity search
- Powers Retrieval-Augmented Generation (RAG)
- Supports long-term contract memory

### 4) Interactive Visualization Dashboard
Instead of reading long text outputs, you get a clean and visual dashboard:
- Agent-wise Risk Comparison
- Compliance & Health Scores
- Important Clause Tables
- Priority Action Panels
- Risk Distribution Charts

### 5)RAG-Based Question Answering
- Semantic search using Pinecone
- Query grounded only in contract clauses
- Prevents hallucinated answers
- Context-aware responses

### 6) Customizable and Downloadable Report Generation
- Downloadable structured reports
- Adjustable tone and focus (business/legal/technical)

### 7) Negotiation AI 
ClauseAI doesn’t just analyze — it helps you act.
- Converts detected risks into negotiation suggestions
- Helps users revise contract terms before signing

### 8) Smart Action Plan
After analysis, you get a step-by-step guide
- Step-by-step guidance after analysis
-eg What to fix, What to question, What to negotiate
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



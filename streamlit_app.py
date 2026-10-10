from dotenv import load_dotenv
load_dotenv()

import os
import json
import streamlit as st
from pypdf import PdfReader
from docx import Document
from groq import Groq
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph_flow import run_langgraph_pipeline
from embeddings.embedding_service import generate_embedding
from vectorstore.pinecone_store import init_pinecone, store_clause_vectors
from vectorstore.pinecone_query import query_contract_clauses
from qa.answer_generator import generate_answer


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="ClauseAI — Contract Intelligence",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------
# SESSION DEFAULTS
# --------------------------------------------------
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "custom_report" not in st.session_state:
    st.session_state.custom_report = None


# --------------------------------------------------
# GROQ CLIENT
# --------------------------------------------------
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))


def safe_llm(prompt: str):
    try:
        r = groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print("LLM Error:", e)
        return None


def safe_json(text):
    try:
        return json.loads(text)
    except:
        return {}


# --------------------------------------------------
# SPLIT CONTRACT FOR AGENTS
# --------------------------------------------------
def get_agent_context(agent, text):

    parts = {
        "Legal": text[:3000],
        "Finance": text[3000:6000],
        "Compliance": text[6000:9000],
        "Operations": text[-3000:]
    }

    return parts.get(agent, text[:3000])


# --------------------------------------------------
# AUTO SUMMARY GENERATOR
# --------------------------------------------------
def generate_agent_summary(agent, clauses_text, full_contract_text):

    context = clauses_text.strip()

    if not context:
        context = full_contract_text[:3000]

    prompt = f"""
You are a {agent} contract analyst.

Generate a concise domain summary.

Return ONLY valid JSON:

{{
  "overall_risk": "Low|Medium|High",
  "key_concerns": [],
  "priority_actions": []
}}

Context:
{context}
"""

    result = safe_llm(prompt)

    return safe_json(result) if result else {}


# --------------------------------------------------
# CUSTOM REPORT GENERATOR (NEW FEATURE)
# --------------------------------------------------
def generate_custom_report(
        agent_results,
        verdict,
        tone,
        focus,
        length,
        structure
):

    prompt = f"""
You are a professional contract consultant.

Generate a customized contract report.

User Preferences:
- Tone: {tone}
- Focus: {focus}
- Length: {length}
- Structure: {structure}

Use this analysis data:
{json.dumps(agent_results)[:6000]}

Final Verdict:
{verdict}

Rules:
- Follow preferences strictly
- Be clear and structured
- No markdown symbols
- Produce clean final report
"""

    return safe_llm(prompt)


# --------------------------------------------------
# UI — HERO
# --------------------------------------------------
st.markdown("""
<h1 style='text-align:center;'>📄 ClauseAI</h1>
<p style='text-align:center; font-size:1.1em;'>
AI-powered Contract Intelligence Platform
</p>
""", unsafe_allow_html=True)

st.divider()


# --------------------------------------------------
# UPLOAD
# --------------------------------------------------
st.markdown("## 🚀 Upload & Analyze Contract")

uploaded_file = st.file_uploader(
    "Upload PDF / DOCX / TXT",
    ["pdf", "docx", "txt"]
)

analyze_btn = st.button("🔍 Analyze Contract", use_container_width=True)


# --------------------------------------------------
# TEXT EXTRACTION
# --------------------------------------------------
def extract_text(file, ext):

    if ext == "pdf":
        return "\n".join(
            p.extract_text() or ""
            for p in PyPDF2.PdfReader(file).pages
        )

    if ext == "docx":
        return "\n".join(
            p.text for p in Document(file).paragraphs if p.text
        )

    if ext == "txt":
        return file.read().decode("utf-8")

    return ""


# --------------------------------------------------
# AGENT PROMPT
# --------------------------------------------------
def agent_prompt(agent, text):

    return f"""
You are a senior {agent} contract risk auditor.

Return ONLY valid JSON:

{{
 "domain_summary": {{
   "overall_risk": "Low|Medium|High",
   "key_concerns": ["..."],
   "priority_actions": ["..."]
 }},
 "clauses": [
   {{
     "clause_type":"",
     "summary":"",
     "risk_level":"Low|Medium|High",
     "recommendation":""
   }}
 ]
}}

Focus only on {agent.upper()} risks.

Contract:
{text}
"""


# --------------------------------------------------
# PARALLEL AGENTS
# --------------------------------------------------
def run_agents_parallel(text):

    agents = ["Legal", "Compliance", "Finance", "Operations"]
    results = {}

    with ThreadPoolExecutor(max_workers=4) as executor:

        futures = {
            executor.submit(
                safe_llm,
                agent_prompt(agent, get_agent_context(agent, text))
            ): agent
            for agent in agents
        }

        for future in as_completed(futures):

            agent = futures[future]
            output = future.result()

            if not output:
                output = safe_llm(
                    agent_prompt(agent, get_agent_context(agent, text))
                )

            results[agent] = output if output else "{}"

    return results


# --------------------------------------------------
# FINAL VERDICT
# --------------------------------------------------
def generate_verdict(agent_results):

    return safe_llm(f"""
Summarize overall contract risk:

{json.dumps(agent_results)[:6000]}

Format:
### Overall Risk
**Key Issues**
**Red Flags**
**Next Steps**
""")


# --------------------------------------------------
# MAIN ANALYSIS
# --------------------------------------------------
if uploaded_file and analyze_btn:

    ext = uploaded_file.name.split(".")[-1].lower()
    text = extract_text(uploaded_file, ext)

    if not text:
        st.error("❌ Failed to extract text")
        st.stop()

    st.session_state.full_contract_text = text

    with st.spinner("🧠 Running multi-agent analysis..."):

        run_langgraph_pipeline(text)

        agent_results = run_agents_parallel(text)

        verdict = generate_verdict(agent_results)

        # Pinecone
        index = init_pinecone()
        contract_id = uploaded_file.name.replace(" ", "_")

        for agent, content in agent_results.items():

            parsed = safe_json(content)
            clauses = parsed.get("clauses", [])

            for c in clauses:
                c["embedding"] = generate_embedding(
                    f"{c.get('clause_type')} {c.get('summary')}"
                )

            if clauses:
                store_clause_vectors(index, contract_id, agent, clauses)


    st.session_state.analysis_done = True
    st.session_state.agent_results = agent_results
    st.session_state.verdict = verdict
    st.session_state.contract_name = uploaded_file.name
    st.session_state.custom_report = None

    st.success("✅ Analysis completed successfully!")


# --------------------------------------------------
# RESULTS
# --------------------------------------------------
if st.session_state.analysis_done:

    st.divider()
    st.markdown("## 📊 Analysis Results")


    # ---------- Verdict ----------
    st.subheader("✅ Final Verdict")
    st.markdown(st.session_state.verdict)


    # ---------- Agent Tabs ----------
    st.subheader("🧠 Agent Reports")

    tabs = st.tabs(st.session_state.agent_results.keys())


    for tab, agent in zip(tabs, st.session_state.agent_results):

        with tab:

            parsed = safe_json(st.session_state.agent_results[agent])

            clauses = parsed.get("clauses", [])

            summary = parsed.get("domain_summary")

            if not summary or not summary.get("overall_risk"):

                clause_text = "\n".join(
                    f"{c.get('clause_type')}: {c.get('summary')}"
                    for c in clauses
                )

                summary = generate_agent_summary(
                    agent,
                    clause_text,
                    st.session_state.full_contract_text
                )

                parsed["domain_summary"] = summary
                st.session_state.agent_results[agent] = json.dumps(parsed)


            st.markdown(f"### 📊 {agent} Summary")

            st.metric("Overall Risk", summary.get("overall_risk", "N/A"))

            st.markdown("**Key Concerns**")
            for x in summary.get("key_concerns", ["None"]):
                st.write("•", x)

            st.markdown("**Priority Actions**")
            for x in summary.get("priority_actions", ["None"]):
                st.write("•", x)

            st.divider()


    # --------------------------------------------------
    # CUSTOM REPORT SECTION (NEW)
    # --------------------------------------------------
    st.subheader("📄 Customize & Download Report")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        tone = st.selectbox(
            "Tone",
            ["Formal", "Simple", "Professional", "Legal-Style"]
        )

    with c2:
        focus = st.selectbox(
            "Focus",
            ["Balanced", "Risk", "Finance", "Compliance", "Operations"]
        )

    with c3:
        length = st.selectbox(
            "Length",
            ["Short", "Medium", "Detailed"]
        )

    with c4:
        structure = st.selectbox(
            "Structure",
            ["Executive Summary", "Bullet Points", "Paragraph"]
        )


    gen_btn = st.button("✨ Generate Custom Report")


    if gen_btn:

        with st.spinner("📝 Generating personalized report..."):

            report = generate_custom_report(
                st.session_state.agent_results,
                st.session_state.verdict,
                tone,
                focus,
                length,
                structure
            )

            if report:
                st.session_state.custom_report = report
                st.success("✅ Custom report generated!")
            else:
                st.error("❌ Failed to generate report")


    # Preview + Download
    if st.session_state.custom_report:

        st.markdown("### 👀 Report Preview")

        st.text_area(
            "Generated Report",
            st.session_state.custom_report,
            height=300
        )

        st.download_button(
            "⬇️ Download Custom Report",
            st.session_state.custom_report,
            "ClauseAI_Custom_Report.txt"
        )


    # ---------- RAG ----------
    st.subheader("💬 Ask Questions (RAG)")

    q = st.text_input("Ask a question about this contract")

    if st.button("Ask"):

        matches = query_contract_clauses(q)

        answer = generate_answer(q, matches)

        st.markdown(answer)


# --------------------------------------------------
# FEATURES
# --------------------------------------------------
st.divider()
st.markdown("## ✨ Advanced Features")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### 📊 Visualization Dashboard")

with c2:
    st.markdown("### 🤝 Negotiation AI")

with c3:
    st.markdown("### 🗺️ Smart Action Plan")

# --------------------------------------------------
# CONTACT & FEEDBACK (MINIMAL PROTOTYPE)
# --------------------------------------------------
st.divider()


col1, col2 = st.columns(2)

# -------------------------
# CONTACT INFO (Simple)
# -------------------------
with col1:
    st.markdown("### 📩 Contact Us")

    st.write("Have questions or suggestions?")
    st.write("📧 Email: support@clauseai.ai")
    st.write("🌐 Website: www.clauseai.ai")
    st.write("📍 Location: India")


# -------------------------
# QUICK FEEDBACK
# -------------------------
with col2:
    st.markdown("### ⭐ Quick Feedback")

    rating = st.slider("Rate ClauseAI", 1, 5, 4)

    feedback = st.text_area(
        "Your suggestion (optional)",
        height=100
    )

    if st.button("Submit"):
        st.success("✅ Thank you for your feedback!")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.markdown("© 2026 ClauseAI — Developed by Pari")

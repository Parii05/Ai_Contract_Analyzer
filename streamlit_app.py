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
# PRESENTATION ONLY — UI THEME
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#172033; --muted:#65718a; --brand:#6257e8; --brand2:#8b7cff; --line:#e5e9f2; --surface:#ffffff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 12% 0%, rgba(112,101,255,.10), transparent 32%), #f7f8fc; color:var(--ink); }
[data-testid="stHeader"] { background:rgba(247,248,252,.82); }
.block-container { max-width:1180px; padding-top:2.2rem; padding-bottom:3rem; }
h1,h2,h3,h4 { font-family:'Manrope','DM Sans',sans-serif; letter-spacing:-.035em; color:#172033; }
h1 { font-weight:800; font-size:clamp(2.5rem,5vw,4rem)!important; line-height:1.05!important; }
h2 { font-weight:800; margin-top:1.4rem!important; }
h3 { font-weight:700; }
p, label, .stMarkdown, [data-testid="stCaptionContainer"] { color:#45516a; }
div[data-testid="stFileUploader"] { background:#fff; border:1px dashed #b8bdf5; border-radius:18px; padding:1rem; }
div[data-testid="stFileUploader"] section { border:0; }
.stButton > button, .stDownloadButton > button { border-radius:12px; border:1px solid #deddfb; min-height:2.8rem; font-weight:700; transition:all .18s ease; }
.stButton > button[kind="primary"], .stButton > button:hover, .stDownloadButton > button:hover { background:linear-gradient(120deg,#6257e8,#8275fa); color:white; border-color:transparent; box-shadow:0 8px 22px rgba(98,87,232,.18); transform:translateY(-1px); }
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div { border-radius:12px; border-color:#dfe4ef; background:#fff; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:1.1rem 1.2rem; border-radius:16px; box-shadow:0 4px 18px rgba(32,44,80,.035); }
[data-testid="stMetricLabel"] { color:#69758c; font-weight:600; }
[data-testid="stMetricValue"] { color:#292d50; font-family:'Manrope',sans-serif; font-weight:800; }
div[data-testid="stTabs"] button { border-radius:10px 10px 0 0; font-weight:700; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#6257e8; }
div[data-testid="stAlert"] { border-radius:14px; }
hr { border-color:#e4e8f1; margin:1.8rem 0; }
div[data-testid="stExpander"] { background:#fff; border:1px solid var(--line); border-radius:14px; }
footer { visibility:hidden; }
@media (max-width: 768px) {
 .block-container { padding:1.2rem 1rem 2rem; }
 h1 { font-size:2.5rem!important; }
}
</style>
""", unsafe_allow_html=True)



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
<div style="text-align:center;padding:2.2rem 1rem 1.6rem;">
  <div style="display:inline-flex;align-items:center;gap:.5rem;padding:.4rem .8rem;border:1px solid #e1defe;border-radius:999px;background:#f0efff;color:#5d52d8;font-size:.82rem;font-weight:700;margin-bottom:1.2rem;">
    <span>✦</span> AI-POWERED CONTRACT INTELLIGENCE
  </div>
  <h1 style="text-align:center;margin:0 0 .8rem;">Clause<span style="color:#6257e8;">AI</span></h1>
  <p style="text-align:center;font-size:1.12rem;max-width:650px;margin:0 auto;color:#65718a;line-height:1.7;">
    Understand the fine print. Surface risks. Make confident contract decisions.
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()


# --------------------------------------------------
# UPLOAD
# --------------------------------------------------
st.markdown("""
<div style="padding:1.5rem 1.6rem 1rem;background:linear-gradient(135deg,#fff 0%,#f4f3ff 100%);border:1px solid #e7e6fb;border-radius:20px 20px 0 0;margin-top:.5rem;">
  <div style="font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;color:#7066dd;font-weight:800;margin-bottom:.4rem;">GET STARTED</div>
  <h2 style="margin:.1rem 0 .4rem;font-size:1.65rem;">Upload &amp; analyze your contract</h2>
  <p style="margin:0;color:#65718a;">Add a PDF, Word document, or text file to generate a multi-agent risk review.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a contract file · PDF, DOCX or TXT",
    ["pdf", "docx", "txt"]
)

analyze_btn = st.button("✦  Analyze Contract", use_container_width=True, type="primary")


# --------------------------------------------------
# TEXT EXTRACTION
# --------------------------------------------------
def extract_text(file, ext):

    if ext == "pdf":
        return "\n".join(
            page.extract_text() or ""
            for page in PdfReader(file).pages
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

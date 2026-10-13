"""
app.py - Streamlit Dashboard for Call QA Scoring.
Automated call scoring and risk detection for Indian Contact Centres.
"""

import os
import io
import time
from pathlib import Path
import streamlit as st
import pandas as pd

from rubric import DEFAULT_RUBRIC, DEFAULT_CATEGORICAL_FLAGS
from transcribe import transcribe_with_speakers, format_transcript_for_llm
from score import score_transcript
from db import (
    init_db,
    save_call_record,
    get_all_calls,
    get_call_by_id,
    get_flagged_calls,
    get_summary_stats,
    get_calls_dataframe,
    seed_sample_records
)

# Page configuration
st.set_page_config(
    page_title="Call QA | Indian Contact Centre Scoring",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure DB initialized & seeded
init_db()
seed_sample_records()

# Custom Styling for executive UI
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222D;
        border: 1px solid #2E3440;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .agent-bubble {
        background-color: #1a2736;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.95rem;
    }
    .customer-bubble {
        background-color: #27251f;
        border-left: 4px solid #f59e0b;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.95rem;
    }
    .flag-badge-red {
        background-color: #dc2626;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .flag-badge-orange {
        background-color: #d97706;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .flag-badge-green {
        background-color: #059669;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# --- SIDEBAR: Upload & Engine Settings ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/headset.png", width=64)
    st.title("Call QA Engine")
    st.caption("100% automated call QA for Indian BPOs & Contact Centres")
    
    st.divider()
    st.subheader("Batch Audio Upload")
    uploaded_files = st.file_uploader(
        "Upload Call Recordings (.wav, .mp3)",
        type=["wav", "mp3"],
        accept_multiple_files=True,
        help="Upload single or batch recordings to transcribe and score."
    )
    
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    if uploaded_files:
        if st.button("🚀 Process Batch Calls", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                file_path = upload_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                status_text.text(f"Processing ({idx+1}/{len(uploaded_files)}): {uploaded_file.name}...")
                
                # Run transcription & scoring
                transcript = transcribe_with_speakers(str(file_path))
                scores = score_transcript(transcript)
                
                record = {
                    "file_name": uploaded_file.name,
                    "audio_path": str(file_path),
                    "transcript": transcript,
                    "scores": scores
                }
                save_call_record(record)
                progress_bar.progress((idx + 1) / len(uploaded_files))
                
            status_text.text("✅ Batch processing complete!")
            st.success(f"Successfully scored {len(uploaded_files)} calls.")
            time.sleep(1)
            st.rerun()

    st.divider()
    st.subheader("Demo Tools")
    if st.button("⚡ Score Sample Call (1-Click)", use_container_width=True):
        sample_path = Path("samples/sample_call.wav")
        if sample_path.exists():
            with st.spinner("Transcribing & evaluating sample call..."):
                t = transcribe_with_speakers(str(sample_path))
                s = score_transcript(t)
                rec = {
                    "file_name": "sample_call.wav",
                    "audio_path": str(sample_path),
                    "transcript": t,
                    "scores": s
                }
                cid = save_call_record(rec)
                st.success(f"Sample call evaluated! Saved as ID #{cid}")
                time.sleep(0.5)
                st.rerun()
        else:
            st.error("samples/sample_call.wav not found. Run generate_sample.py first.")

    st.divider()
    st.subheader("Engine Status")
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if groq_key:
        st.markdown("🟢 **Groq LLM**: Active (`llama-3.3-70b-versatile`)")
    else:
        st.markdown("🟡 **Groq LLM**: No key (Using offline rule engine)")
        
    if gemini_key:
        st.markdown("🟢 **Gemini Fallback**: Active")
    else:
        st.markdown("⚪ **Gemini Fallback**: Inactive")
        
    st.caption("Stack: faster-whisper + pyannote + Groq/Llama-3.3-70B + SQLite")


# --- MAIN CONTENT AREA ---
st.title("📞 Call QA Scoring Dashboard")
st.markdown("Automated 10-dimension rubric evaluation and risk flagging across 100% of contact centre calls.")

# Summary Metrics Bar
stats = get_summary_stats()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Calls Audited", value=stats["total_calls"])
with col2:
    st.metric(label="Average QA Score", value=f"{stats['avg_score']}/100", delta=None)
with col3:
    st.metric(label="Flagged High-Risk Calls", value=f"{stats['flagged_percentage']}%", delta_color="inverse")
with col4:
    critical_count = stats["total_compliance_violations"] + stats["total_escalations"]
    st.metric(label="Critical Risk Events", value=critical_count)

st.markdown("---")

# Main Navigation Tabs
tab_flagged, tab_drilldown, tab_export, tab_rubric = st.tabs([
    "🚨 Flagged Calls (Top 20 Worst)",
    "🔍 Call Drilldown & Transcript",
    "📥 CSV Data Export",
    "📋 QA Rubric Dimensions"
])

# --- TAB 1: FLAGGED CALLS ---
with tab_flagged:
    st.subheader("Top Flagged Calls (Immediate QA Review Queue)")
    st.caption("Instead of randomly sampling 2%, QA Managers review the worst-performing calls prioritized by regulatory risk and escalation severity.")
    
    flagged_calls = get_flagged_calls(limit=20)
    
    if not flagged_calls:
        st.info("No flagged or low-scoring calls found in database.")
    else:
        table_rows = []
        for call in flagged_calls:
            flags_list = []
            if call["compliance_violation"]:
                flags_list.append("🚨 Compliance Violation")
            if call["customer_escalated"]:
                flags_list.append("⚠️ Customer Escalated")
            if call["prohibited_words_used"]:
                flags_list.append("⛔ Prohibited Words")
            if not flags_list and call["overall_score"] < 70:
                flags_list.append("📉 Low Quality Score")
                
            table_rows.append({
                "Call ID": call["id"],
                "File Name": call["file_name"],
                "Date": call["created_at"][:16].replace("T", " "),
                "Overall Score": f"{call['overall_score']:.1f}",
                "Active Flags": ", ".join(flags_list),
                "Executive Summary": call["summary"]
            })
            
        df_flagged = pd.DataFrame(table_rows)
        st.dataframe(df_flagged, use_container_width=True, hide_index=True)
        
    st.markdown("---")
    st.subheader("Dimension Score Distribution")
    df_all = get_calls_dataframe()
    if not df_all.empty:
        dim_cols = [
            "Greeting", "Customer Verification", "Active Listening", "Empathy",
            "Clarity", "Problem Resolution", "Compliance", "Hold Handling",
            "Objection Handling", "Closing"
        ]
        avg_dims = df_all[dim_cols].mean().round(1)
        st.bar_chart(avg_dims)


# --- TAB 2: CALL DRILLDOWN ---
with tab_drilldown:
    all_calls = get_all_calls()
    if not all_calls:
        st.warning("No calls processed yet. Upload audio or click 'Score Sample Call' in sidebar.")
    else:
        call_options = {f"#{c['id']} - {c['file_name']} (Score: {c['overall_score']})": c['id'] for c in all_calls}
        selected_label = st.selectbox("Select Call for In-Depth Audit:", list(call_options.keys()))
        selected_id = call_options[selected_label]
        
        call_data = get_call_by_id(selected_id)
        
        if call_data:
            parsed = call_data["parsed_json"]
            scores_data = parsed.get("scores", {})
            flags = scores_data.get("flags", {})
            
            c_header1, c_header2 = st.columns([2, 1])
            with c_header1:
                st.markdown(f"### {call_data['file_name']}")
                st.caption(f"Processed: {call_data['created_at']}")
                st.write(f"**Executive Summary:** {call_data['summary']}")
            with c_header2:
                ov_score = call_data["overall_score"]
                color = "green" if ov_score >= 80 else "orange" if ov_score >= 60 else "red"
                st.markdown(f"<div style='text-align:right;'><h1 style='color:{color}; font-size:3rem; margin:0;'>{ov_score:.1f}</h1><p>Overall Score / 100</p></div>", unsafe_allow_html=True)
                
            # Badges
            badge_html = []
            if flags.get("compliance_violation"):
                badge_html.append("<span class='flag-badge-red'>🚨 Compliance Violation</span>")
            if flags.get("customer_escalated"):
                badge_html.append("<span class='flag-badge-orange'>⚠️ Customer Escalated</span>")
            if flags.get("prohibited_words_used"):
                badge_html.append("<span class='flag-badge-red'>⛔ Prohibited Words</span>")
            if not badge_html:
                badge_html.append("<span class='flag-badge-green'>✅ No Risk Flags Detected</span>")
            st.markdown("".join(badge_html), unsafe_allow_html=True)
            
            # Audio playback if file exists
            audio_path = parsed.get("audio_path")
            if audio_path and Path(audio_path).exists():
                st.audio(str(audio_path))
                
            st.markdown("---")
            
            # Two-column layout: Rubric Breakdown vs Speaker Transcript
            col_rubric, col_transcript = st.columns([1, 1])
            
            with col_rubric:
                st.subheader("10-Dimension Quality Rubric")
                dimensions = scores_data.get("dimensions", {})
                
                for key, meta in DEFAULT_RUBRIC.items():
                    item = dimensions.get(key, {})
                    sc = item.get("score", 0) if isinstance(item, dict) else (item or 0)
                    reason = item.get("reason", "N/A") if isinstance(item, dict) else ""
                    
                    with st.expander(f"**{meta['name']}**: {sc}/10", expanded=(sc < 7)):
                        st.caption(f"**Focus:** {meta['description']}")
                        st.write(f"**Auditor Reasoning:** {reason}")
                        st.progress(sc / 10.0)
                        
                strengths = scores_data.get("strengths", [])
                if strengths:
                    st.success(f"**Agent Strengths:**\n- " + "\n- ".join(strengths))
                coaching = scores_data.get("coaching_areas", [])
                if coaching:
                    st.warning(f"**Coaching Areas:**\n- " + "\n- ".join(coaching))

            with col_transcript:
                st.subheader("Speaker-Diarized Transcript")
                transcript_segments = parsed.get("transcript", [])
                
                if not transcript_segments:
                    st.info("No transcript segments available.")
                else:
                    for seg in transcript_segments:
                        speaker = seg.get("speaker", "Speaker")
                        text = seg.get("text", "")
                        ts = f"{int(seg.get('start', 0) // 60):02d}:{int(seg.get('start', 0) % 60):02d}"
                        
                        if speaker.lower() == "agent":
                            st.markdown(f"<div class='agent-bubble'><strong>🎧 [{ts}] Agent:</strong> {text}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='customer-bubble'><strong>👤 [{ts}] Customer:</strong> {text}</div>", unsafe_allow_html=True)


# --- TAB 3: CSV EXPORT ---
with tab_export:
    st.subheader("Export Audit Records")
    st.markdown("Download full structured call records including all 10 dimension scores and categorical risk flags.")
    
    df_export = get_calls_dataframe()
    if df_export.empty:
        st.info("No call records available to export.")
    else:
        st.dataframe(df_export, use_container_width=True)
        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")
        
        st.download_button(
            label="📥 Download Calls Audit CSV",
            data=csv_bytes,
            file_name="CallQA_Audit_Report.csv",
            mime="text/csv",
            type="primary"
        )


# --- TAB 4: RUBRIC GUIDE ---
with tab_rubric:
    st.subheader("Standard QA Rubric for Indian Contact Centres")
    st.markdown("""
    The starter rubric assesses **10 core communication & compliance dimensions** (0–10 each) plus 3 critical flags.
    In commercial deployments, this rubric is calibrated to each client's specific compliance scripts and QA scorecards.
    """)
    
    for idx, (dim_k, dim_v) in enumerate(DEFAULT_RUBRIC.items(), 1):
        st.markdown(f"### {idx}. {dim_v['name']}")
        st.write(dim_v["description"])
        st.markdown(f"""
        - **Good (8-10):** {dim_v['criteria']['good']}
        - **Average (5-7):** {dim_v['criteria']['average']}
        - **Poor (0-4):** {dim_v['criteria']['poor']}
        """)
        st.markdown("---")

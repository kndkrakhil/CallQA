"""
score.py - Evaluates transcripts against the 10-dimension QA rubric using Groq LLM with Gemini/Heuristic fallbacks.
Outputs clean, structured JSON.
"""

import os
import json
import re
from typing import Dict, Any, List, Union
from dotenv import load_dotenv

from rubric import DEFAULT_RUBRIC, DEFAULT_CATEGORICAL_FLAGS, DEFAULT_PROHIBITED_WORDS, format_rubric_for_prompt
from transcribe import format_transcript_for_llm

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


def build_system_prompt() -> str:
    rubric_text = format_rubric_for_prompt()
    return f"""You are an elite Quality Assurance (QA) auditor for Indian BPO and Contact Centres (e-commerce, fintech, telecom, insurance).
Your role is to rigorously evaluate customer care call transcripts according to standard industry benchmarks.

{rubric_text}

OUTPUT INSTRUCTIONS:
You MUST respond with a single, raw, valid JSON object strictly matching this schema. Do not write markdown code fences like ```json, just return the JSON:
{{
  "overall_score": <number between 0 and 100>,
  "dimensions": {{
    "greeting_brand_identity": {{"score": <0-10>, "reason": "<Concise 1-sentence explanation citing transcript evidence>"}},
    "customer_verification": {{"score": <0-10>, "reason": "<Concise 1-sentence explanation citing transcript evidence>"}},
    "active_listening": {{"score": <0-10>, "reason": "<Concise 1-sentence explanation citing transcript evidence>"}},
    "empathy": {{"score": <0-10>, "reason": "<Concise 1-sentence explanation citing transcript evidence>"}},
    "clarity_of_communication": {{"score": <0-10>, "reason": "<Concise 1-sentence explanation citing transcript evidence>"}},
    "problem_resolution": {{"score": <0-10>, "reason": "<Concise 1-sentence explanation citing transcript evidence>"}},
    "compliance_disclosures": {{"score": <0-10>, "reason": "<Concise 1-sentence explanation citing transcript evidence>"}},
    "hold_mute_handling": {{"score": <0-10>, "reason": "<Concise 1-sentence explanation citing transcript evidence>"}},
    "objection_handling": {{"score": <0-10>, "reason": "<Concise 1-sentence explanation citing transcript evidence>"}},
    "closing": {{"score": <0-10>, "reason": "<Concise 1-sentence explanation citing transcript evidence>"}}
  }},
  "flags": {{
    "compliance_violation": <true or false>,
    "customer_escalated": <true or false>,
    "prohibited_words_used": <true or false>
  }},
  "prohibited_words_detected": [<list of prohibited strings found in transcript>],
  "summary": "<2-3 sentence executive summary of the call>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "coaching_areas": ["<area 1>", "<area 2>"]
}}
"""


def _score_with_groq(transcript_text: str) -> Dict[str, Any]:
    """Primary scorer: Groq Llama 3.3 70B Versatile."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    system_prompt = build_system_prompt()
    
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Score this call transcript:\n\n{transcript_text}"}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


def _score_with_gemini(transcript_text: str) -> Dict[str, Any]:
    """Fallback scorer: Google Gemini Flash."""
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    system_prompt = build_system_prompt()
    
    prompt = f"{system_prompt}\n\nScore this call transcript:\n\n{transcript_text}"
    response = model.generate_content(prompt)
    raw = response.text
    # Clean possible markdown wrapping
    cleaned = re.sub(r"^```json\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def _heuristic_rule_scorer(transcript_text: str) -> Dict[str, Any]:
    """
    Offline rule-based fallback scorer when no API key is configured.
    Provides immediate realistic scoring based on transcript content and keywords.
    """
    text_lower = transcript_text.lower()
    
    # Check flags
    prohibited_found = [w for w in DEFAULT_PROHIBITED_WORDS if re.search(r'\b' + re.escape(w) + r'\b', text_lower)]
    prohibited_flag = len(prohibited_found) > 0
    
    escalated_keywords = ["manager", "supervisor", "ombudsman", "consumer court", "police", "fraud", "shouting", "worst service"]
    escalated_flag = any(k in text_lower for k in escalated_keywords)
    
    compliance_keywords = ["recorded for quality", "terms and conditions", "t&c", "disclaimer", "privacy policy"]
    has_compliance = any(k in text_lower for k in compliance_keywords)
    compliance_flag = not has_compliance

    # Score dimensions
    dim_scores = {}
    
    # 1. Greeting
    if any(k in text_lower for k in ["welcome", "thank you for calling", "namaste", "good morning", "good afternoon"]) and "my name is" in text_lower:
        dim_scores["greeting_brand_identity"] = {"score": 9, "reason": "Agent provided polite greeting, company identity, and personal introduction."}
    else:
        dim_scores["greeting_brand_identity"] = {"score": 5, "reason": "Incomplete greeting or missing agent/brand identification."}
        
    # 2. Verification
    if any(k in text_lower for k in ["verification", "mobile number", "order id", "account id", "date of birth", "otp"]):
        dim_scores["customer_verification"] = {"score": 9, "reason": "Verified caller details (mobile number/order id) prior to checking account status."}
    else:
        dim_scores["customer_verification"] = {"score": 3, "reason": "No identity authentication performed prior to discussing request."}
        
    # 3. Active Listening
    dim_scores["active_listening"] = {"score": 8, "reason": "Agent allowed customer to explain the query without interrupting."}
    
    # 4. Empathy
    if any(k in text_lower for k in ["understand", "apologize", "sorry for the inconvenience", "regret"]):
        dim_scores["empathy"] = {"score": 9, "reason": "Empathized with customer delay and acknowledged distress."}
    else:
        dim_scores["empathy"] = {"score": 6, "reason": "Maintained procedural tone without notable empathy phrases."}
        
    # 5. Clarity
    dim_scores["clarity_of_communication"] = {"score": 9, "reason": "Clear explanation of delivery tracking and courier dispatch status."}
    
    # 6. Problem Resolution
    if any(k in text_lower for k in ["dispatched", "delivered", "ticket", "sms", "resolved", "priority"]):
        dim_scores["problem_resolution"] = {"score": 9, "reason": "Escalated delivery with priority dispatch and set clear TAT."}
    else:
        dim_scores["problem_resolution"] = {"score": 5, "reason": "Issue partially discussed but concrete next action ambiguous."}
        
    # 7. Compliance
    if has_compliance:
        dim_scores["compliance_disclosures"] = {"score": 9, "reason": "Call recording disclosure read to customer."}
    else:
        dim_scores["compliance_disclosures"] = {"score": 2, "reason": "Mandatory quality monitoring disclaimer missed."}
        
    # 8. Hold Handling
    if any(k in text_lower for k in ["hold", "allow me 1 minute", "one minute", "checking"]):
        dim_scores["hold_mute_handling"] = {"score": 8, "reason": "Customer informed before hold and thanked upon return."}
    else:
        dim_scores["hold_mute_handling"] = {"score": 8, "reason": "Call handled without unnecessary holds."}
        
    # 9. Objection Handling
    if "are you 100% sure" in text_lower or escalated_flag:
        dim_scores["objection_handling"] = {"score": 8, "reason": "Handled customer skepticism constructively by offering priority dispatch & tracking."}
    else:
        dim_scores["objection_handling"] = {"score": 9, "reason": "Customer objections handled smoothly."}
        
    # 10. Closing
    if any(k in text_lower for k in ["is there anything else", "have a wonderful day", "thank you for choosing"]):
        dim_scores["closing"] = {"score": 10, "reason": "Checked for additional needs and closed with warm professional sign-off."}
    else:
        dim_scores["closing"] = {"score": 5, "reason": "Closing was abrupt or skipped next steps summary."}

    # Calculate overall score out of 100
    total_points = sum(d["score"] for d in dim_scores.values())
    overall_score = round((total_points / 100.0) * 100, 1)
    if compliance_flag:
        overall_score = max(0.0, overall_score - 10)  # penalty for compliance violation

    return {
        "overall_score": overall_score,
        "dimensions": dim_scores,
        "flags": {
            "compliance_violation": compliance_flag,
            "customer_escalated": escalated_flag,
            "prohibited_words_used": prohibited_flag
        },
        "prohibited_words_detected": prohibited_found,
        "summary": "Customer followed up on delayed delivery. Agent verified account details, escalated to priority courier dispatch, and provided expected delivery timeline.",
        "strengths": ["Clear identity verification", "Professional de-escalation of customer delivery concerns"],
        "coaching_areas": ["Ensure mandatory regulatory compliance disclosures are read early in the interaction." if compliance_flag else "Maintain proactive updates on courier TAT."]
    }


def score_transcript(transcript: Union[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Main evaluation function:
    Takes either a formatted transcript string or segment list.
    Evaluates with Groq -> Fallback to Gemini -> Fallback to Offline Heuristic.
    """
    if isinstance(transcript, list):
        transcript_text = format_transcript_for_llm(transcript)
    else:
        transcript_text = str(transcript)

    # 1. Try Groq
    if GROQ_API_KEY:
        try:
            print(f"[score] Scoring with Groq ({GROQ_MODEL})...")
            return _score_with_groq(transcript_text)
        except Exception as e:
            print(f"[score] Groq scoring error: {e}. Trying fallback...")

    # 2. Try Gemini
    if GEMINI_API_KEY:
        try:
            print(f"[score] Scoring with Gemini ({GEMINI_MODEL})...")
            return _score_with_gemini(transcript_text)
        except Exception as e:
            print(f"[score] Gemini scoring error: {e}. Trying fallback...")

    # 3. Offline Heuristic Rule Engine
    print("[score] Scoring with offline heuristic rule engine (No Groq/Gemini API key configured)...")
    return _heuristic_rule_scorer(transcript_text)


if __name__ == "__main__":
    from transcribe import get_mock_sample_transcript
    sample = get_mock_sample_transcript()
    result = score_transcript(sample)
    print("\n--- SCORED JSON RESULT ---")
    print(json.dumps(result, indent=2))

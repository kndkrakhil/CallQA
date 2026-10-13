"""
rubric.py - Starter QA Rubric for Indian Contact Centres
Defines the 10 quality dimensions and 3 critical categorical flags.
"""

from typing import Dict, Any, List

DEFAULT_RUBRIC: Dict[str, Dict[str, Any]] = {
    "greeting_brand_identity": {
        "name": "Greeting & Brand Identity",
        "description": "Did agent greet properly, state their name, and clearly identify the company/brand?",
        "max_score": 10,
        "criteria": {
            "good": "Warm greeting, clearly mentioned agent name and company within first 10 seconds.",
            "average": "Greeted but missed company name or rushed through greeting.",
            "poor": "Abrupt start, no greeting, or company name mispronounced/omitted."
        }
    },
    "customer_verification": {
        "name": "Customer Verification",
        "description": "Did agent verify customer identity (e.g., registered phone, account ID, OTP, DOB/address) before disclosing sensitive account details?",
        "max_score": 10,
        "criteria": {
            "good": "Standard 2-point or 3-point authentication completed before sharing sensitive info.",
            "average": "Partial verification performed; asked only name.",
            "poor": "Shared sensitive account details without any verification (Critical security risk)."
        }
    },
    "active_listening": {
        "name": "Active Listening",
        "description": "Did agent let customer finish speaking without interrupting, and acknowledge customer concerns?",
        "max_score": 10,
        "criteria": {
            "good": "Listened attentively, minimal interruptions, used verbal nods, repeated key details.",
            "average": "Occasional over-talking or slight inattention.",
            "poor": "Frequently spoke over customer, talked down, or ignored customer statements."
        }
    },
    "empathy": {
        "name": "Empathy & Tone",
        "description": "Did agent express understanding and genuine care for the customer's distress or situation?",
        "max_score": 10,
        "criteria": {
            "good": "Demonstrated sincere empathy ('I completely understand how frustrating this must be...').",
            "average": "Neutral, robotic tone with formal scripted apologies.",
            "poor": "Cold, dismissive, sarcastic, or hostile attitude towards customer frustration."
        }
    },
    "clarity_of_communication": {
        "name": "Clarity of Communication",
        "description": "Jargon-free, clear speech, correct information, and appropriate pace/language (English/Hindi/Hinglish)?",
        "max_score": 10,
        "criteria": {
            "good": "Crystal clear explanations, matched customer language preference seamlessly.",
            "average": "Mostly clear but used internal jargon or had slight hesitation.",
            "poor": "Confusing explanations, heavy jargon, inaccurate instructions given."
        }
    },
    "problem_resolution": {
        "name": "Problem Resolution & Ownership",
        "description": "Was the customer's core query or issue properly addressed, resolved, or set on a concrete next step?",
        "max_score": 10,
        "criteria": {
            "good": "First-contact resolution achieved or accurate ticket/escalation reference given with TAT.",
            "average": "Issue partially addressed; left some customer doubts unresolved.",
            "poor": "Gave incorrect solution, passed the buck without logging, or hung up without solving."
        }
    },
    "compliance_disclosures": {
        "name": "Compliance Disclosures",
        "description": "Did agent state mandatory regulatory and legal disclaimers (call recording notice, T&C, charges, privacy terms)?",
        "max_score": 10,
        "criteria": {
            "good": "All mandatory disclaimers read completely and verbatim where required.",
            "average": "Disclaimers mentioned but incomplete or mumbled.",
            "poor": "Mandatory compliance disclaimers entirely skipped."
        }
    },
    "hold_mute_handling": {
        "name": "Hold & Mute Handling",
        "description": "Did agent ask permission and state reason before putting customer on hold, give estimated time, and thank upon return?",
        "max_score": 10,
        "criteria": {
            "good": "Asked permission, gave time limit (e.g., 2 mins), refreshed if extended, thanked customer on return.",
            "average": "Put on hold without explaining reason or did not thank customer upon returning.",
            "poor": "Dead air / unexplained silence >45s or kept on hold without permission."
        }
    },
    "objection_handling": {
        "name": "Objection Handling",
        "description": "When customer pushed back, questioned policies, or resisted solutions, did agent remain calm and de-escalate effectively?",
        "max_score": 10,
        "criteria": {
            "good": "Maintained composure, acknowledged objection, provided constructive alternatives.",
            "average": "Repeated policy blindly without adapting to customer objection.",
            "poor": "Became defensive, argumentative, or aggressive when customer objected."
        }
    },
    "closing": {
        "name": "Closing & Next Steps",
        "description": "Did agent summarize next steps/TAT, ask if any other assistance was needed, thank the customer, and sign off professionally?",
        "max_score": 10,
        "criteria": {
            "good": "Clear summary of actions + SLA/TAT, asked 'Is there anything else?', warm brand sign-off.",
            "average": "Closed conversation but missed summarizing next steps or SLA.",
            "poor": "Abrupt disconnection, no closing pleasantry, or disconnected while customer was speaking."
        }
    }
}

DEFAULT_CATEGORICAL_FLAGS: Dict[str, Dict[str, str]] = {
    "compliance_violation": {
        "name": "Compliance Violation",
        "description": "Any mandatory regulatory, privacy, KYC, or statutory disclosure was skipped or violated."
    },
    "customer_escalated": {
        "name": "Customer Escalated",
        "description": "Customer raised their voice, showed extreme frustration, used abusive language, or explicitly demanded a supervisor/manager/ombudsman."
    },
    "prohibited_words_used": {
        "name": "Prohibited Words Used",
        "description": "Agent used unapproved slang, rude phrases, competitor names, unauthorized personal promises ('guarantee 100% refund'), or prohibited words."
    }
}

DEFAULT_PROHIBITED_WORDS: List[str] = [
    "guaranteed", "pakka", "promise kar raha hu", "mera guarantee hai",
    "fraud", "scam", "idiot", "pagal", "chup", "bakwaas", "shut up",
    "stupid", "useless company", "not my problem"
]

def format_rubric_for_prompt() -> str:
    """Formats the 10 dimensions and flags into a clean prompt string for the LLM."""
    prompt_lines = ["### Quality Scoring Rubric (10 Dimensions, score each 0 to 10):\n"]
    for i, (dim_key, dim_val) in enumerate(DEFAULT_RUBRIC.items(), 1):
        prompt_lines.append(f"{i}. **{dim_val['name']}** (`{dim_key}`):")
        prompt_lines.append(f"   - Focus: {dim_val['description']}")
        prompt_lines.append(f"   - Scoring guide: Good (8-10): {dim_val['criteria']['good']} | Average (5-7): {dim_val['criteria']['average']} | Poor (0-4): {dim_val['criteria']['poor']}")
    
    prompt_lines.append("\n### Categorical Risk Flags (Boolean: true / false):")
    for flag_key, flag_val in DEFAULT_CATEGORICAL_FLAGS.items():
        prompt_lines.append(f"- **{flag_val['name']}** (`{flag_key}`): {flag_val['description']}")
    
    prompt_lines.append("\nProhibited words to monitor for: " + ", ".join(f"'{w}'" for w in DEFAULT_PROHIBITED_WORDS))
    return "\n".join(prompt_lines)

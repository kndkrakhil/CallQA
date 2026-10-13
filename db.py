"""
db.py - SQLite storage layer and CSV exporter for Call QA Scoring.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

DB_FILE = Path(__file__).parent / "callqa.db"


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    target = db_path or DB_FILE
    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None):
    """Initializes the database schema."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        call_uuid TEXT UNIQUE NOT NULL,
        file_name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        overall_score REAL NOT NULL,
        compliance_violation INTEGER NOT NULL DEFAULT 0,
        customer_escalated INTEGER NOT NULL DEFAULT 0,
        prohibited_words_used INTEGER NOT NULL DEFAULT 0,
        score_greeting REAL,
        score_verification REAL,
        score_listening REAL,
        score_empathy REAL,
        score_clarity REAL,
        score_resolution REAL,
        score_compliance REAL,
        score_hold REAL,
        score_objection REAL,
        score_closing REAL,
        summary TEXT,
        raw_json TEXT NOT NULL
    );
    """)
    conn.commit()
    conn.close()


def save_call_record(record: Dict[str, Any], db_path: Optional[Path] = None) -> int:
    """Saves a processed call result dictionary into the SQLite database."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    
    call_uuid = str(uuid.uuid4())
    file_name = record.get("file_name", "unnamed_call.wav")
    created_at = datetime.utcnow().isoformat()
    
    scores = record.get("scores", {})
    overall_score = float(scores.get("overall_score", 0.0))
    flags = scores.get("flags", {})
    
    compliance_violation = 1 if flags.get("compliance_violation") else 0
    customer_escalated = 1 if flags.get("customer_escalated") else 0
    prohibited_words_used = 1 if flags.get("prohibited_words_used") else 0
    
    dims = scores.get("dimensions", {})
    
    def get_dim_val(key: str) -> float:
        val = dims.get(key, {})
        if isinstance(val, dict):
            return float(val.get("score", 0.0))
        return float(val) if val else 0.0

    score_greeting = get_dim_val("greeting_brand_identity")
    score_verification = get_dim_val("customer_verification")
    score_listening = get_dim_val("active_listening")
    score_empathy = get_dim_val("empathy")
    score_clarity = get_dim_val("clarity_of_communication")
    score_resolution = get_dim_val("problem_resolution")
    score_compliance = get_dim_val("compliance_disclosures")
    score_hold = get_dim_val("hold_mute_handling")
    score_objection = get_dim_val("objection_handling")
    score_closing = get_dim_val("closing")
    
    summary = scores.get("summary", "")
    raw_json = json.dumps(record)
    
    cur.execute("""
        INSERT INTO calls (
            call_uuid, file_name, created_at, overall_score,
            compliance_violation, customer_escalated, prohibited_words_used,
            score_greeting, score_verification, score_listening, score_empathy,
            score_clarity, score_resolution, score_compliance, score_hold,
            score_objection, score_closing, summary, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        call_uuid, file_name, created_at, overall_score,
        compliance_violation, customer_escalated, prohibited_words_used,
        score_greeting, score_verification, score_listening, score_empathy,
        score_clarity, score_resolution, score_compliance, score_hold,
        score_objection, score_closing, summary, raw_json
    ))
    conn.commit()
    call_id = cur.lastrowid
    conn.close()
    return call_id


def get_all_calls(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Fetch all scored calls."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM calls ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_call_by_id(call_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve full call details by database ID."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM calls WHERE id = ?", (call_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        data = dict(row)
        data["parsed_json"] = json.loads(data["raw_json"])
        return data
    return None


def get_flagged_calls(db_path: Optional[Path] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch top worst calls sorted by severity (compliance violations first, lowest score first).
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM calls 
        WHERE compliance_violation = 1 OR customer_escalated = 1 OR prohibited_words_used = 1 OR overall_score < 70
        ORDER BY 
            (compliance_violation * 4 + customer_escalated * 3 + prohibited_words_used * 3) DESC,
            overall_score ASC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_summary_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Calculate dashboard aggregate KPIs."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            COUNT(*) as total_calls,
            AVG(overall_score) as avg_score,
            SUM(compliance_violation) as total_compliance_violations,
            SUM(customer_escalated) as total_escalations,
            SUM(prohibited_words_used) as total_prohibited_words
        FROM calls
    """)
    row = cur.fetchone()
    conn.close()
    
    total = row["total_calls"] if row and row["total_calls"] else 0
    avg_score = round(row["avg_score"], 1) if row and row["avg_score"] is not None else 0.0
    comp_v = row["total_compliance_violations"] if row and row["total_compliance_violations"] else 0
    esc_v = row["total_escalations"] if row and row["total_escalations"] else 0
    proh_v = row["total_prohibited_words"] if row and row["total_prohibited_words"] else 0
    
    flagged_percent = round(((comp_v + esc_v + proh_v) / max(1, total)) * 100, 1)
    
    return {
        "total_calls": total,
        "avg_score": avg_score,
        "total_compliance_violations": comp_v,
        "total_escalations": esc_v,
        "total_prohibited_words": proh_v,
        "flagged_percentage": min(100.0, flagged_percent)
    }


def get_calls_dataframe(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Returns a pandas DataFrame of all calls suitable for export."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    df = pd.read_sql_query("""
        SELECT 
            id as "Call ID",
            file_name as "Audio File",
            created_at as "Processed Timestamp",
            overall_score as "Overall Score",
            CASE WHEN compliance_violation = 1 THEN 'YES' ELSE 'NO' END as "Compliance Violation",
            CASE WHEN customer_escalated = 1 THEN 'YES' ELSE 'NO' END as "Customer Escalated",
            CASE WHEN prohibited_words_used = 1 THEN 'YES' ELSE 'NO' END as "Prohibited Words",
            score_greeting as "Greeting",
            score_verification as "Customer Verification",
            score_listening as "Active Listening",
            score_empathy as "Empathy",
            score_clarity as "Clarity",
            score_resolution as "Problem Resolution",
            score_compliance as "Compliance",
            score_hold as "Hold Handling",
            score_objection as "Objection Handling",
            score_closing as "Closing",
            summary as "Executive Summary"
        FROM calls
        ORDER BY id DESC
    """, conn)
    conn.close()
    return df


def seed_sample_records():
    """Populates initial realistic demo calls if database is empty."""
    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM calls")
    count = cur.fetchone()[0]
    conn.close()
    
    if count > 0:
        return

    sample_calls = [
        {
            "file_name": "delhi_courier_delayed_01.wav",
            "scores": {
                "overall_score": 88.0,
                "flags": {"compliance_violation": False, "customer_escalated": False, "prohibited_words_used": False},
                "dimensions": {
                    "greeting_brand_identity": {"score": 9, "reason": "Standard greeting with brand and agent name."},
                    "customer_verification": {"score": 10, "reason": "Verified mobile and order number."},
                    "active_listening": {"score": 8, "reason": "Listened attentively without interrupting."},
                    "empathy": {"score": 9, "reason": "Expressed sincere empathy for shipment delay."},
                    "clarity_of_communication": {"score": 9, "reason": "Clear delivery timeline provided."},
                    "problem_resolution": {"score": 9, "reason": "Dispatched priority delivery tracking."},
                    "compliance_disclosures": {"score": 9, "reason": "Call recording disclosure read."},
                    "hold_mute_handling": {"score": 8, "reason": "Customer informed before 1-minute check."},
                    "objection_handling": {"score": 8, "reason": "Addressed delivery guarantee concern constructively."},
                    "closing": {"score": 9, "reason": "Polite sign-off with ticket reference."}
                },
                "summary": "Customer followed up on delayed order. Agent verified account, escalated with courier partner, and confirmed next-day delivery."
            },
            "transcript": [
                {"start": 0.0, "end": 4.0, "speaker": "Agent", "text": "Namaste, thank you for calling QuickCart. My name is Rahul. How may I help you?", "confidence": 0.95},
                {"start": 4.2, "end": 8.5, "speaker": "Customer", "text": "Hi Rahul, where is my order? It was supposed to be delivered yesterday.", "confidence": 0.92},
                {"start": 8.7, "end": 12.0, "speaker": "Agent", "text": "I apologize for the delay sir. May I have your order ID?", "confidence": 0.96},
                {"start": 12.2, "end": 14.5, "speaker": "Customer", "text": "It is QC-9921.", "confidence": 0.94},
                {"start": 15.0, "end": 20.0, "speaker": "Agent", "text": "Thank you. It has reached Delhi main depot and will be delivered by 2 PM tomorrow. Call is recorded for quality.", "confidence": 0.93}
            ]
        },
        {
            "file_name": "fintech_loan_escalation_02.wav",
            "scores": {
                "overall_score": 42.0,
                "flags": {"compliance_violation": True, "customer_escalated": True, "prohibited_words_used": False},
                "dimensions": {
                    "greeting_brand_identity": {"score": 6, "reason": "Mumbled greeting, company name not audible."},
                    "customer_verification": {"score": 2, "reason": "CRITICAL: Agent disclosed loan balance without verifying OTP or DOB."},
                    "active_listening": {"score": 4, "reason": "Repeatedly interrupted customer when they disputed interest charges."},
                    "empathy": {"score": 3, "reason": "Cold and argumentative tone."},
                    "clarity_of_communication": {"score": 6, "reason": "Used internal billing acronyms without explaining."},
                    "problem_resolution": {"score": 4, "reason": "Did not raise disputed charge grievance ticket."},
                    "compliance_disclosures": {"score": 1, "reason": "RBI mandatory call recording disclaimer was skipped."},
                    "hold_mute_handling": {"score": 5, "reason": "Placed on hold without asking permission for 3 minutes."},
                    "objection_handling": {"score": 3, "reason": "Escalated argument when customer threatened ombudsman complaint."},
                    "closing": {"score": 4, "reason": "Abrupt line disconnection before customer query was resolved."}
                },
                "summary": "Customer contested unauthorized interest penalty. Agent failed KYC verification, argued with customer, skipped statutory disclosure, and hung up abruptly."
            },
            "transcript": [
                {"start": 0.0, "end": 3.0, "speaker": "Agent", "text": "Hello, loan desk.", "confidence": 0.88},
                {"start": 3.2, "end": 8.5, "speaker": "Customer", "text": "Why have you debited 1,500 rupees penalty on my EMI? I paid on 3rd itself!", "confidence": 0.94},
                {"start": 8.7, "end": 13.0, "speaker": "Agent", "text": "System shows bounce charges. You have to pay, nothing we can do.", "confidence": 0.90},
                {"start": 13.2, "end": 18.0, "speaker": "Customer", "text": "Connect me to your supervisor right now! This is complete fraud, I will complain to RBI!", "confidence": 0.95},
                {"start": 18.5, "end": 22.0, "speaker": "Agent", "text": "Supervisor is not available. Pay the fee.", "confidence": 0.89}
            ]
        },
        {
            "file_name": "insurance_health_claim_03.wav",
            "scores": {
                "overall_score": 58.0,
                "flags": {"compliance_violation": False, "customer_escalated": False, "prohibited_words_used": True},
                "dimensions": {
                    "greeting_brand_identity": {"score": 8, "reason": "Proper greeting and brand disclosure."},
                    "customer_verification": {"score": 8, "reason": "Verified policy number and policyholder name."},
                    "active_listening": {"score": 6, "reason": "Agent kept talking while customer was reading hospital bill details."},
                    "empathy": {"score": 7, "reason": "Expressed sympathy for hospital admission."},
                    "clarity_of_communication": {"score": 7, "reason": "Explained cashless approval procedure."},
                    "problem_resolution": {"score": 6, "reason": "Claim document checklist emailed."},
                    "compliance_disclosures": {"score": 8, "reason": "Read standard policy clause notice."},
                    "hold_mute_handling": {"score": 7, "reason": "Standard hold procedure followed."},
                    "objection_handling": {"score": 5, "reason": "Gave verbal guarantee not backed by policy terms."},
                    "closing": {"score": 7, "reason": "Provided claim ticket number."}
                },
                "summary": "Claimant inquired about cashless hospitalization claim. Agent gave helpful instructions but uttered unauthorized promise 'mera guarantee hai pass ho jayega' to pacify customer."
            },
            "transcript": [
                {"start": 0.0, "end": 4.5, "speaker": "Agent", "text": "Welcome to ShieldCare Health Insurance. My name is Priya, how can I support you?", "confidence": 0.96},
                {"start": 4.8, "end": 9.5, "speaker": "Customer", "text": "My father is admitted at Apollo for knee surgery. Cashless desk says pre-auth is pending.", "confidence": 0.93},
                {"start": 10.0, "end": 14.5, "speaker": "Agent", "text": "Please provide your policy number sir.", "confidence": 0.95},
                {"start": 15.0, "end": 17.5, "speaker": "Customer", "text": "Policy is POL-448102.", "confidence": 0.95},
                {"start": 18.0, "end": 24.0, "speaker": "Agent", "text": "Checking the portal sir. Don't worry at all, mera guarantee hai pass ho jayega by evening.", "confidence": 0.91}
            ]
        }
    ]
    
    for sc in sample_calls:
        save_call_record(sc, db_path)


if __name__ == "__main__":
    init_db()
    seed_sample_records()
    print("Database initialized and seeded.")
    print("Stats:", get_summary_stats())

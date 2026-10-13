"""
tests/test_pipeline.py - Automated test suite for the Call QA system.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rubric import DEFAULT_RUBRIC, DEFAULT_CATEGORICAL_FLAGS, format_rubric_for_prompt
from transcribe import transcribe_with_speakers, format_transcript_for_llm, get_mock_sample_transcript
from score import score_transcript
import db


def test_rubric_dimensions():
    """Verify all 10 QA dimensions and 3 flags are properly registered."""
    assert len(DEFAULT_RUBRIC) == 10, f"Expected 10 rubric dimensions, got {len(DEFAULT_RUBRIC)}"
    assert len(DEFAULT_CATEGORICAL_FLAGS) == 3, f"Expected 3 flags, got {len(DEFAULT_CATEGORICAL_FLAGS)}"
    prompt = format_rubric_for_prompt()
    assert "greeting_brand_identity" in prompt
    assert "compliance_violation" in prompt
    print("[PASS] test_rubric_dimensions")


def test_transcription_and_formatter():
    """Verify transcript formatting and fallback transcription."""
    sample = get_mock_sample_transcript()
    assert len(sample) > 0
    formatted = format_transcript_for_llm(sample)
    assert "[Agent]" in formatted or "Agent" in formatted
    assert "[Customer]" in formatted or "Customer" in formatted
    
    # Test reading existing sample call
    wav_path = Path("samples/sample_call.wav")
    if wav_path.exists():
        res = transcribe_with_speakers(str(wav_path))
        assert len(res) > 0
        assert "speaker" in res[0]
        assert "text" in res[0]
    print("[PASS] test_transcription_and_formatter")


def test_scoring_pipeline():
    """Verify scoring logic returns structured schema."""
    sample = get_mock_sample_transcript()
    result = score_transcript(sample)
    
    assert "overall_score" in result
    assert "dimensions" in result
    assert "flags" in result
    assert "compliance_violation" in result["flags"]
    assert "customer_escalated" in result["flags"]
    assert "prohibited_words_used" in result["flags"]
    
    # Check that dimensions have scores
    dims = result["dimensions"]
    for dim_key in DEFAULT_RUBRIC.keys():
        assert dim_key in dims, f"Missing dimension {dim_key} in result"
    print("[PASS] test_scoring_pipeline")


def test_database_operations():
    """Verify SQLite database storage, queries, and CSV conversion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_callqa.db"
        db.init_db(test_db)
        
        sample = get_mock_sample_transcript()
        scores = score_transcript(sample)
        
        record = {
            "file_name": "test_audio.wav",
            "scores": scores,
            "transcript": sample
        }
        
        call_id = db.save_call_record(record, test_db)
        assert call_id > 0
        
        all_calls = db.get_all_calls(test_db)
        assert len(all_calls) == 1
        assert all_calls[0]["file_name"] == "test_audio.wav"
        
        stats = db.get_summary_stats(test_db)
        assert stats["total_calls"] == 1
        assert stats["avg_score"] > 0
        
        df = db.get_calls_dataframe(test_db)
        assert len(df) == 1
        assert "Overall Score" in df.columns
        print("[PASS] test_database_operations")


if __name__ == "__main__":
    test_rubric_dimensions()
    test_transcription_and_formatter()
    test_scoring_pipeline()
    test_database_operations()
    print("\n[SUCCESS] ALL PIPELINE TESTS PASSED!")

"""
run.py - End-to-end pipeline: Audio file in -> Scored JSON out.
Usage:
    python run.py samples/sample_call.wav [--save-db] [--output result.json]
"""

import sys
import json
import argparse
from pathlib import Path

from transcribe import transcribe_with_speakers
from score import score_transcript


def process_call(audio_path: str, save_to_db: bool = False, output_file: str = None) -> dict:
    """Processes a single audio call end-to-end and returns structured results."""
    audio_file = Path(audio_path)
    print(f"[run.py] Processing call: {audio_file.name}")
    
    # 1. Transcribe with speaker diarization
    transcript = transcribe_with_speakers(str(audio_file))
    
    # 2. Score transcript against the 10-dimension rubric
    scores = score_transcript(transcript)
    
    result = {
        "file_name": audio_file.name,
        "audio_path": str(audio_file.resolve()) if audio_file.exists() else str(audio_file),
        "transcript": transcript,
        "scores": scores
    }
    
    # Optional DB storage
    if save_to_db:
        try:
            from db import save_call_record
            call_id = save_call_record(result)
            result["call_id"] = call_id
            print(f"[run.py] Saved to database with Call ID: {call_id}")
        except Exception as e:
            print(f"[run.py] DB save warning: {e}")
            
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[run.py] Output written to {output_file}")
        
    return result


def main():
    parser = argparse.ArgumentParser(description="Call QA: Automatic scoring for Indian contact centres.")
    parser.add_argument("audio_path", nargs="?", default="samples/sample_call.wav", help="Path to MP3 or WAV audio file")
    parser.add_argument("--save-db", action="store_true", help="Persist record to local SQLite database")
    parser.add_argument("--output", "-o", type=str, default=None, help="Save JSON results to output file")
    args = parser.parse_args()

    result = process_call(args.audio_path, save_to_db=args.save_db, output_file=args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

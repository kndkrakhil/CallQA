"""
generate_sample.py - Generates sample WAV audio and companion transcription files.
Allows immediate local testing of the Call QA pipeline.
"""

import os
import wave
import math
import struct
import json
from pathlib import Path


def generate_synthetic_wav(output_wav_path: str, duration_seconds: int = 12):
    """
    Generates a clear 16-bit PCM mono 16kHz WAV file with alternating conversational tone bursts.
    """
    sample_rate = 16000
    num_samples = int(sample_rate * duration_seconds)
    
    Path(output_wav_path).parent.mkdir(parents=True, exist_ok=True)
    
    with wave.open(output_wav_path, "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        frames = bytearray()
        for i in range(num_samples):
            t = float(i) / sample_rate
            # Alternate frequencies between Agent (~320Hz + harmonics) and Customer (~480Hz + harmonics)
            cycle = int(t) % 6
            if cycle in [0, 1, 4]:
                # Agent speaking turn
                freq = 320.0
                envelope = 0.5 * (1.0 + 0.5 * math.sin(2.0 * math.pi * 3.0 * t))
            elif cycle in [2, 3]:
                # Customer speaking turn
                freq = 460.0
                envelope = 0.6 * (1.0 + 0.4 * math.sin(2.0 * math.pi * 4.0 * t))
            else:
                freq = 0.0
                envelope = 0.05
            
            if freq > 0:
                sample_val = 0.6 * math.sin(2.0 * math.pi * freq * t) + 0.3 * math.sin(2.0 * math.pi * (freq * 2) * t)
            else:
                sample_val = 0.0
                
            val = int(sample_val * envelope * 24000.0)
            val = max(-32768, min(32767, val))
            frames.extend(struct.pack("<h", val))
            
        wav_file.writeframes(frames)
    print(f"[generate_sample] Created sample audio at: {output_wav_path}")


def create_companion_transcript(output_json_path: str):
    sample_transcript = [
        {
            "start": 0.0,
            "end": 4.5,
            "speaker": "Agent",
            "text": "Namaste, thank you for calling Revive Customer Care. My name is Rahul. How may I assist you today?",
            "confidence": 0.96
        },
        {
            "start": 5.0,
            "end": 10.2,
            "speaker": "Customer",
            "text": "Hi Rahul, I ordered an electric scooter battery 5 days ago. Tracking hasn't updated since Tuesday!",
            "confidence": 0.94
        },
        {
            "start": 10.8,
            "end": 15.5,
            "speaker": "Agent",
            "text": "I truly understand your concern sir and apologize for the delay. May I confirm your registered mobile number and order ID for verification?",
            "confidence": 0.95
        },
        {
            "start": 16.0,
            "end": 20.0,
            "speaker": "Customer",
            "text": "Yes, mobile number is 9876543210 and Order ID is RV-8821.",
            "confidence": 0.95
        },
        {
            "start": 20.5,
            "end": 26.0,
            "speaker": "Agent",
            "text": "Thank you for the details. Please allow me 1 minute while I check the transit status with our logistics dispatch team.",
            "confidence": 0.93
        },
        {
            "start": 27.0,
            "end": 33.5,
            "speaker": "Agent",
            "text": "Thank you for holding. The package has cleared Pune logistics center and is on priority express delivery scheduled for tomorrow by 3 PM.",
            "confidence": 0.92
        },
        {
            "start": 34.0,
            "end": 38.5,
            "speaker": "Customer",
            "text": "Okay, that is a big relief. Can you send me the tracking SMS?",
            "confidence": 0.94
        },
        {
            "start": 39.0,
            "end": 45.0,
            "speaker": "Agent",
            "text": "Certainly sir, I have triggered an SMS notification to your number. Is there anything else I may help you with today? Note this call was recorded for quality assurance.",
            "confidence": 0.96
        },
        {
            "start": 45.5,
            "end": 48.0,
            "speaker": "Customer",
            "text": "No, that's all. Thank you Rahul.",
            "confidence": 0.93
        },
        {
            "start": 48.5,
            "end": 51.5,
            "speaker": "Agent",
            "text": "Thank you for choosing Revive. Have a wonderful day ahead!",
            "confidence": 0.97
        }
    ]
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(sample_transcript, f, indent=2)
    print(f"[generate_sample] Created companion transcript at: {output_json_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    wav_target = base_dir / "sample_call.wav"
    json_target = base_dir / "sample_call.json"
    generate_synthetic_wav(str(wav_target), duration_seconds=15)
    create_companion_transcript(str(json_target))

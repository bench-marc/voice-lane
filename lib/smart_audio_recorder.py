#!/usr/bin/env python3
"""
Smart Audio Recorder with Voice Activity Detection
Similar to Verbi's approach but optimized for hotel booking conversations
"""

import speech_recognition as sr
import wave
import tempfile
import os
import sys
import json
from pathlib import Path

class SmartAudioRecorder:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
        # Voice activity detection parameters (similar to Verbi)
        self.recognizer.energy_threshold = 300  # Minimum audio energy to consider for recording
        self.recognizer.dynamic_energy_threshold = True  # Automatically adjust for ambient noise
        self.recognizer.pause_threshold = 1.0  # Seconds of non-speaking audio before a phrase is considered complete
        self.recognizer.phrase_threshold = 0.3  # Minimum seconds of speaking audio before we consider the speaking audio a phrase
        
        # Recording limits
        self.timeout = 5.0  # Seconds to wait for a phrase to start
        self.phrase_time_limit = 30.0  # Maximum seconds for a single phrase
        self.calibration_duration = 1.0  # Seconds for ambient noise calibration
        
    def calibrate_for_ambient_noise(self):
        """Calibrate the recognizer for ambient noise"""
        try:
            with sr.Microphone() as source:
                print("🎤 Calibrating for ambient noise... Please be quiet for a moment.")
                self.recognizer.adjust_for_ambient_noise(source, duration=self.calibration_duration)
                print(f"✅ Calibration complete. Energy threshold: {self.recognizer.energy_threshold}")
        except Exception as e:
            print(f"❌ Calibration failed: {e}")
            
    def record_until_silence(self, output_file=None):
        """
        Record audio until the speaker stops talking (using voice activity detection)
        Returns the path to the recorded audio file
        """
        if not output_file:
            # Create temporary file with .wav extension
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            output_file = temp_file.name
            temp_file.close()
        
        try:
            with sr.Microphone() as source:
                print("🎤 Listening... Speak now!")
                
                # Record audio until silence is detected
                audio_data = self.recognizer.listen(
                    source, 
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit
                )
                
                print("✅ Recording complete (silence detected)")
                
                # Save the audio data to WAV file
                with open(output_file, "wb") as f:
                    f.write(audio_data.get_wav_data())
                
                return output_file
                
        except sr.WaitTimeoutError:
            print("❌ No speech detected within timeout period")
            return None
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return None

def main():
    """CLI interface for the smart audio recorder"""
    if len(sys.argv) < 2:
        print("Usage: python smart_audio_recorder.py <command> [output_file]")
        print("Commands:")
        print("  calibrate - Calibrate for ambient noise")
        print("  record [file] - Record until silence detected")
        sys.exit(1)
    
    command = sys.argv[1]
    recorder = SmartAudioRecorder()
    
    if command == "calibrate":
        recorder.calibrate_for_ambient_noise()
        
    elif command == "record":
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        result_file = recorder.record_until_silence(output_file)
        
        if result_file:
            print(json.dumps({"status": "success", "file": result_file}))
        else:
            print(json.dumps({"status": "error", "message": "Recording failed"}))
            sys.exit(1)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
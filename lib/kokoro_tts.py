#!/usr/bin/env python3
"""
Kokoro TTS Wrapper for Ruby Voice Agent
Provides high-quality text-to-speech generation using the Kokoro-82M model
"""

import sys
import json
import tempfile
import os
from pathlib import Path

try:
    from kokoro import KPipeline
    import soundfile as sf
    import numpy as np
except ImportError as e:
    print(json.dumps({"status": "error", "message": f"Missing dependency: {e}"}))
    sys.exit(1)


class KokoroTTS:
    def __init__(self):
        """Initialize Kokoro TTS with American English pipeline"""
        try:
            # Initialize pipeline for American English
            self.pipeline = KPipeline(lang_code='a')
            
            # Available professional voices for business calls
            self.professional_voices = {
                'male': ['am_adam', 'am_michael'],  
                'female': ['af_sarah', 'af_bella', 'af_alloy'],
                'neutral': ['af_heart']  # Often good for professional tone
            }
            
        except Exception as e:
            print(json.dumps({"status": "error", "message": f"Failed to initialize Kokoro: {e}"}))
            sys.exit(1)
    
    def generate_speech(self, text, voice='af_sarah', speed=1.0, output_file=None):
        """
        Generate speech from text using Kokoro TTS
        
        Args:
            text (str): Text to convert to speech
            voice (str): Voice to use (default: af_sarah - professional female)
            speed (float): Speech speed multiplier (default: 1.0)
            output_file (str): Path to save audio file (optional)
        
        Returns:
            dict: Status and file path of generated audio
        """
        try:
            # Clean the text
            text = text.strip()
            if not text:
                return {"status": "error", "message": "Empty text provided"}
            
            # Generate audio using Kokoro pipeline
            generator = self.pipeline(text, voice=voice)
            
            # Collect all audio chunks
            audio_chunks = []
            for i, (graphemes, phonemes, audio) in enumerate(generator):
                if audio is not None:
                    audio_chunks.append(audio)
            
            if not audio_chunks:
                return {"status": "error", "message": "No audio generated"}
            
            # Concatenate all audio chunks
            full_audio = np.concatenate(audio_chunks)
            
            # Apply speed adjustment if needed
            if speed != 1.0:
                # Simple speed adjustment by resampling
                # Note: This is a basic implementation, more sophisticated methods exist
                original_length = len(full_audio)
                new_length = int(original_length / speed)
                if new_length > 0:
                    indices = np.linspace(0, original_length - 1, new_length)
                    full_audio = np.interp(indices, np.arange(original_length), full_audio)
            
            # Create output file if not specified
            if not output_file:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                output_file = temp_file.name
                temp_file.close()
            
            # Save audio to file (Kokoro outputs at 24kHz)
            sf.write(output_file, full_audio, 24000)
            
            return {
                "status": "success", 
                "file": output_file,
                "sample_rate": 24000,
                "voice": voice,
                "duration": len(full_audio) / 24000
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Speech generation failed: {e}"}
    
    def list_voices(self):
        """List available professional voices"""
        return {
            "status": "success",
            "voices": self.professional_voices
        }


def main():
    """CLI interface for Kokoro TTS"""
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "error", 
            "message": "Usage: python kokoro_tts.py <command> [args...]",
            "commands": {
                "generate <text> [voice] [speed] [output_file]": "Generate speech from text",
                "voices": "List available voices"
            }
        }))
        sys.exit(1)
    
    command = sys.argv[1]
    tts = KokoroTTS()
    
    if command == "generate":
        if len(sys.argv) < 3:
            print(json.dumps({"status": "error", "message": "Text required for generation"}))
            sys.exit(1)
        
        text = sys.argv[2]
        voice = sys.argv[3] if len(sys.argv) > 3 else 'af_sarah'
        speed = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        output_file = sys.argv[5] if len(sys.argv) > 5 else None
        
        result = tts.generate_speech(text, voice, speed, output_file)
        print(json.dumps(result))
        
    elif command == "voices":
        result = tts.list_voices()
        print(json.dumps(result))
        
    else:
        print(json.dumps({"status": "error", "message": f"Unknown command: {command}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
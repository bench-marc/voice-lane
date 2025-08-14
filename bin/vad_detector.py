#!/usr/bin/env python3
import webrtcvad
import sys
import wave
import os

def detect_speech(audio_file):
    if not os.path.exists(audio_file):
        return False
        
    vad = webrtcvad.Vad(2)
    
    try:
        with wave.open(audio_file, 'rb') as wf:
            if wf.getnchannels() != 1:
                return False  # VAD requires mono audio
            if wf.getsampwidth() != 2:
                return False  # VAD requires 16-bit samples
            if wf.getframerate() not in [8000, 16000, 32000, 48000]:
                return False  # VAD requires specific sample rates
                
            frames = wf.readframes(wf.getnframes())
            
        # Check multiple 20ms chunks for speech
        frame_rate = wf.getframerate()
        frame_duration = 20  # ms
        frame_size = int(frame_rate * frame_duration / 1000) * 2  # 16-bit = 2 bytes per sample
        
        speech_frames = 0
        total_frames = 0
        
        for start in range(0, len(frames) - frame_size, frame_size):
            chunk = frames[start:start + frame_size]
            if len(chunk) == frame_size:
                total_frames += 1
                if vad.is_speech(chunk, frame_rate):
                    speech_frames += 1
        
        # Consider it speech if more than 30% of frames contain speech
        if total_frames > 0:
            speech_ratio = speech_frames / total_frames
            return speech_ratio > 0.3
        
    except Exception as e:
        # If VAD fails, check file size as a simple heuristic
        file_size = os.path.getsize(audio_file)
        return file_size > 1000  # Assume files larger than 1KB contain speech
        
    return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = detect_speech(sys.argv[1])
        print("speech" if result else "silence")
    else:
        print("silence")
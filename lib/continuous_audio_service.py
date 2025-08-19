#!/usr/bin/env python3
"""
Continuous Audio Monitoring Service
Real-time voice activity detection with speech buffering and queue management
"""

import pyaudio
import wave
import threading
import queue
import time
import json
import sys
import tempfile
import os
from collections import deque
import numpy as np

try:
    import webrtcvad
except ImportError:
    print("webrtcvad not available, using energy-based VAD")
    webrtcvad = None

try:
    import speech_recognition as sr
except ImportError:
    print("speech_recognition not available")
    sr = None


class ContinuousAudioService:
    def __init__(self):
        # Audio configuration
        self.sample_rate = 16000  # 16kHz for optimal VAD performance
        self.channels = 1
        self.chunk_size = 320  # 20ms at 16kHz (optimal for WebRTC VAD)
        self.format = pyaudio.paInt16
        
        # Voice Activity Detection
        self.vad_aggressiveness = 1  # 0-3, higher = more aggressive filtering (reduced for longer sentences)
        self.vad = webrtcvad.Vad(self.vad_aggressiveness) if webrtcvad else None
        
        # Speech detection parameters
        self.speech_threshold = 0.5  # Minimum speech probability
        self.silence_duration = 2.0  # Seconds of silence to end speech (increased)
        self.min_speech_duration = 0.5  # Minimum speech length to process
        self.max_speech_duration = 30.0  # Maximum speech length
        
        # Audio buffers
        self.audio_buffer = deque(maxlen=int(self.sample_rate * 2))  # 2-second circular buffer
        self.speech_frames = []
        self.speech_queue = queue.Queue()
        self.buffered_speech = queue.Queue()  # For speech during agent talking
        
        # State management  
        self.listening = False
        self.recording_speech = False
        self.agent_speaking = False
        self.last_speech_time = 0
        self.speech_start_time = 0
        
        # Threading
        self.audio_thread = None
        self.processing_thread = None
        self.pyaudio_instance = None
        self.audio_stream = None
        
        # Statistics
        self.total_chunks = 0
        self.speech_chunks = 0
        self.processed_utterances = 0
        
        print("🎤 Continuous Audio Service initialized")
        if self.vad:
            print(f"✅ WebRTC VAD enabled (aggressiveness: {self.vad_aggressiveness})")
        else:
            print("⚠️ Using energy-based VAD (WebRTC VAD not available)")
    
    def start_listening(self):
        """Start continuous audio monitoring"""
        if self.listening:
            return
            
        self.listening = True
        print("🎤 Starting continuous audio monitoring...")
        
        # Initialize PyAudio
        self.pyaudio_instance = pyaudio.PyAudio()
        
        try:
            # Open audio stream
            self.audio_stream = self.pyaudio_instance.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # Start audio capture thread
            self.audio_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
            self.audio_thread.start()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            print("✅ Continuous audio monitoring started")
            
        except Exception as e:
            print(f"❌ Failed to start audio monitoring: {e}")
            self.stop_listening()
            
    def stop_listening(self):
        """Stop continuous audio monitoring"""
        if not self.listening:
            return
            
        self.listening = False
        print("🛑 Stopping continuous audio monitoring...")
        
        # Stop audio stream
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            
        # Wait for threads to finish
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2)
            
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
            
        print("✅ Audio monitoring stopped")
        
    def set_agent_speaking(self, speaking):
        """Update agent speaking state"""
        self.agent_speaking = speaking
        if speaking:
            print("🤖 Agent started speaking - buffering hotel speech")
        else:
            print("🎤 Agent finished speaking - processing buffered speech")
            self._process_buffered_speech()
    
    def get_next_speech(self, timeout=None):
        """Get the next detected speech utterance"""
        try:
            return self.speech_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_status(self):
        """Get service status and statistics"""
        return {
            "listening": self.listening,
            "recording_speech": self.recording_speech,
            "agent_speaking": self.agent_speaking,
            "total_chunks": self.total_chunks,
            "speech_chunks": self.speech_chunks,
            "processed_utterances": self.processed_utterances,
            "queue_size": self.speech_queue.qsize(),
            "buffered_size": self.buffered_speech.qsize()
        }
    
    def _audio_capture_loop(self):
        """Continuous audio capture loop"""
        print("🔄 Audio capture thread started")
        
        while self.listening:
            try:
                # Read audio chunk
                audio_data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                
                # Add to circular buffer
                self.audio_buffer.extend(audio_np)
                self.total_chunks += 1
                
                # Perform voice activity detection
                is_speech = self._detect_speech(audio_data, audio_np)
                
                # Handle speech detection
                self._handle_speech_detection(is_speech, audio_np)
                
            except Exception as e:
                if self.listening:  # Only log if we're supposed to be listening
                    print(f"Audio capture error: {e}")
                    time.sleep(0.1)  # Brief pause to prevent spam
                    
        print("🔄 Audio capture thread ended")
    
    def _detect_speech(self, audio_data, audio_np):
        """Detect speech in audio chunk"""
        try:
            if self.vad and len(audio_data) == self.chunk_size * 2:  # 16-bit samples
                # Use WebRTC VAD (more reliable)
                return self.vad.is_speech(audio_data, self.sample_rate)
            else:
                # Fallback to energy-based detection
                energy = np.mean(np.abs(audio_np))
                return energy > 500  # Adjust threshold as needed
                
        except Exception as e:
            # Fallback to energy-based detection on error
            energy = np.mean(np.abs(audio_np))
            return energy > 500
    
    def _handle_speech_detection(self, is_speech, audio_np):
        """Handle speech detection results"""
        current_time = time.time()
        
        if is_speech:
            self.speech_chunks += 1
            self.last_speech_time = current_time
            
            # Start recording if not already
            if not self.recording_speech:
                self.recording_speech = True
                self.speech_start_time = current_time
                self.speech_frames = []
                print("🗣️ Speech detected - starting recording")
            
            # Add audio to speech buffer
            self.speech_frames.extend(audio_np)
            
            # Check for maximum speech duration
            if current_time - self.speech_start_time > self.max_speech_duration:
                print("⚠️ Maximum speech duration reached - processing utterance")
                self._finalize_speech_recording()
        
        else:
            # Check if we should end speech recording
            if self.recording_speech:
                silence_duration = current_time - self.last_speech_time
                
                if silence_duration >= self.silence_duration:
                    # End speech recording
                    speech_duration = current_time - self.speech_start_time
                    
                    if speech_duration >= self.min_speech_duration:
                        print(f"✅ Speech ended after {silence_duration:.1f}s silence - {speech_duration:.1f}s duration")
                        self._finalize_speech_recording()
                    else:
                        print(f"❌ Speech too short ({speech_duration:.1f}s) - discarding")
                        self.recording_speech = False
                        self.speech_frames = []
    
    def _finalize_speech_recording(self):
        """Finalize speech recording and queue for processing"""
        if not self.speech_frames:
            self.recording_speech = False
            return
            
        try:
            # Convert to numpy array
            speech_audio = np.array(self.speech_frames, dtype=np.int16)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.close()
            
            # Write WAV file
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(speech_audio.tobytes())
            
            # Create speech object
            speech_data = {
                "file": temp_file.name,
                "timestamp": time.time(),
                "duration": len(speech_audio) / self.sample_rate,
                "agent_was_speaking": self.agent_speaking
            }
            
            # Queue based on agent state
            if self.agent_speaking:
                self.buffered_speech.put(speech_data)
                print(f"📦 Speech buffered (agent speaking): {speech_data['duration']:.1f}s")
            else:
                self.speech_queue.put(speech_data)
                print(f"✉️ Speech queued for processing: {speech_data['duration']:.1f}s")
            
            self.processed_utterances += 1
            
        except Exception as e:
            print(f"❌ Error finalizing speech recording: {e}")
        
        finally:
            self.recording_speech = False
            self.speech_frames = []
    
    def _process_buffered_speech(self):
        """Process any buffered speech when agent stops speaking"""
        while not self.buffered_speech.empty():
            try:
                speech_data = self.buffered_speech.get_nowait()
                self.speech_queue.put(speech_data)
                print(f"📤 Moved buffered speech to processing queue: {speech_data['duration']:.1f}s")
            except queue.Empty:
                break
    
    def _processing_loop(self):
        """Speech processing loop - now just monitors, doesn't consume"""
        print("🔄 Speech processing thread started")
        
        while self.listening:
            try:
                # Just sleep and let the HTTP endpoint handle speech retrieval
                # This thread is now just for monitoring
                time.sleep(1.0)
                
                # Optional: Log queue status
                if not self.speech_queue.empty():
                    print(f"🎯 {self.speech_queue.qsize()} speech items ready for HTTP retrieval")
                
            except Exception as e:
                print(f"Processing monitor error: {e}")
                
        print("🔄 Speech processing thread ended")


class AudioServiceHTTPServer:
    """HTTP server interface for the continuous audio service"""
    
    def __init__(self, host='127.0.0.1', port=8766):
        self.host = host
        self.port = port
        self.service = ContinuousAudioService()
        self.server = None
        
    def start_server(self):
        """Start HTTP server"""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import urllib.parse
            
            class AudioServiceHandler(BaseHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                
                def do_GET(self):
                    if self.path == '/status':
                        self.send_json_response(self.server.service.get_status())
                    elif self.path == '/speech':
                        speech = self.server.service.get_next_speech(timeout=0.01)  # Very short timeout
                        if speech:
                            self.send_json_response({"status": "success", "speech": speech})
                        else:
                            self.send_json_response({"status": "no_speech"})
                    else:
                        self.send_error(404)
                
                def do_POST(self):
                    if self.path == '/start':
                        self.server.service.start_listening()
                        self.send_json_response({"status": "started"})
                    elif self.path == '/stop':
                        self.server.service.stop_listening()
                        self.send_json_response({"status": "stopped"})
                    elif self.path == '/agent_speaking':
                        content_length = int(self.headers.get('Content-Length', 0))
                        post_data = self.rfile.read(content_length)
                        data = json.loads(post_data.decode('utf-8'))
                        self.server.service.set_agent_speaking(data.get('speaking', False))
                        self.send_json_response({"status": "updated"})
                    else:
                        self.send_error(404)
                
                def send_json_response(self, data):
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(data).encode('utf-8'))
                
                def log_message(self, format, *args):
                    # Suppress default logging
                    pass
            
            # Create server
            self.server = HTTPServer((self.host, self.port), AudioServiceHandler)
            self.server.service = self.service
            
            print(f"🌐 Audio service HTTP server starting on {self.host}:{self.port}")
            self.server.serve_forever()
            
        except Exception as e:
            print(f"❌ Failed to start HTTP server: {e}")
    
    def stop_server(self):
        """Stop HTTP server"""
        if self.server:
            self.server.shutdown()
            self.service.stop_listening()


def main():
    """CLI interface for the continuous audio service"""
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "error", 
            "message": "Usage: python continuous_audio_service.py <command>",
            "commands": {
                "server": "Start HTTP server mode",
                "start": "Start interactive mode",
                "test": "Test audio capture for 10 seconds",
                "status": "Get service status"
            }
        }))
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "server":
        # HTTP server mode for Ruby integration
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8766
        server = AudioServiceHTTPServer(port=port)
        
        try:
            server.start_server()
        except KeyboardInterrupt:
            print("\n🛑 Server shutting down...")
        finally:
            server.stop_server()
    
    elif command == "start":
        # Interactive mode
        service = ContinuousAudioService()
        
        try:
            service.start_listening()
            
            print("Service started. Commands:")
            print("  'status' - Show status")
            print("  'agent_speaking' - Toggle agent speaking state")
            print("  'quit' - Stop service")
            
            while True:
                try:
                    user_input = input("> ").strip().lower()
                    
                    if user_input == 'quit':
                        break
                    elif user_input == 'status':
                        status = service.get_status()
                        print(json.dumps(status, indent=2))
                    elif user_input == 'agent_speaking':
                        service.set_agent_speaking(not service.agent_speaking)
                    elif user_input == 'speech':
                        speech = service.get_next_speech(timeout=1.0)
                        if speech:
                            print(f"Got speech: {speech}")
                        else:
                            print("No speech available")
                            
                except KeyboardInterrupt:
                    break
                except EOFError:
                    break
        
        finally:
            service.stop_listening()
    
    elif command == "test":
        service = ContinuousAudioService()
        service.start_listening()
        print("Testing audio capture for 10 seconds...")
        time.sleep(10)
        service.stop_listening()
        
        status = service.get_status()
        print(f"Test results: {json.dumps(status, indent=2)}")
    
    elif command == "status":
        print(json.dumps({"status": "service_not_running"}, indent=2))
    
    else:
        print(json.dumps({"status": "error", "message": f"Unknown command: {command}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
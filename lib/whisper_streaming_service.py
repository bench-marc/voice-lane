#!/usr/bin/env python3
"""
Whisper Streaming Service
Real-time speech transcription using whisper_streaming library
"""

import sys
import json
import time
import tempfile
import os
import threading
import queue
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import numpy as np
import soundfile as sf
import librosa

try:
    # Import whisper_online from the copied file
    import importlib.util
    spec = importlib.util.spec_from_file_location("whisper_online", os.path.join(os.path.dirname(__file__), "whisper_online.py"))
    whisper_online = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(whisper_online)
    
    # Import the classes we need
    FasterWhisperASR = whisper_online.FasterWhisperASR
    WhisperTimestampedASR = whisper_online.WhisperTimestampedASR
    OnlineASRProcessor = whisper_online.OnlineASRProcessor
    
    whisper_streaming_available = True
except ImportError as e:
    print(f"❌ whisper_streaming not available: {e}")
    whisper_streaming_available = False
    sys.exit(1)

try:
    import webrtcvad
    vad_available = True
except ImportError:
    print("⚠️ webrtcvad not available, using basic silence detection")
    vad_available = False


class WhisperStreamingService:
    def __init__(self, model_size='base', language='en'):
        """Initialize the streaming ASR service"""
        print(f"🎤 Initializing Whisper Streaming Service (model: {model_size}, language: {language})")
        
        self.model_size = model_size
        self.language = language
        
        # Audio configuration
        self.sample_rate = 16000
        self.chunk_duration_ms = 20  # 20ms chunks
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        
        # Initialize ASR components - simplified to avoid hanging
        try:
            print("🔄 Initializing Whisper ASR backend...")
            # Try FasterWhisperASR first
            self.asr = FasterWhisperASR(language, model_size)
            self.online_processor = OnlineASRProcessor(self.asr)
            print("✅ FasterWhisper ASR initialized successfully")
        except Exception as e:
            print(f"⚠️ FasterWhisper failed: {e}")
            try:
                print("🔄 Trying WhisperTimestamped fallback...")
                self.asr = WhisperTimestampedASR(language, model_size)
                self.online_processor = OnlineASRProcessor(self.asr)
                print("✅ WhisperTimestamped ASR initialized (fallback)")
            except Exception as e2:
                print(f"❌ All Whisper backends failed: {e2}")
                # Create a dummy processor for testing
                self.asr = None
                self.online_processor = None
                print("⚠️ Running in test mode without Whisper")
        
        # Voice Activity Detection
        if vad_available:
            self.vad = webrtcvad.Vad(1)  # Aggressiveness level 1
            print("✅ WebRTC VAD initialized")
        else:
            self.vad = None
            print("⚠️ Using energy-based VAD")
        
        # Processing state
        self.is_processing = False
        self.processing_lock = threading.Lock()
        
        # Real-time streaming state
        self.streaming_sessions = {}  # session_id -> OnlineASRProcessor
        self.session_lock = threading.Lock()
        
        print("✅ Whisper Streaming Service ready")

    def transcribe_file(self, audio_file_path, streaming=True):
        """Transcribe an audio file using streaming processing"""
        if not os.path.exists(audio_file_path):
            return {"status": "error", "message": f"Audio file not found: {audio_file_path}"}
        
        # Check if ASR is available
        if not self.asr or not self.online_processor:
            return {"status": "error", "message": "Whisper ASR not initialized - service in test mode"}
        
        try:
            with self.processing_lock:
                print(f"🔍 Processing file: {audio_file_path} (streaming: {streaming})")
                
                # Reset the online processor for new transcription
                self.online_processor = OnlineASRProcessor(self.asr)
                
                # Load audio file
                audio_data, sr = librosa.load(audio_file_path, sr=self.sample_rate, mono=True)
                duration = len(audio_data) / self.sample_rate
                print(f"📊 Audio loaded: {len(audio_data)} samples, {sr}Hz → {self.sample_rate}Hz")
                print(f"📊 Audio duration: {duration:.2f}s")
                print(f"📊 Audio energy: min={np.min(audio_data):.4f}, max={np.max(audio_data):.4f}, rms={np.sqrt(np.mean(audio_data**2)):.4f}")
                
                # Skip very short audio clips to prevent whisper errors
                if duration < 0.5:
                    print(f"⚠️ Audio too short ({duration:.2f}s), skipping transcription")
                    return {
                        "status": "success",
                        "transcription": "",
                        "duration": duration,
                        "streaming": streaming,
                        "skipped": "too_short"
                    }
                
                if streaming:
                    # Process in chunks for streaming behavior
                    transcription = self._process_audio_streaming(audio_data)
                    print(f"📝 Streaming transcription: '{transcription}'")
                    
                    # Extract text from transcription result first
                    transcription_text = self._extract_text_from_result(transcription)
                    print(f"📝 Extracted transcription text: '{transcription_text}'")
                    
                    # If streaming returns empty, try direct ASR as fallback
                    if not transcription_text or transcription_text.strip() == "":
                        print("⚠️ Streaming returned empty, trying direct ASR fallback...")
                        transcription = self._transcribe_direct(audio_data)
                        transcription_text = self._extract_text_from_result(transcription)
                        print(f"📝 Direct ASR result: '{transcription_text}'")
                else:
                    # Process entire file at once
                    print("📤 Inserting full audio chunk...")
                    self.online_processor.insert_audio_chunk(audio_data)
                    result = self.online_processor.finish()
                    print(f"📝 Batch result: {result}")
                    
                    # Handle different return types
                    if isinstance(result, tuple):
                        transcription = result[0] if result else ""
                    elif isinstance(result, str):
                        transcription = result
                    else:
                        transcription = str(result) if result else ""
                    
                    # If batch returns empty, try direct ASR as fallback
                    transcription_text = self._extract_text_from_result(transcription)
                    if not transcription_text or transcription_text.strip() == "":
                        print("⚠️ Batch processing returned empty, trying direct ASR fallback...")
                        transcription = self._transcribe_direct(audio_data)
                        transcription_text = self._extract_text_from_result(transcription)
                        print(f"📝 Direct ASR fallback result: '{transcription_text}'")
                
                # Clean up transcription - ensure we extract text from result first
                final_text = self._extract_text_from_result(transcription)
                clean_text = self._clean_transcription(final_text)
                print(f"✨ Final clean text: '{clean_text}'")
                
                return {
                    "status": "success",
                    "transcription": clean_text,
                    "duration": len(audio_data) / self.sample_rate,
                    "streaming": streaming
                }
                
        except Exception as e:
            print(f"❌ Transcription error: {str(e)}")
            import traceback
            print(f"📍 Traceback: {traceback.format_exc()}")
            return {"status": "error", "message": f"Transcription failed: {str(e)}"}

    def transcribe_chunks(self, audio_chunks, sample_rate=16000, channels=1):
        """Transcribe audio chunks in real-time"""
        # Check if ASR is available
        if not self.asr or not self.online_processor:
            return {"status": "error", "message": "Whisper ASR not initialized - service in test mode"}
            
        try:
            with self.processing_lock:
                # Reset processor for new session
                self.online_processor = OnlineASRProcessor(self.asr)
                
                # Convert chunks to numpy array
                if isinstance(audio_chunks, list):
                    audio_data = np.concatenate([np.array(chunk, dtype=np.float32) for chunk in audio_chunks])
                else:
                    audio_data = np.array(audio_chunks, dtype=np.float32)
                
                # Ensure correct sample rate
                if sample_rate != self.sample_rate:
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)
                
                # Process with streaming
                transcription = self._process_audio_streaming(audio_data)
                clean_text = self._clean_transcription(transcription)
                
                return {
                    "status": "success",
                    "transcription": clean_text,
                    "duration": len(audio_data) / self.sample_rate,
                    "chunks_processed": len(audio_chunks) if isinstance(audio_chunks, list) else 1
                }
                
        except Exception as e:
            return {"status": "error", "message": f"Chunk transcription failed: {str(e)}"}

    def start_streaming_session(self, session_id="default"):
        """Start a real-time streaming session"""
        if not self.asr:
            return {"status": "error", "message": "Whisper ASR not initialized"}
        
        with self.session_lock:
            # Create new OnlineASRProcessor for this session
            self.streaming_sessions[session_id] = OnlineASRProcessor(self.asr)
            return {"status": "success", "session_id": session_id}
    
    def stream_audio_chunk(self, session_id, audio_chunk, sample_rate=16000):
        """Process a single audio chunk in real-time streaming mode"""
        if not self.asr:
            return {"status": "error", "message": "Whisper ASR not initialized"}
        
        with self.session_lock:
            if session_id not in self.streaming_sessions:
                # Auto-create session if it doesn't exist
                self.streaming_sessions[session_id] = OnlineASRProcessor(self.asr)
            
            processor = self.streaming_sessions[session_id]
        
        try:
            # Convert audio chunk to proper format
            if isinstance(audio_chunk, list):
                audio_data = np.array(audio_chunk, dtype=np.float32)
            else:
                audio_data = np.array(audio_chunk, dtype=np.float32)
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)
            
            # Insert chunk into streaming processor
            processor.insert_audio_chunk(audio_data)
            
            # Get partial result
            partial_result = processor.process_iter()
            
            # Extract text from partial result (handle tuples)
            partial_text = self._extract_text_from_result(partial_result) if partial_result else ""
            
            return {
                "status": "success",
                "partial_text": partial_text,
                "chunk_processed": True
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Stream chunk processing failed: {str(e)}"}
    
    def finalize_streaming_session(self, session_id):
        """Finalize a streaming session and get final transcription"""
        if not self.asr:
            return {"status": "error", "message": "Whisper ASR not initialized"}
        
        with self.session_lock:
            if session_id not in self.streaming_sessions:
                return {"status": "error", "message": f"Session {session_id} not found"}
            
            processor = self.streaming_sessions[session_id]
        
        try:
            # Get final transcription
            final_result = processor.finish()
            
            # Handle different return types
            if isinstance(final_result, tuple):
                final_transcription = final_result[0] if final_result else ""
            elif isinstance(final_result, str):
                final_transcription = final_result
            else:
                final_transcription = str(final_result) if final_result else ""
            
            # Clean up session
            with self.session_lock:
                del self.streaming_sessions[session_id]
            
            return {
                "status": "success", 
                "final_transcription": self._clean_transcription(final_transcription),
                "session_id": session_id
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Session finalization failed: {str(e)}"}

    def _process_audio_streaming(self, audio_data):
        """Process audio data using streaming approach"""
        print(f"🔄 Starting streaming processing with {len(audio_data)} samples")
        
        # Process audio in small chunks for real-time behavior
        chunk_samples = int(self.sample_rate * 0.1)  # 100ms chunks
        print(f"📦 Using chunk size: {chunk_samples} samples (100ms)")
        
        partial_transcriptions = []  # Collect all partial results
        final_transcription = ""
        
        chunks_processed = 0
        for i in range(0, len(audio_data), chunk_samples):
            chunk = audio_data[i:i + chunk_samples]
            chunks_processed += 1
            
            print(f"📤 Processing chunk {chunks_processed}: {len(chunk)} samples")
            
            # Insert audio chunk
            try:
                self.online_processor.insert_audio_chunk(chunk)
                print(f"✅ Chunk {chunks_processed} inserted successfully")
            except Exception as e:
                print(f"❌ Error inserting chunk {chunks_processed}: {e}")
                continue
            
            # Get partial results
            try:
                result = self.online_processor.process_iter()
                if result:
                    # Extract text from tuple if needed: (start_time, end_time, text)
                    if isinstance(result, tuple) and len(result) >= 3:
                        text_part = result[2]  # Get the text part
                        if text_part and text_part.strip():
                            partial_transcriptions.append(text_part)
                            print(f"📝 Partial result from chunk {chunks_processed}: '{text_part}' (extracted from tuple)")
                        else:
                            print(f"⚪ Empty text in tuple from chunk {chunks_processed}: {result}")
                    elif isinstance(result, str) and result.strip():
                        partial_transcriptions.append(result)
                        print(f"📝 Partial result from chunk {chunks_processed}: '{result}' (direct string)")
                    else:
                        print(f"⚪ Unusable result from chunk {chunks_processed}: {result}")
                else:
                    print(f"⚪ No partial result from chunk {chunks_processed}")
            except Exception as e:
                print(f"❌ Error processing chunk {chunks_processed}: {e}")
        
        print(f"🏁 Processed {chunks_processed} chunks, finalizing...")
        
        # Finalize transcription
        try:
            final_result = self.online_processor.finish()
            print(f"🎯 Raw final result: {final_result} (type: {type(final_result)})")
            
            # Handle different return types from finish() - tuples are (start_time, end_time, text)
            if isinstance(final_result, tuple):
                # Extract text from position 2 in tuple (start_time, end_time, text)
                if len(final_result) >= 3 and final_result[2]:
                    final_transcription = str(final_result[2])
                    print(f"🔧 Extracted text from tuple position 2: '{final_transcription}'")
                elif len(final_result) >= 1 and final_result[0]:
                    final_transcription = str(final_result[0])
                    print(f"🔧 Extracted from tuple position 0: '{final_transcription}'")
                else:
                    final_transcription = ""
                    print(f"🔧 Empty tuple result: {final_result}")
            elif isinstance(final_result, str):
                final_transcription = final_result
                print(f"🔧 Direct string: '{final_transcription}'")
            else:
                final_transcription = str(final_result) if final_result else ""
                print(f"🔧 Converted to string: '{final_transcription}'")
        except Exception as e:
            print(f"❌ Error finalizing: {e}")
            final_transcription = ""
        
        # Combine partial results if final transcription is empty
        if final_transcription and final_transcription.strip():
            result = final_transcription
            print(f"✅ Using final transcription: '{result}'")
        elif partial_transcriptions:
            # Filter out empty partial results and combine
            valid_partials = [p for p in partial_transcriptions if p and p.strip()]
            result = " ".join(valid_partials)
            print(f"🔗 Combined {len(valid_partials)} partial results: {valid_partials} → '{result}'")
        else:
            result = ""
        
        print(f"✨ Streaming result: '{result}'")
        return result

    def _transcribe_direct(self, audio_data):
        """Direct transcription using the ASR model without OnlineASRProcessor"""
        try:
            print(f"🔄 Direct ASR transcription with {len(audio_data)} samples")
            
            # Try to use the ASR model directly
            if hasattr(self.asr, 'transcribe'):
                # Use direct transcribe method if available
                print("📤 Using direct ASR.transcribe() method")
                result = self.asr.transcribe(audio_data)
                print(f"📝 Direct transcribe result: {result} (type: {type(result)})")
                
                # Handle different result types
                if isinstance(result, dict):
                    text = result.get('text', '')
                elif isinstance(result, str):
                    text = result
                elif isinstance(result, tuple):
                    text = result[0] if result and len(result) > 0 else ""
                else:
                    text = str(result) if result else ""
                
                return text
            else:
                print("❌ ASR model has no direct transcribe method")
                return ""
                
        except Exception as e:
            print(f"❌ Direct transcription error: {e}")
            import traceback
            print(f"📍 Direct transcription traceback: {traceback.format_exc()}")
            return ""

    def _extract_text_from_result(self, result):
        """Extract text from various result formats (tuple, string, etc.)"""
        if not result:
            return ""
        
        if isinstance(result, tuple):
            # Handle tuple format: (start_time, end_time, text)
            if len(result) >= 3:
                return str(result[2]) if result[2] else ""
            elif len(result) >= 1:
                return str(result[0]) if result[0] else ""
            else:
                return ""
        elif isinstance(result, str):
            return result
        else:
            return str(result)

    def _detect_speech(self, audio_chunk):
        """Detect speech in audio chunk"""
        if self.vad and len(audio_chunk) == self.chunk_size:
            # Convert to 16-bit PCM for WebRTC VAD
            pcm_data = (audio_chunk * 32767).astype(np.int16).tobytes()
            return self.vad.is_speech(pcm_data, self.sample_rate)
        else:
            # Energy-based detection
            energy = np.mean(np.abs(audio_chunk))
            return energy > 0.01  # Adjust threshold as needed

    def _clean_transcription(self, text):
        """Clean transcription text"""
        if not text:
            return ""
        
        # Handle non-string types - extract text if it's a tuple
        if isinstance(text, tuple):
            text = self._extract_text_from_result(text)
        elif not isinstance(text, str):
            text = str(text)
        
        # Remove extra whitespace
        cleaned = ' '.join(text.split())
        
        # Remove common artifacts
        cleaned = cleaned.replace('[BLANK_AUDIO]', '')
        cleaned = cleaned.replace('[MUSIC]', '')
        cleaned = cleaned.replace('(background noise)', '')
        
        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned.strip()

    def get_status(self):
        """Get service status"""
        return {
            "status": "running",
            "model_size": self.model_size,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "vad_available": vad_available,
            "is_processing": self.is_processing
        }


class WhisperStreamingHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the streaming service"""

    def do_GET(self):
        if self.path == '/health':
            self.send_json_response({"status": "healthy", "service": "whisper_streaming"})
        elif self.path == '/status':
            self.send_json_response(self.server.service.get_status())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/transcribe_file':
            self.handle_transcribe_file()
        elif self.path == '/transcribe_chunks':
            self.handle_transcribe_chunks()
        elif self.path == '/stream_audio':
            self.handle_stream_audio()
        elif self.path == '/finalize_stream':
            self.handle_finalize_stream()
        else:
            self.send_error(404)

    def handle_transcribe_file(self):
        """Handle file transcription request"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            audio_file = data.get('audio_file')
            streaming = data.get('streaming', True)
            
            if not audio_file:
                self.send_json_response({"status": "error", "message": "No audio_file provided"})
                return
            
            result = self.server.service.transcribe_file(audio_file, streaming)
            self.send_json_response(result)
            
        except Exception as e:
            self.send_json_response({"status": "error", "message": str(e)})

    def handle_transcribe_chunks(self):
        """Handle audio chunks transcription request"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            audio_chunks = data.get('audio_chunks', [])
            sample_rate = data.get('sample_rate', 16000)
            channels = data.get('channels', 1)
            
            if not audio_chunks:
                self.send_json_response({"status": "error", "message": "No audio_chunks provided"})
                return
            
            result = self.server.service.transcribe_chunks(audio_chunks, sample_rate, channels)
            self.send_json_response(result)
            
        except Exception as e:
            self.send_json_response({"status": "error", "message": str(e)})

    def handle_stream_audio(self):
        """Handle real-time audio streaming request"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            session_id = data.get('session_id', 'default')
            audio_chunk = data.get('audio_chunk', [])
            sample_rate = data.get('sample_rate', 16000)
            
            if not audio_chunk:
                self.send_json_response({"status": "error", "message": "No audio_chunk provided"})
                return
            
            result = self.server.service.stream_audio_chunk(session_id, audio_chunk, sample_rate)
            self.send_json_response(result)
            
        except Exception as e:
            self.send_json_response({"status": "error", "message": str(e)})

    def handle_finalize_stream(self):
        """Handle stream finalization request"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            session_id = data.get('session_id', 'default')
            
            result = self.server.service.finalize_streaming_session(session_id)
            self.send_json_response(result)
            
        except Exception as e:
            self.send_json_response({"status": "error", "message": str(e)})

    def send_json_response(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


def start_server(host='127.0.0.1', port=8767, model_size='base', language='en'):
    """Start the HTTP server"""
    try:
        service = WhisperStreamingService(model_size, language)
        
        server = HTTPServer((host, port), WhisperStreamingHTTPHandler)
        server.service = service
        
        print(f"🌐 Whisper Streaming Service starting on {host}:{port}")
        print(f"📋 Model: {model_size}, Language: {language}")
        print("✅ Service ready for requests")
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n🛑 Server shutting down...")
    except Exception as e:
        print(f"❌ Server error: {e}")


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "error",
            "message": "Usage: python whisper_streaming_service.py <command>",
            "commands": {
                "server": "Start HTTP server mode",
                "test": "Test transcription with a sample file",
                "health": "Check service health"
            }
        }))
        sys.exit(1)

    command = sys.argv[1]

    if command == "server":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8767
        model_size = sys.argv[3] if len(sys.argv) > 3 else 'base'
        language = sys.argv[4] if len(sys.argv) > 4 else 'en'
        
        start_server(port=port, model_size=model_size, language=language)

    elif command == "test":
        if len(sys.argv) < 3:
            print("Usage: python whisper_streaming_service.py test <audio_file>")
            sys.exit(1)
        
        audio_file = sys.argv[2]
        service = WhisperStreamingService()
        result = service.transcribe_file(audio_file)
        print(json.dumps(result, indent=2))

    elif command == "health":
        try:
            import requests
            response = requests.get('http://127.0.0.1:8767/health')
            print(json.dumps(response.json(), indent=2))
        except:
            print(json.dumps({"status": "service_not_running"}))

    else:
        print(json.dumps({"status": "error", "message": f"Unknown command: {command}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
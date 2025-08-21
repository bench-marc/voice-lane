#!/usr/bin/env python3
"""
Streaming Audio Service
High-performance real-time audio capture with silence-based segmentation
Optimized for speed with minimal latency using existing Whisper infrastructure
"""

import sys
import json
import time
import tempfile
import os
import threading
import queue
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import soundfile as sf
from typing import Optional, Callable, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import pyaudio
    audio_capture_available = True
    logger.info("✅ PyAudio available for real-time capture")
except ImportError:
    logger.warning("⚠️ PyAudio not available, file-based processing only")
    audio_capture_available = False

try:
    import webrtcvad
    vad_available = True
    logger.info("✅ WebRTC VAD available")
except ImportError:
    logger.warning("⚠️ webrtcvad not available, using energy-based VAD")
    vad_available = False

try:
    # Import whisper processing from existing infrastructure
    import subprocess
    whisper_available = True
    logger.info("✅ Whisper available via subprocess")
except ImportError:
    logger.error("❌ Whisper not available")
    whisper_available = False


class StreamingAudioService:
    """
    Real-time streaming audio capture with silence-based transcription
    Optimized for minimal latency using existing Whisper setup
    """
    
    def __init__(self, model_name='tiny', language=None, whisper_path=None):
        """Initialize streaming audio service with focus on speed"""
        logger.info(f"🎤 Initializing Streaming Audio Service")
        logger.info(f"📋 Model: {model_name}, Language: {language or 'auto-detect'}")
        
        self.model_name = model_name
        self.language = language
        self.whisper_path = whisper_path or os.path.expanduser('~/.local/bin/whisper')
        
        # Audio configuration optimized for low latency
        self.sample_rate = 16000  # Standard Whisper rate
        self.channels = 1
        self.chunk_duration_ms = 50   # 50ms chunks for very low latency
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        
        # Silence detection configuration - tuned for fast response  
        self.silence_threshold_ms = 800   # 0.8 seconds silence to trigger transcription
        self.min_audio_duration_ms = 200  # Minimum audio to process (200ms)
        self.max_audio_duration_ms = 10000 # Maximum audio before forced processing
        
        # Voice Activity Detection setup
        if vad_available:
            self.vad = webrtcvad.Vad(1)  # Moderate aggressiveness for speed
            logger.info("✅ WebRTC VAD initialized (aggressiveness: 1)")
        else:
            self.vad = None
            logger.info("⚠️ Using energy-based VAD fallback")
        
        # Streaming state
        self.is_streaming = False
        self.audio_buffer = []
        self.last_speech_time = 0
        self.speech_start_time = 0
        self.processing_lock = threading.Lock()
        
        # Real-time capture setup
        if audio_capture_available:
            self.pyaudio_instance = None
            self.audio_stream = None
            self.capture_thread = None
            self.processing_queue = queue.Queue(maxsize=10)  # Limit queue size for low latency
            
        # Transcription results queue for Ruby client polling
        self.transcription_results = queue.Queue(maxsize=20)
        self.result_queue_lock = threading.Lock()
            
        # Statistics
        self.total_processed = 0
        self.total_duration_processed = 0.0
        self.avg_processing_time = 0.0
        
        logger.info("✅ Streaming Audio Service ready")
    
    def start_realtime_capture(self, callback: Callable[[str, float], None]):
        """Start real-time audio capture and streaming transcription"""
        if not audio_capture_available:
            raise RuntimeError("PyAudio not available for real-time capture")
        
        if self.is_streaming:
            logger.warning("⚠️ Already streaming")
            return
        
        logger.info("🎤 Starting real-time audio capture...")
        
        self.is_streaming = True
        self.audio_buffer = []
        self.callback = callback
        
        # Initialize PyAudio
        self.pyaudio_instance = pyaudio.PyAudio()
        
        try:
            # Open audio stream with minimal latency settings
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=None,  # Use default microphone
            )
            
            # Start capture thread
            self.capture_thread = threading.Thread(
                target=self._capture_loop, 
                daemon=True
            )
            self.capture_thread.start()
            
            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True
            )
            self.processing_thread.start()
            
            logger.info("✅ Real-time capture started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start audio capture: {e}")
            self.stop_realtime_capture()
            raise
    
    def stop_realtime_capture(self):
        """Stop real-time audio capture"""
        if not self.is_streaming:
            return
        
        logger.info("🛑 Stopping real-time capture...")
        
        self.is_streaming = False
        
        # Process any remaining audio
        if self.audio_buffer:
            self._queue_audio_for_processing(force=True)
        
        # Stop audio stream
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
        
        # Wait for threads
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
            
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        
        logger.info("✅ Real-time capture stopped")
    
    def _capture_loop(self):
        """Main capture loop for real-time audio processing"""
        logger.info("🔄 Audio capture thread started")
        
        while self.is_streaming:
            try:
                # Read audio chunk
                audio_data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                
                # Detect speech in this chunk
                is_speech = self._detect_speech_in_chunk(audio_data, audio_np)
                
                current_time = time.time() * 1000  # milliseconds
                
                if is_speech:
                    # Add to buffer and update speech timing
                    self.audio_buffer.extend(audio_np)
                    self.last_speech_time = current_time
                    
                    # Start speech timing if this is the beginning
                    if self.speech_start_time == 0:
                        self.speech_start_time = current_time
                        logger.debug("🗣️ Speech started")
                
                else:
                    # Check if we should process accumulated speech
                    if self.audio_buffer:
                        silence_duration = current_time - self.last_speech_time
                        total_duration = current_time - self.speech_start_time
                        
                        # Process if silence threshold reached or max duration exceeded
                        should_process = (
                            silence_duration >= self.silence_threshold_ms or 
                            total_duration >= self.max_audio_duration_ms
                        )
                        
                        if should_process and total_duration >= self.min_audio_duration_ms:
                            self._queue_audio_for_processing()
                
            except Exception as e:
                if self.is_streaming:
                    logger.error(f"Capture loop error: {e}")
                    time.sleep(0.01)  # Brief pause to prevent spam
                    
        logger.info("🔄 Audio capture thread ended")
    
    def _processing_loop(self):
        """Background thread for processing queued audio"""
        logger.info("🔄 Audio processing thread started")
        
        while self.is_streaming or not self.processing_queue.empty():
            try:
                # Get audio data from queue with timeout
                audio_data = self.processing_queue.get(timeout=0.5)
                
                if audio_data is None:  # Sentinel to stop processing
                    break
                    
                # Process the audio
                self._process_audio_data(audio_data)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                
        logger.info("🔄 Audio processing thread ended")
    
    def _detect_speech_in_chunk(self, raw_audio_bytes, audio_np):
        """Detect speech in a single audio chunk"""
        try:
            if self.vad and len(raw_audio_bytes) == self.chunk_size * 2:  # 16-bit samples
                # Use WebRTC VAD (more reliable and faster)
                return self.vad.is_speech(raw_audio_bytes, self.sample_rate)
            else:
                # Fallback to energy-based detection
                energy = np.mean(np.abs(audio_np.astype(np.float32)))
                # Lower threshold for faster response, adjusted for chunk size
                return energy > 200  # Adjust based on testing
                
        except Exception as e:
            logger.debug(f"VAD error: {e}")
            # Fallback to energy detection
            energy = np.mean(np.abs(audio_np.astype(np.float32)))
            return energy > 200
    
    def _queue_audio_for_processing(self, force=False):
        """Queue accumulated audio buffer for transcription processing"""
        if not self.audio_buffer:
            return
        
        try:
            # Convert buffer to numpy array
            audio_data = np.array(self.audio_buffer, dtype=np.int16)
            audio_duration = len(audio_data) / self.sample_rate
            
            logger.debug(f"📤 Queuing {audio_duration:.2f}s of audio for processing")
            
            # Try to add to processing queue (non-blocking)
            audio_info = {
                'audio_data': audio_data.copy(),
                'duration': audio_duration,
                'timestamp': time.time()
            }
            
            try:
                self.processing_queue.put_nowait(audio_info)
            except queue.Full:
                # Queue full - drop oldest item and add new one for low latency
                try:
                    self.processing_queue.get_nowait()
                    self.processing_queue.put_nowait(audio_info)
                    logger.warning("⚠️ Processing queue full, dropped old audio for low latency")
                except queue.Empty:
                    pass
                    
        except Exception as e:
            logger.error(f"❌ Error queuing audio: {e}")
        finally:
            # Clear buffer and reset timing
            self.audio_buffer = []
            self.speech_start_time = 0
            self.last_speech_time = 0
    
    def _process_audio_data(self, audio_info):
        """Process audio data using Whisper transcription"""
        try:
            audio_data = audio_info['audio_data']
            audio_duration = audio_info['duration']
            
            logger.debug(f"🔄 Processing {audio_duration:.2f}s of audio ({len(audio_data)} samples)")
            
            # Save to temporary WAV file for Whisper
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix='.wav',
                dir='tmp'
            )
            temp_file.close()
            
            # Write audio data to file
            sf.write(temp_file.name, audio_data, self.sample_rate)
            
            # Process using existing Whisper setup
            start_time = time.time()
            transcription = self._transcribe_with_whisper(temp_file.name)
            processing_time = time.time() - start_time
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            # Update statistics
            self.total_processed += 1
            self.total_duration_processed += audio_duration
            self.avg_processing_time = (
                (self.avg_processing_time * (self.total_processed - 1) + processing_time) / 
                self.total_processed
            )
            
            # Clean transcription
            clean_text = self._clean_transcription(transcription)
            
            logger.info(f"📝 Transcribed: '{clean_text}' ({processing_time:.2f}s processing)")
            
            # Store result for client polling
            if clean_text:
                result_data = {
                    'text': clean_text,
                    'duration': audio_duration,
                    'timestamp': time.time(),
                    'processing_time': processing_time
                }
                
                try:
                    # Store in results queue for client polling
                    with self.result_queue_lock:
                        if self.transcription_results.full():
                            # Remove oldest result if queue is full
                            try:
                                self.transcription_results.get_nowait()
                            except queue.Empty:
                                pass
                        self.transcription_results.put_nowait(result_data)
                except queue.Full:
                    logger.warning("⚠️ Results queue full, dropping transcription")
            
            # Call callback if transcription is not empty
            if clean_text and hasattr(self, 'callback') and self.callback:
                self.callback(clean_text, audio_duration)
                
        except Exception as e:
            logger.error(f"❌ Audio processing error: {e}")
    
    def _transcribe_with_whisper(self, audio_file_path):
        """Transcribe audio file using existing Whisper setup"""
        try:
            # Build Whisper command with optimizations for speed
            cmd = [
                self.whisper_path,
                audio_file_path,
                '--model', self.model_name,
                '--task', 'transcribe',
                '--output_format', 'json',
                '--output_dir', 'tmp/',
                '--fp16', 'False',
                '--threads', '4',
                '--best_of', '1',      # Fastest processing
                '--beam_size', '1',    # Fastest processing  
                '--temperature', '0'   # Most deterministic
            ]
            
            # Add language if specified
            if self.language:
                cmd.extend(['--language', self.language])
            
            # Run Whisper
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10  # 10 second timeout for responsiveness
            )
            
            if result.returncode == 0:
                # Parse JSON result
                json_file = os.path.join('tmp', os.path.basename(audio_file_path).replace('.wav', '.json'))
                if os.path.exists(json_file):
                    with open(json_file, 'r') as f:
                        whisper_result = json.load(f)
                    
                    # Clean up JSON file
                    os.unlink(json_file)
                    
                    return whisper_result.get('text', '').strip()
            else:
                logger.error(f"Whisper error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Whisper timeout - audio may be too long")
        except Exception as e:
            logger.error(f"❌ Whisper transcription error: {e}")
            
        return ""
    
    def _clean_transcription(self, text):
        """Clean and normalize transcription text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        cleaned = ' '.join(text.split())
        
        # Remove common artifacts
        artifacts = ['[BLANK_AUDIO]', '[MUSIC]', '(background noise)', '(silence)', '(no speech)']
        for artifact in artifacts:
            cleaned = cleaned.replace(artifact, '')
        
        # Capitalize first letter
        cleaned = cleaned.strip()
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned
    
    def transcribe_file(self, audio_file_path):
        """Transcribe an audio file"""
        if not os.path.exists(audio_file_path):
            return {"status": "error", "message": f"Audio file not found: {audio_file_path}"}
        
        try:
            logger.info(f"🔍 Transcribing file: {audio_file_path}")
            
            # Get file duration
            info = sf.info(audio_file_path)
            duration = info.duration
            
            if duration < 0.2:
                logger.warning(f"⚠️ Audio too short ({duration:.2f}s), skipping")
                return {
                    "status": "success",
                    "transcription": "",
                    "duration": duration,
                    "processing_time": 0.0
                }
            
            # Process using Whisper
            start_time = time.time()
            transcription = self._transcribe_with_whisper(audio_file_path)
            processing_time = time.time() - start_time
            
            clean_text = self._clean_transcription(transcription)
            
            logger.info(f"✅ File transcription completed: '{clean_text}' ({processing_time:.2f}s)")
            
            return {
                "status": "success",
                "transcription": clean_text,
                "duration": duration,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ File transcription error: {e}")
            return {"status": "error", "message": f"Transcription failed: {str(e)}"}
    
    def get_transcription_results(self, timeout=0.1):
        """Get pending transcription results (non-blocking)"""
        results = []
        
        try:
            # Get all available results with a short timeout
            while True:
                try:
                    result = self.transcription_results.get(timeout=timeout)
                    results.append(result)
                    timeout = 0.01  # Reduce timeout for subsequent results
                except queue.Empty:
                    break
        except Exception as e:
            logger.debug(f"Error getting transcription results: {e}")
        
        return results

    def get_status(self):
        """Get service status and performance statistics"""
        return {
            "status": "running" if whisper_available else "degraded",
            "model_name": self.model_name,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "is_streaming": self.is_streaming,
            "whisper_available": whisper_available,
            "vad_available": vad_available,
            "audio_capture_available": audio_capture_available,
            "chunk_duration_ms": self.chunk_duration_ms,
            "silence_threshold_ms": self.silence_threshold_ms,
            "statistics": {
                "total_processed": self.total_processed,
                "total_duration_processed": self.total_duration_processed,
                "avg_processing_time": self.avg_processing_time,
                "queue_size": self.processing_queue.qsize() if hasattr(self, 'processing_queue') else 0,
                "results_queue_size": self.transcription_results.qsize()
            }
        }


class StreamingAudioHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for streaming audio service"""

    def do_GET(self):
        if self.path == '/health':
            self.send_json_response({"status": "healthy", "service": "streaming_audio"})
        elif self.path == '/status':
            self.send_json_response(self.server.service.get_status())
        elif self.path == '/results':
            # Get pending transcription results
            results = self.server.service.get_transcription_results()
            self.send_json_response({"status": "success", "results": results, "count": len(results)})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/transcribe_file':
            self.handle_transcribe_file()
        elif self.path == '/start_streaming':
            self.handle_start_streaming()
        elif self.path == '/stop_streaming':
            self.handle_stop_streaming()
        else:
            self.send_error(404)

    def handle_transcribe_file(self):
        """Handle file transcription request"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            audio_file = data.get('audio_file')
            
            if not audio_file:
                self.send_json_response({"status": "error", "message": "No audio_file provided"})
                return
            
            result = self.server.service.transcribe_file(audio_file)
            self.send_json_response(result)
            
        except Exception as e:
            self.send_json_response({"status": "error", "message": str(e)})

    def handle_start_streaming(self):
        """Handle start streaming request"""
        try:
            # Store reference to send results back
            def transcription_callback(text, duration):
                logger.info(f"🎯 Transcription ready: '{text}' ({duration:.2f}s)")
            
            self.server.service.start_realtime_capture(transcription_callback)
            self.send_json_response({"status": "streaming_started"})
            
        except Exception as e:
            self.send_json_response({"status": "error", "message": str(e)})

    def handle_stop_streaming(self):
        """Handle stop streaming request"""
        try:
            self.server.service.stop_realtime_capture()
            self.send_json_response({"status": "streaming_stopped"})
            
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


def start_server(host='127.0.0.1', port=8769, model_name='tiny', language=None):
    """Start the HTTP server for streaming audio service"""
    try:
        service = StreamingAudioService(model_name, language)
        
        server = HTTPServer((host, port), StreamingAudioHTTPHandler)
        server.service = service
        
        logger.info(f"🌐 Streaming Audio Service starting on {host}:{port}")
        logger.info(f"📋 Model: {model_name}, Language: {language or 'auto-detect'}")
        logger.info("✅ Service ready for requests")
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Server shutting down...")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")


def main():
    """CLI interface for streaming audio service"""
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "error",
            "message": "Usage: python streaming_audio_service.py <command>",
            "commands": {
                "server": "Start HTTP server mode",
                "test": "Test transcription with a sample file",
                "health": "Check service health"
            }
        }))
        sys.exit(1)

    command = sys.argv[1]

    if command == "server":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8769
        model_name = sys.argv[3] if len(sys.argv) > 3 else 'tiny'
        language = sys.argv[4] if len(sys.argv) > 4 else None
        
        start_server(port=port, model_name=model_name, language=language)

    elif command == "test":
        if len(sys.argv) < 3:
            print("Usage: python streaming_audio_service.py test <audio_file>")
            sys.exit(1)
        
        audio_file = sys.argv[2]
        service = StreamingAudioService()
        result = service.transcribe_file(audio_file)
        print(json.dumps(result, indent=2))

    elif command == "health":
        try:
            import requests
            response = requests.get('http://127.0.0.1:8769/health')
            print(json.dumps(response.json(), indent=2))
        except:
            print(json.dumps({"status": "service_not_running"}))

    else:
        print(json.dumps({"status": "error", "message": f"Unknown command: {command}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
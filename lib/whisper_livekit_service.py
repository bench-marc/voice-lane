#!/usr/bin/env python3
"""
WhisperLiveKit Streaming Service
High-performance real-time speech transcription using WhisperLiveKit
Optimized for speed over accuracy with immediate silence detection
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
import asyncio
import websockets
from typing import Optional, Callable, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from whisperlivekit import AudioProcessor, TranscriptionEngine
    whisperlivekit_available = True
    logger.info("✅ WhisperLiveKit successfully imported")
except ImportError as e:
    logger.error(f"❌ WhisperLiveKit not available: {e}")
    whisperlivekit_available = False
    sys.exit(1)

try:
    import webrtcvad
    vad_available = True
    logger.info("✅ WebRTC VAD available")
except ImportError:
    logger.warning("⚠️ webrtcvad not available, using basic silence detection")
    vad_available = False

try:
    import pyaudio
    audio_capture_available = True
    logger.info("✅ PyAudio available for real-time capture")
except ImportError:
    logger.warning("⚠️ PyAudio not available, file-based processing only")
    audio_capture_available = False


class WhisperLiveKitStreamingService:
    """
    Streaming speech-to-text service using WhisperLiveKit
    Optimized for low latency with silence-based segmentation
    """
    
    def __init__(self, model_name='tiny.en', language='en', device='cpu'):
        """Initialize streaming service with focus on speed"""
        logger.info(f"🎤 Initializing WhisperLiveKit Streaming Service")
        logger.info(f"📋 Model: {model_name}, Language: {language}, Device: {device}")
        
        self.model_name = model_name
        self.language = language 
        self.device = device
        
        # Audio configuration optimized for low latency
        self.sample_rate = 16000  # WhisperLiveKit standard
        self.channels = 1
        self.chunk_duration_ms = 20  # Very small chunks for minimal latency
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        
        # Silence detection configuration - tuned for fast response
        self.silence_threshold_ms = 1000  # 1 second silence to trigger transcription
        self.min_audio_duration_ms = 300   # Minimum audio to process (300ms)
        self.max_audio_duration_ms = 15000  # Maximum audio before forced processing
        
        # Initialize WhisperLiveKit components
        try:
            logger.info("🔄 Setting up WhisperLiveKit transcription engine...")
            self.transcription_engine = TranscriptionEngine(
                model_name=self.model_name,
                language=self.language,
                device=self.device,
                # Performance optimizations
                beam_size=1,           # Faster beam search
                vad_filter=True,       # Use VAD filtering
                vad_threshold=0.5,     # Moderate VAD sensitivity
            )
            
            logger.info("🔄 Setting up WhisperLiveKit audio processor...")
            self.audio_processor = AudioProcessor(
                transcription_engine=self.transcription_engine,
                # Low latency settings
                chunk_length_s=0.1,    # 100ms processing chunks
                stride_length_s=0.05,  # 50ms stride for overlap
            )
            
            logger.info("✅ WhisperLiveKit components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize WhisperLiveKit: {e}")
            self.transcription_engine = None
            self.audio_processor = None
        
        # Voice Activity Detection
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
            
        # Statistics
        self.total_processed = 0
        self.total_duration_processed = 0.0
        self.avg_processing_time = 0.0
        
        logger.info("✅ WhisperLiveKit Streaming Service ready")
    
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
                stream_callback=self._audio_callback
            )
            
            # Start capture thread
            self.capture_thread = threading.Thread(
                target=self._capture_loop, 
                daemon=True
            )
            self.capture_thread.start()
            
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
        
        # Stop audio stream
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
        
        # Wait for capture thread
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        # Process any remaining audio
        if self.audio_buffer:
            self._process_audio_buffer(force=True)
        
        logger.info("✅ Real-time capture stopped")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for real-time audio processing"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Detect speech in this chunk
        is_speech = self._detect_speech_in_chunk(in_data, audio_data)
        
        current_time = time.time() * 1000  # milliseconds
        
        if is_speech:
            # Add to buffer and update speech timing
            self.audio_buffer.extend(audio_data)
            self.last_speech_time = current_time
            
            # Start speech timing if this is the beginning
            if not hasattr(self, 'speech_start_time') or self.speech_start_time == 0:
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
                    # Schedule processing in separate thread to avoid blocking callback
                    threading.Thread(
                        target=self._process_audio_buffer,
                        args=(False,),
                        daemon=True
                    ).start()
        
        return (None, pyaudio.paContinue)
    
    def _capture_loop(self):
        """Main capture loop for monitoring and cleanup"""
        logger.info("🔄 Capture monitoring thread started")
        
        while self.is_streaming:
            try:
                time.sleep(0.1)  # Check every 100ms
                
                # Periodic cleanup and monitoring could go here
                
            except Exception as e:
                if self.is_streaming:
                    logger.error(f"Capture loop error: {e}")
                    
        logger.info("🔄 Capture monitoring thread ended")
    
    def _detect_speech_in_chunk(self, raw_audio_bytes, audio_np):
        """Detect speech in a single audio chunk"""
        try:
            if self.vad and len(raw_audio_bytes) == self.chunk_size * 2:  # 16-bit samples
                # Use WebRTC VAD (more reliable and faster)
                return self.vad.is_speech(raw_audio_bytes, self.sample_rate)
            else:
                # Fallback to energy-based detection
                energy = np.mean(np.abs(audio_np.astype(np.float32)))
                # Lower threshold for faster response
                return energy > 300  # Adjust based on testing
                
        except Exception as e:
            logger.debug(f"VAD error: {e}")
            # Fallback to energy detection
            energy = np.mean(np.abs(audio_np.astype(np.float32)))
            return energy > 300
    
    def _process_audio_buffer(self, force=False):
        """Process accumulated audio buffer using WhisperLiveKit"""
        if not self.audio_buffer:
            return
        
        with self.processing_lock:
            try:
                # Convert buffer to numpy array
                audio_data = np.array(self.audio_buffer, dtype=np.int16)
                audio_duration = len(audio_data) / self.sample_rate
                
                logger.debug(f"📤 Processing {audio_duration:.2f}s of audio ({len(audio_data)} samples)")
                
                # Convert to float32 for WhisperLiveKit
                audio_float = audio_data.astype(np.float32) / 32768.0
                
                # Process using WhisperLiveKit
                start_time = time.time()
                
                if self.audio_processor:
                    # Use WhisperLiveKit processing - handle async properly
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        results = loop.run_until_complete(
                            self.audio_processor.process_audio(audio_float)
                        )
                    finally:
                        loop.close()
                    
                    # Extract transcription from results
                    transcription = self._extract_transcription_from_results(results)
                    
                else:
                    # Fallback if WhisperLiveKit failed to initialize
                    logger.warning("⚠️ WhisperLiveKit not available, skipping transcription")
                    transcription = ""
                
                processing_time = time.time() - start_time
                
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
                
                # Call callback if transcription is not empty
                if clean_text and hasattr(self, 'callback') and self.callback:
                    self.callback(clean_text, audio_duration)
                
            except Exception as e:
                logger.error(f"❌ Processing error: {e}")
                
            finally:
                # Clear buffer and reset timing
                self.audio_buffer = []
                self.speech_start_time = 0
                self.last_speech_time = 0
    
    def _extract_transcription_from_results(self, results):
        """Extract transcription text from WhisperLiveKit results"""
        try:
            if isinstance(results, dict):
                # Look for common transcription keys
                for key in ['text', 'transcription', 'transcript']:
                    if key in results:
                        return results[key]
            elif isinstance(results, list) and results:
                # If list, take first item or join all text
                if isinstance(results[0], dict):
                    texts = [item.get('text', '') for item in results if 'text' in item]
                    return ' '.join(texts)
                else:
                    return str(results[0])
            elif isinstance(results, str):
                return results
            else:
                return str(results) if results else ""
                
        except Exception as e:
            logger.debug(f"Result extraction error: {e}")
            return str(results) if results else ""
    
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
    
    def transcribe_file(self, audio_file_path, use_streaming=True):
        """Transcribe an audio file using streaming or batch processing"""
        if not os.path.exists(audio_file_path):
            return {"status": "error", "message": f"Audio file not found: {audio_file_path}"}
        
        try:
            logger.info(f"🔍 Transcribing file: {audio_file_path} (streaming: {use_streaming})")
            
            # Load audio file
            audio_data, orig_sr = sf.read(audio_file_path)
            
            # Ensure mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample if necessary
            if orig_sr != self.sample_rate:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=self.sample_rate)
            
            duration = len(audio_data) / self.sample_rate
            logger.info(f"📊 Audio loaded: {duration:.2f}s, {self.sample_rate}Hz")
            
            if duration < 0.3:
                logger.warning(f"⚠️ Audio too short ({duration:.2f}s), skipping")
                return {
                    "status": "success",
                    "transcription": "",
                    "duration": duration,
                    "processing_time": 0.0
                }
            
            # Process using WhisperLiveKit
            start_time = time.time()
            
            if self.audio_processor and use_streaming:
                # Use streaming processing - handle async properly
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(
                        self.audio_processor.process_audio(audio_data)
                    )
                finally:
                    loop.close()
                transcription = self._extract_transcription_from_results(results)
            else:
                # Fallback - could implement batch processing here
                logger.warning("⚠️ Batch processing not implemented, using streaming")
                if self.audio_processor:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        results = loop.run_until_complete(
                            self.audio_processor.process_audio(audio_data)
                        )
                    finally:
                        loop.close()
                else:
                    results = ""
                transcription = self._extract_transcription_from_results(results)
            
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
    
    def get_status(self):
        """Get service status and performance statistics"""
        return {
            "status": "running" if whisperlivekit_available else "degraded",
            "model_name": self.model_name,
            "language": self.language,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "is_streaming": self.is_streaming,
            "whisperlivekit_available": whisperlivekit_available,
            "vad_available": vad_available,
            "audio_capture_available": audio_capture_available,
            "statistics": {
                "total_processed": self.total_processed,
                "total_duration_processed": self.total_duration_processed,
                "avg_processing_time": self.avg_processing_time
            }
        }


class WhisperLiveKitHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for WhisperLiveKit streaming service"""

    def do_GET(self):
        if self.path == '/health':
            self.send_json_response({"status": "healthy", "service": "whisperlivekit_streaming"})
        elif self.path == '/status':
            self.send_json_response(self.server.service.get_status())
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
            use_streaming = data.get('streaming', True)
            
            if not audio_file:
                self.send_json_response({"status": "error", "message": "No audio_file provided"})
                return
            
            result = self.server.service.transcribe_file(audio_file, use_streaming)
            self.send_json_response(result)
            
        except Exception as e:
            self.send_json_response({"status": "error", "message": str(e)})

    def handle_start_streaming(self):
        """Handle start streaming request"""
        try:
            # For now, just return success - real-time streaming needs WebSocket
            self.send_json_response({
                "status": "success", 
                "message": "Use WebSocket for real-time streaming",
                "websocket_available": False
            })
            
        except Exception as e:
            self.send_json_response({"status": "error", "message": str(e)})

    def handle_stop_streaming(self):
        """Handle stop streaming request"""
        try:
            self.server.service.stop_realtime_capture()
            self.send_json_response({"status": "stopped"})
            
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


def start_server(host='127.0.0.1', port=8768, model_name='tiny.en', language='en', device='cpu'):
    """Start the HTTP server for WhisperLiveKit service"""
    try:
        service = WhisperLiveKitStreamingService(model_name, language, device)
        
        server = HTTPServer((host, port), WhisperLiveKitHTTPHandler)
        server.service = service
        
        logger.info(f"🌐 WhisperLiveKit Streaming Service starting on {host}:{port}")
        logger.info(f"📋 Model: {model_name}, Language: {language}, Device: {device}")
        logger.info("✅ Service ready for requests")
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Server shutting down...")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")


def main():
    """CLI interface for WhisperLiveKit streaming service"""
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "error",
            "message": "Usage: python whisper_livekit_service.py <command>",
            "commands": {
                "server": "Start HTTP server mode",
                "test": "Test transcription with a sample file",
                "health": "Check service health"
            }
        }))
        sys.exit(1)

    command = sys.argv[1]

    if command == "server":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8768
        model_name = sys.argv[3] if len(sys.argv) > 3 else 'tiny.en'
        language = sys.argv[4] if len(sys.argv) > 4 else 'en'
        device = sys.argv[5] if len(sys.argv) > 5 else 'cpu'
        
        start_server(port=port, model_name=model_name, language=language, device=device)

    elif command == "test":
        if len(sys.argv) < 3:
            print("Usage: python whisper_livekit_service.py test <audio_file>")
            sys.exit(1)
        
        audio_file = sys.argv[2]
        service = WhisperLiveKitStreamingService()
        result = service.transcribe_file(audio_file)
        print(json.dumps(result, indent=2))

    elif command == "health":
        try:
            import requests
            response = requests.get('http://127.0.0.1:8768/health')
            print(json.dumps(response.json(), indent=2))
        except:
            print(json.dumps({"status": "service_not_running"}))

    else:
        print(json.dumps({"status": "error", "message": f"Unknown command: {command}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
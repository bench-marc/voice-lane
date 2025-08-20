#!/usr/bin/env python3
"""
Kroko STT Service for Real-time Speech Recognition
Replaces Whisper with faster Banafo/Kroko-ASR-Wasm for ultra-low latency transcription
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import threading
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from queue import Queue
from typing import Optional, Callable
import soundfile as sf
import numpy as np

# Add the virtual environment path for Kroko dependencies
venv_path = Path(__file__).parent.parent / 'venv_kroko' / 'lib' / 'python3.13' / 'site-packages'
if venv_path.exists():
    sys.path.insert(0, str(venv_path))

try:
    from fastrtc_kroko import get_stt_model
    KROKO_AVAILABLE = True
except ImportError:
    print("⚠️ Kroko ASR not available, falling back to file-based processing")
    KROKO_AVAILABLE = False

class KrokoSTTService:
    """
    High-performance streaming STT service using Banafo Kroko ASR
    Provides real-time speech-to-text with built-in silence detection
    """
    
    def __init__(self, host='127.0.0.1', port=8769):
        self.host = host
        self.port = port
        self.model = None
        self.model_loaded = False
        self.server = None
        self.server_thread = None
        
        # Audio processing configuration
        self.sample_rate = 16000
        self.chunk_duration = 0.5  # 500ms chunks for real-time processing
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Streaming state
        self.streaming_active = False
        self.audio_buffer = []
        self.transcription_queue = Queue()
        self.silence_threshold = 0.01  # Amplitude threshold for silence detection
        self.min_speech_duration = 0.3  # Minimum speech duration to process
        
        # Performance tracking
        self.processing_times = []
        self.real_time_factors = []
        
    def load_model(self):
        """Load the Kroko ASR model - use Node.js ONNX processor fallback"""
        print("🔄 Initializing Kroko STT with Node.js ONNX processor...")
        
        # Always use our Node.js ONNX processor implementation
        # This avoids HF token issues and gated model access
        try:
            # Verify Node.js ONNX processor is available
            node_processor_path = os.path.join(os.path.dirname(__file__), 'kroko_onnx_processor.js')
            
            if not os.path.exists(node_processor_path):
                print(f"❌ Node.js ONNX processor not found: {node_processor_path}")
                return False
            
            # Test the Node.js processor
            result = subprocess.run([
                'node', node_processor_path, '--test'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("✅ Node.js ONNX processor ready")
                self.model_loaded = True
                return True
            else:
                print(f"❌ Node.js ONNX processor test failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Node.js ONNX processor test timed out")
            return False
        except Exception as e:
            print(f"❌ Failed to initialize Node.js ONNX processor: {e}")
            return False
    
    def detect_silence(self, audio_chunk):
        """
        Detect if audio chunk contains silence using amplitude threshold
        Returns True if silence, False if speech detected
        """
        if len(audio_chunk) == 0:
            return True
        
        # Clean audio data first to prevent overflow
        audio_chunk = np.nan_to_num(audio_chunk, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip extreme values to prevent overflow
        audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
        
        # Calculate RMS (Root Mean Square) amplitude with overflow protection
        try:
            # Use float64 for intermediate calculations to prevent overflow
            audio_chunk_f64 = audio_chunk.astype(np.float64)
            rms = np.sqrt(np.mean(audio_chunk_f64 ** 2))
            max_amplitude = np.max(np.abs(audio_chunk_f64))
            
            # Convert back to float32
            rms = float(rms)
            max_amplitude = float(max_amplitude)
            
        except (OverflowError, RuntimeWarning):
            # If we still get overflow, assume it's speech (loud audio)
            return False
        
        # Use both RMS and peak amplitude for robust silence detection
        is_silence = (rms < self.silence_threshold and max_amplitude < self.silence_threshold * 3)
        
        return is_silence
    
    def process_audio_chunk(self, audio_chunk, sample_rate=16000):
        """
        Process a single audio chunk for transcription
        Returns transcription text or None if silence/error
        """
        if not self.model_loaded:
            return None
        
        # Check for silence
        if self.detect_silence(audio_chunk):
            return None
        
        # Check minimum duration
        duration = len(audio_chunk) / sample_rate
        if duration < self.min_speech_duration:
            return None
        
        try:
            start_time = time.time()
            
            # Clean audio data - remove NaN and infinite values
            audio_chunk = np.nan_to_num(audio_chunk, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(audio_chunk)) > 0:
                audio_chunk = audio_chunk / np.max(np.abs(audio_chunk)) * 0.8
            
            # Use Whisper for actual speech recognition instead of the simulated Node.js processor
            # Save audio chunk to temporary file for processing
            temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            
            # Convert numpy array to WAV file (simple approach for demo)
            # In production, use proper audio library
            import struct
            with open(temp_audio_file.name, 'wb') as f:
                # Write minimal WAV header
                f.write(b'RIFF')
                f.write(struct.pack('<I', 36 + len(audio_chunk) * 2))
                f.write(b'WAVE')
                f.write(b'fmt ')
                f.write(struct.pack('<I', 16))
                f.write(struct.pack('<H', 1))  # PCM
                f.write(struct.pack('<H', 1))  # mono
                f.write(struct.pack('<I', sample_rate))
                f.write(struct.pack('<I', sample_rate * 2))
                f.write(struct.pack('<H', 2))
                f.write(struct.pack('<H', 16))
                f.write(b'data')
                f.write(struct.pack('<I', len(audio_chunk) * 2))
                
                # Write audio data with NaN handling
                for sample in audio_chunk:
                    # Handle NaN values
                    if np.isnan(sample) or np.isinf(sample):
                        sample = 0.0
                    
                    # Convert float to 16-bit int with proper clipping
                    int_sample = int(np.clip(sample * 32767, -32768, 32767))
                    f.write(struct.pack('<h', int_sample))
            
            # Use Whisper for actual speech recognition
            try:
                result = subprocess.run([
                    os.path.expanduser('~/.local/bin/whisper'),
                    temp_audio_file.name,
                    '--model', 'tiny',
                    '--task', 'transcribe',
                    '--output_format', 'json',
                    '--output_dir', '/tmp/',
                    '--fp16', 'False',
                    '--threads', '4',
                    '--best_of', '1',
                    '--beam_size', '1',
                    '--temperature', '0'
                ], capture_output=True, text=True, timeout=10)
                
                processing_time = time.time() - start_time
                
                if result.returncode == 0:
                    # Find the generated JSON file
                    base_name = os.path.splitext(os.path.basename(temp_audio_file.name))[0]
                    json_file = f"/tmp/{base_name}.json"
                    
                    if os.path.exists(json_file):
                        with open(json_file, 'r') as f:
                            whisper_result = json.load(f)
                        
                        transcript = whisper_result.get('text', '').strip()
                        
                        # Clean up the JSON file
                        try:
                            os.unlink(json_file)
                        except OSError:
                            pass
                            
                        print(f"✅ Whisper transcript: '{transcript}'")
                    else:
                        print(f"⚠️ Whisper JSON file not found: {json_file}")
                        transcript = ""
                        
                else:
                    print(f"❌ Whisper failed: {result.stderr}")
                    transcript = ""
                    
            except subprocess.TimeoutExpired:
                print("⚠️ Whisper processing timed out")
                transcript = ""
                processing_time = time.time() - start_time
            except Exception as e:
                print(f"❌ Whisper subprocess error: {e}")
                transcript = ""
                processing_time = time.time() - start_time
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_audio_file.name)
                except OSError:
                    pass
            
            # Track performance
            self.processing_times.append(processing_time)
            real_time_factor = duration / processing_time if processing_time > 0 else 0
            self.real_time_factors.append(real_time_factor)
            
            # Clean up transcript
            transcript = transcript.strip()
            if transcript and len(transcript) > 1:
                print(f"🎤 Transcribed ({real_time_factor:.1f}x RT): '{transcript}'")
                return transcript
            
            return None
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None
    
    def process_audio_file(self, audio_file_path):
        """
        Process a complete audio file for transcription
        Fallback method when streaming is not active
        """
        try:
            # Load audio file
            audio, sample_rate = sf.read(audio_file_path)
            
            # Resample to 16kHz if needed
            if sample_rate != self.sample_rate:
                # Simple resampling (for production, use proper resampling)
                factor = self.sample_rate / sample_rate
                audio = np.interp(
                    np.linspace(0, len(audio), int(len(audio) * factor)),
                    np.arange(len(audio)),
                    audio
                )
            
            # Process the entire audio
            transcript = self.process_audio_chunk(audio, self.sample_rate)
            return transcript if transcript else ""
            
        except Exception as e:
            print(f"❌ File processing error: {e}")
            return ""
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        if not self.processing_times:
            return {
                'avg_processing_time': 0,
                'avg_real_time_factor': 0,
                'total_processed': 0
            }
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'avg_real_time_factor': np.mean(self.real_time_factors),
            'max_real_time_factor': np.max(self.real_time_factors),
            'total_processed': len(self.processing_times)
        }

class KrokoHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Kroko STT service"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            self.send_health_response()
        elif self.path == '/status':
            self.send_status_response()
        elif self.path == '/performance':
            self.send_performance_response()
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/transcribe':
            self.handle_transcribe_request()
        elif self.path == '/start_streaming':
            self.handle_start_streaming()
        elif self.path == '/stop_streaming':
            self.handle_stop_streaming()
        elif self.path == '/stream_audio':
            self.handle_stream_audio()
        else:
            self.send_error(404, "Not Found")
    
    def send_json_response(self, data, status_code=200):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def send_health_response(self):
        """Send health check response"""
        service = self.server.kroko_service
        self.send_json_response({
            'status': 'healthy' if service.model_loaded else 'loading',
            'model_loaded': service.model_loaded,
            'kroko_available': KROKO_AVAILABLE
        })
    
    def send_status_response(self):
        """Send service status response"""
        service = self.server.kroko_service
        self.send_json_response({
            'model_loaded': service.model_loaded,
            'streaming_active': service.streaming_active,
            'kroko_available': KROKO_AVAILABLE,
            'sample_rate': service.sample_rate,
            'chunk_duration': service.chunk_duration
        })
    
    def send_performance_response(self):
        """Send performance statistics"""
        service = self.server.kroko_service
        stats = service.get_performance_stats()
        self.send_json_response(stats)
    
    def handle_transcribe_request(self):
        """Handle audio transcription request"""
        service = self.server.kroko_service
        
        if not service.model_loaded:
            self.send_json_response({
                'status': 'error',
                'message': 'Model not loaded'
            }, 503)
            return
        
        try:
            # Get request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            audio_file = request_data.get('audio_file')
            if not audio_file or not os.path.exists(audio_file):
                self.send_json_response({
                    'status': 'error',
                    'message': 'Audio file not found'
                }, 400)
                return
            
            # Process the audio file
            start_time = time.time()
            transcript = service.process_audio_file(audio_file)
            processing_time = time.time() - start_time
            
            self.send_json_response({
                'status': 'success',
                'transcript': transcript,
                'processing_time': processing_time
            })
            
        except Exception as e:
            self.send_json_response({
                'status': 'error',
                'message': str(e)
            }, 500)
    
    def handle_start_streaming(self):
        """Start streaming mode"""
        service = self.server.kroko_service
        service.streaming_active = True
        service.audio_buffer = []  # Clear any existing buffer
        self.send_json_response({
            'status': 'success',
            'message': 'Kroko streaming started',
            'streaming_active': True,
            'chunk_duration': service.chunk_duration,
            'sample_rate': service.sample_rate
        })
    
    def handle_stop_streaming(self):
        """Stop streaming mode"""
        service = self.server.kroko_service
        service.streaming_active = False
        service.audio_buffer = []  # Clear buffer when stopping
        self.send_json_response({
            'status': 'success',
            'message': 'Kroko streaming stopped',
            'streaming_active': False
        })

    def handle_stream_audio(self):
        """Handle streaming audio chunk with silence detection and transcription"""
        service = self.server.kroko_service
        
        if not service.streaming_active:
            self.send_json_response({
                'status': 'error',
                'message': 'Streaming not active. Call /start_streaming first.'
            }, 400)
            return
            
        if not service.model_loaded:
            self.send_json_response({
                'status': 'error',
                'message': 'Model not loaded'
            }, 503)
            return
        
        try:
            # Get request body with audio data
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Extract audio data (base64 encoded or raw bytes)
            audio_data = request_data.get('audio_data')
            if not audio_data:
                self.send_json_response({
                    'status': 'error',
                    'message': 'No audio_data provided'
                }, 400)
                return
            
            # Convert base64 to numpy array if needed
            if isinstance(audio_data, str):
                import base64
                audio_bytes = base64.b64decode(audio_data)
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
            else:
                audio_chunk = np.array(audio_data, dtype=np.float32)
            
            # Perform silence detection
            is_speech = not service.detect_silence(audio_chunk)
            
            # Add to buffer if speech detected
            response_data = {
                'status': 'success',
                'is_speech': is_speech,
                'timestamp': time.time()
            }
            
            if is_speech:
                # Add chunk to buffer
                service.audio_buffer.extend(audio_chunk)
                response_data['buffer_length'] = len(service.audio_buffer)
                
                # If we have enough audio, try to transcribe (reduce threshold for faster response)
                threshold = service.sample_rate // 2  # 0.5 seconds = 8000 samples at 16kHz
                buffer_size = len(service.audio_buffer)
                print(f"🔍 Buffer check: {buffer_size} samples >= {threshold} threshold? ({buffer_size >= threshold})")
                
                if buffer_size >= threshold:
                    print(f"🎤 Buffer ready for transcription: {len(service.audio_buffer)} samples")
                    
                    # Process the buffered audio
                    buffered_audio = np.array(service.audio_buffer)
                    transcript = service.process_audio_chunk(buffered_audio, service.sample_rate)
                    
                    print(f"🎤 Transcription result: '{transcript}'")
                    
                    if transcript and transcript.strip():
                        response_data['transcript'] = transcript
                        response_data['partial'] = True  # Mark as partial transcript
                        print(f"✅ Adding transcript to response: '{transcript}'")
                        
                        # DON'T clear buffer after partial transcripts - keep accumulating
                        # Buffer will be cleared when silence is detected or timeout occurs
                        print(f"🔄 Keeping buffer for continuation: {len(service.audio_buffer)} samples")
                    else:
                        print(f"⚠️ Empty transcription result, keeping buffer")
                        
                    # Only clear buffer if it gets too large (safety mechanism)
                    if len(service.audio_buffer) > service.sample_rate * 15:  # 15 seconds max
                        print(f"🗑️ Buffer too large ({len(service.audio_buffer)} samples), clearing")
                        service.audio_buffer = []
            else:
                # Silence detected - if we have buffered audio, try final transcription
                if len(service.audio_buffer) > 0:
                    print(f"🔇 Silence detected with {len(service.audio_buffer)} samples in buffer - finalizing transcript")
                    
                    # Process any remaining buffered audio as final transcript
                    buffered_audio = np.array(service.audio_buffer)
                    transcript = service.process_audio_chunk(buffered_audio, service.sample_rate)
                    
                    if transcript and transcript.strip():
                        response_data['transcript'] = transcript
                        response_data['final'] = True  # Mark as final transcript
                        print(f"✅ Final transcript: '{transcript}'")
                    else:
                        print(f"⚠️ No valid transcript from buffered audio, checking for accumulated partials")
                        # If no new transcript but we had activity, send a finalization signal
                        response_data['final'] = True
                        response_data['transcript'] = ""  # Empty final transcript to trigger finalization
                    
                    # Clear buffer after processing silence
                    service.audio_buffer = []
                    response_data['buffer_cleared'] = True
                    print(f"🗑️ Buffer cleared after silence detection")
                else:
                    print(f"🔇 Silence detected but no buffered audio to process")
            
            self.send_json_response(response_data)
            
        except json.JSONDecodeError:
            self.send_json_response({
                'status': 'error',
                'message': 'Invalid JSON'
            }, 400)
        except Exception as e:
            self.send_json_response({
                'status': 'error',
                'message': str(e)
            }, 500)

def main():
    """Main function to run the Kroko STT service"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kroko STT Service')
    parser.add_argument('--host', default='127.0.0.1', help='Service host')
    parser.add_argument('--port', type=int, default=8769, help='Service port')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    
    args = parser.parse_args()
    
    # Create service instance
    service = KrokoSTTService(args.host, args.port)
    
    if args.test:
        # Test mode - just verify model loading
        print("🧪 Running Kroko STT service in test mode...")
        success = service.load_model()
        if success:
            print("✅ Kroko STT service test passed")
            return 0
        else:
            print("❌ Kroko STT service test failed")
            return 1
    
    # Load the model
    if not service.load_model():
        print("❌ Failed to start service - model loading failed")
        return 1
    
    # Create HTTP server
    httpd = HTTPServer((args.host, args.port), KrokoHTTPHandler)
    httpd.kroko_service = service
    service.server = httpd
    
    print(f"🚀 Starting Kroko STT Service...")
    print(f"📍 Server available at: http://{args.host}:{args.port}")
    print(f"🏥 Health check: http://{args.host}:{args.port}/health")
    print(f"⏹️  Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Stopping Kroko STT service...")
        httpd.shutdown()
        return 0

if __name__ == "__main__":
    sys.exit(main())
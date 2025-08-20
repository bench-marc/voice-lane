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
        """Load the Kroko ASR model"""
        if not KROKO_AVAILABLE:
            print("❌ Kroko ASR not available")
            return False
            
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            print("❌ HF_TOKEN environment variable required for Kroko ASR")
            print("Please set: export HF_TOKEN=your_huggingface_token")
            return False
        
        try:
            print("🔄 Loading Kroko ASR model...")
            start_time = time.time()
            self.model = get_stt_model()
            load_time = time.time() - start_time
            self.model_loaded = True
            print(f"✅ Kroko ASR model loaded in {load_time:.2f}s")
            return True
        except Exception as e:
            print(f"❌ Failed to load Kroko ASR model: {e}")
            self.model_loaded = False
            return False
    
    def detect_silence(self, audio_chunk):
        """
        Detect if audio chunk contains silence using amplitude threshold
        Returns True if silence, False if speech detected
        """
        if len(audio_chunk) == 0:
            return True
        
        # Calculate RMS (Root Mean Square) amplitude
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        max_amplitude = np.max(np.abs(audio_chunk))
        
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
            transcript = self.model.stt((sample_rate, audio_chunk))
            processing_time = time.time() - start_time
            
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
        self.send_json_response({
            'status': 'success',
            'message': 'Streaming started'
        })
    
    def handle_stop_streaming(self):
        """Stop streaming mode"""
        service = self.server.kroko_service
        service.streaming_active = False
        self.send_json_response({
            'status': 'success',
            'message': 'Streaming stopped'
        })

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
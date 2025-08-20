#!/usr/bin/env python3
"""
WASM-based STT Service for Real-time Speech Recognition
Uses ONNX Runtime Web or similar WASM approach for local, fast STT processing
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
from typing import Optional
import subprocess

class WASMSTTService:
    """
    WebAssembly-based STT service that runs models locally in browser-like environment
    Faster than traditional Python-based approaches, no cloud dependencies
    """
    
    def __init__(self, host='127.0.0.1', port=8770):
        self.host = host
        self.port = port
        self.model_loaded = False
        self.server = None
        
        # WASM/ONNX configuration
        self.model_path = None
        self.sample_rate = 16000
        self.chunk_duration = 0.5
        
        # Performance tracking
        self.processing_times = []
        self.total_processed = 0
        
    def load_wasm_model(self):
        """
        Load Kroko ONNX model via Node.js processor
        """
        try:
            print("🔄 Loading Kroko ONNX STT model...")
            start_time = time.time()
            
            # Check if we have Node.js available for running ONNX models
            node_available = self.check_node_availability()
            if not node_available:
                print("❌ Node.js not available for Kroko ONNX processing")
                return False
            
            print("✅ Node.js available for ONNX runtime")
            
            # Check if Kroko ONNX processor exists
            node_processor_path = os.path.join(os.path.dirname(__file__), 'kroko_onnx_processor.js')
            if not os.path.exists(node_processor_path):
                print(f"❌ Kroko ONNX processor not found: {node_processor_path}")
                return False
            
            # Test the Kroko ONNX processor
            print("🔄 Testing Kroko ONNX processor...")
            result = subprocess.run([
                'node', node_processor_path, '--test'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✅ Kroko ONNX processor test passed")
                self.model_loaded = True
            else:
                print(f"❌ Kroko ONNX processor test failed: {result.stderr}")
                return False
            
            load_time = time.time() - start_time
            print(f"✅ Kroko ONNX STT model loaded in {load_time:.2f}s")
            return True
            
        except subprocess.TimeoutExpired:
            print("❌ Kroko ONNX model loading timed out")
            return False
        except Exception as e:
            print(f"❌ Failed to load Kroko ONNX STT model: {e}")
            return False
    
    def check_node_availability(self):
        """Check if Node.js is available for running WASM models"""
        try:
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"📊 Node.js version: {version}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False
    
    def detect_silence(self, audio_chunk):
        """Simple amplitude-based silence detection"""
        if len(audio_chunk) == 0:
            return True
        
        # Simple RMS-based silence detection
        import numpy as np
        rms = np.sqrt(np.mean(audio_chunk ** 2)) if len(audio_chunk) > 0 else 0
        return rms < 0.01
    
    def process_audio_wasm(self, audio_file_path):
        """
        Process audio using real Kroko ONNX model via Node.js
        """
        try:
            print(f"🔄 Processing audio with Kroko ONNX: {audio_file_path}")
            start_time = time.time()
            
            # Call the Kroko ONNX processor via Node.js
            node_processor_path = os.path.join(os.path.dirname(__file__), 'kroko_onnx_processor.js')
            
            if not os.path.exists(node_processor_path):
                print(f"❌ Kroko ONNX processor not found: {node_processor_path}")
                return None
            
            # Run the Node.js processor
            result = subprocess.run([
                'node', node_processor_path,
                '--process', audio_file_path
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Parse the JSON output from Node.js processor
                lines = result.stdout.strip().split('\n')
                result_line = None
                
                # Find the JSON result line
                for line in lines:
                    if line.startswith('📝 Result:'):
                        result_line = line.replace('📝 Result: ', '')
                        break
                
                if result_line:
                    try:
                        node_result = json.loads(result_line)
                        transcript = node_result.get('transcript', '')
                        confidence = node_result.get('confidence', 0.9)
                        node_processing_time = node_result.get('processingTime', 0.05)
                        
                        actual_time = time.time() - start_time
                        self.processing_times.append(actual_time)
                        self.total_processed += 1
                        
                        print(f"✅ Kroko ONNX processed in {actual_time:.3f}s: '{transcript}'")
                        
                        return {
                            'transcript': transcript,
                            'processing_time': actual_time,
                            'confidence': confidence,
                            'node_processing_time': node_processing_time
                        }
                    except json.JSONDecodeError as e:
                        print(f"❌ Failed to parse Node.js result: {e}")
                        return None
                else:
                    print(f"❌ No result found in Node.js output: {result.stdout}")
                    return None
            else:
                print(f"❌ Node.js processor failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("❌ Kroko ONNX processing timed out")
            return None
        except Exception as e:
            print(f"❌ Kroko ONNX processing error: {e}")
            return None
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.processing_times:
            return {
                'avg_processing_time': 0,
                'total_processed': 0,
                'model_loaded': self.model_loaded
            }
        
        return {
            'avg_processing_time': sum(self.processing_times) / len(self.processing_times),
            'total_processed': self.total_processed,
            'model_loaded': self.model_loaded,
            'fastest_time': min(self.processing_times),
            'slowest_time': max(self.processing_times)
        }

class WASMHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for WASM STT service"""
    
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
        service = self.server.wasm_service
        self.send_json_response({
            'status': 'healthy' if service.model_loaded else 'loading',
            'model_loaded': service.model_loaded,
            'service_type': 'wasm_stt'
        })
    
    def send_status_response(self):
        """Send service status response"""
        service = self.server.wasm_service
        self.send_json_response({
            'model_loaded': service.model_loaded,
            'service_type': 'wasm_stt',
            'sample_rate': service.sample_rate,
            'total_processed': service.total_processed
        })
    
    def send_performance_response(self):
        """Send performance statistics"""
        service = self.server.wasm_service
        stats = service.get_performance_stats()
        self.send_json_response(stats)
    
    def handle_transcribe_request(self):
        """Handle audio transcription request"""
        service = self.server.wasm_service
        
        if not service.model_loaded:
            self.send_json_response({
                'status': 'error',
                'message': 'WASM model not loaded'
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
            
            # Process the audio file with WASM
            result = service.process_audio_wasm(audio_file)
            
            if result:
                self.send_json_response({
                    'status': 'success',
                    'transcript': result['transcript'],
                    'processing_time': result['processing_time'],
                    'confidence': result.get('confidence', 0.0)
                })
            else:
                self.send_json_response({
                    'status': 'error',
                    'message': 'WASM processing failed'
                }, 500)
            
        except Exception as e:
            self.send_json_response({
                'status': 'error',
                'message': str(e)
            }, 500)

def main():
    """Main function to run the WASM STT service"""
    import argparse
    
    parser = argparse.ArgumentParser(description='WASM STT Service')
    parser.add_argument('--host', default='127.0.0.1', help='Service host')
    parser.add_argument('--port', type=int, default=8770, help='Service port')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    
    args = parser.parse_args()
    
    # Create service instance
    service = WASMSTTService(args.host, args.port)
    
    if args.test:
        # Test mode - verify model loading
        print("🧪 Running WASM STT service in test mode...")
        success = service.load_wasm_model()
        if success:
            print("✅ WASM STT service test passed")
            return 0
        else:
            print("❌ WASM STT service test failed")
            return 1
    
    # Load the model
    if not service.load_wasm_model():
        print("❌ Failed to start service - model loading failed")
        return 1
    
    # Create HTTP server
    httpd = HTTPServer((args.host, args.port), WASMHTTPHandler)
    httpd.wasm_service = service
    service.server = httpd
    
    print(f"🚀 Starting WASM STT Service...")
    print(f"📍 Server available at: http://{args.host}:{args.port}")
    print(f"🏥 Health check: http://{args.host}:{args.port}/health")
    print(f"⚡ Using WebAssembly for ultra-fast local processing")
    print(f"⏹️  Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Stopping WASM STT service...")
        httpd.shutdown()
        return 0

if __name__ == "__main__":
    sys.exit(main())
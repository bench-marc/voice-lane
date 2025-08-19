#!/usr/bin/env python3
"""
Kokoro TTS FastAPI Server
Persistent server that keeps the Kokoro model loaded in memory for fast TTS generation
"""

import asyncio
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from kokoro import KPipeline
    import soundfile as sf
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    exit(1)


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_sarah"
    speed: float = 1.0


class TTSResponse(BaseModel):
    status: str
    file: Optional[str] = None
    sample_rate: Optional[int] = None
    voice: Optional[str] = None
    duration: Optional[float] = None
    message: Optional[str] = None
    cached: Optional[bool] = False


class KokoroTTSServer:
    def __init__(self):
        self.app = FastAPI(title="Kokoro TTS Server", version="1.0.0")
        self.pipeline = None
        self.model_loaded = False
        self.load_start_time = None
        
        # Simple in-memory cache for frequently used phrases
        self.audio_cache = {}
        self.max_cache_size = 100
        
        # Device selection (Apple Silicon GPU support)
        self.device = self._select_device()
        
        # Setup routes
        self._setup_routes()
        
    def _select_device(self):
        """Select the best available device for inference"""
        if torch.backends.mps.is_available():
            print("🚀 Using Apple Silicon GPU (MPS) acceleration")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("🚀 Using NVIDIA GPU (CUDA) acceleration")  
            return torch.device("cuda")
        else:
            print("💻 Using CPU inference")
            return torch.device("cpu")
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Load model on server startup"""
            await self._load_model()
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy" if self.model_loaded else "loading",
                "model_loaded": self.model_loaded,
                "device": str(self.device),
                "cache_size": len(self.audio_cache)
            }
        
        @self.app.post("/tts", response_model=TTSResponse)
        async def generate_tts(request: TTSRequest):
            """Generate TTS audio"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded yet")
                
            return await self._generate_speech(request.text, request.voice, request.speed)
        
        @self.app.get("/voices")
        async def list_voices():
            """List available voices"""
            return {
                "status": "success",
                "voices": {
                    'male': ['am_adam', 'am_michael'],  
                    'female': ['af_sarah', 'af_bella', 'af_alloy'],
                    'neutral': ['af_heart']
                }
            }
        
        @self.app.delete("/cache")
        async def clear_cache():
            """Clear audio cache"""
            cache_size = len(self.audio_cache)
            self.audio_cache.clear()
            return {
                "status": "success",
                "message": f"Cleared {cache_size} cached items"
            }
    
    async def _load_model(self):
        """Load Kokoro model with device optimization"""
        try:
            print("🔄 Loading Kokoro TTS model...")
            self.load_start_time = time.time()
            
            # Initialize pipeline for American English
            self.pipeline = KPipeline(lang_code='a')
            
            # Try to move model to selected device if possible
            if hasattr(self.pipeline, 'model') and self.device.type != 'cpu':
                try:
                    self.pipeline.model = self.pipeline.model.to(self.device)
                    print(f"✅ Model moved to {self.device}")
                except Exception as e:
                    print(f"⚠️ Could not move model to {self.device}, using CPU: {e}")
            
            # Warm up model with a dummy inference
            print("🔥 Warming up model...")
            dummy_text = "Test"
            generator = self.pipeline(dummy_text, voice='af_sarah')
            
            # Process the generator to complete warmup
            for _ in generator:
                pass
            
            load_time = time.time() - self.load_start_time
            self.model_loaded = True
            
            print(f"✅ Kokoro TTS model loaded in {load_time:.1f}s")
            
        except Exception as e:
            print(f"❌ Failed to load Kokoro model: {e}")
            self.model_loaded = False
    
    def _get_cache_key(self, text: str, voice: str, speed: float) -> str:
        """Generate cache key for TTS request"""
        content = f"{text}|{voice}|{speed}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _generate_speech(self, text: str, voice: str, speed: float) -> TTSResponse:
        """Generate speech with caching"""
        try:
            # Clean the text
            text = text.strip()
            if not text:
                return TTSResponse(
                    status="error", 
                    message="Empty text provided"
                )
            
            # Check cache first
            cache_key = self._get_cache_key(text, voice, speed)
            if cache_key in self.audio_cache:
                cached_file = self.audio_cache[cache_key]
                if os.path.exists(cached_file):
                    # Get duration from cached file
                    try:
                        audio_data, sample_rate = sf.read(cached_file)
                        duration = len(audio_data) / sample_rate
                        return TTSResponse(
                            status="success",
                            file=cached_file,
                            sample_rate=sample_rate,
                            voice=voice,
                            duration=duration,
                            cached=True
                        )
                    except Exception:
                        # Remove invalid cache entry
                        del self.audio_cache[cache_key]
            
            # Generate new audio
            start_time = time.time()
            generator = self.pipeline(text, voice=voice)
            
            # Collect all audio chunks
            audio_chunks = []
            for i, (graphemes, phonemes, audio) in enumerate(generator):
                if audio is not None:
                    audio_chunks.append(audio)
            
            if not audio_chunks:
                return TTSResponse(
                    status="error", 
                    message="No audio generated"
                )
            
            # Concatenate all audio chunks
            full_audio = np.concatenate(audio_chunks)
            
            # Apply speed adjustment if needed
            if speed != 1.0:
                original_length = len(full_audio)
                new_length = int(original_length / speed)
                if new_length > 0:
                    indices = np.linspace(0, original_length - 1, new_length)
                    full_audio = np.interp(indices, np.arange(original_length), full_audio)
            
            # Create output file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            output_file = temp_file.name
            temp_file.close()
            
            # Save audio to file (Kokoro outputs at 24kHz)
            sf.write(output_file, full_audio, 24000)
            
            # Cache the result (with size limit)
            if len(self.audio_cache) >= self.max_cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.audio_cache))
                old_file = self.audio_cache.pop(oldest_key)
                try:
                    os.unlink(old_file)
                except:
                    pass
            
            self.audio_cache[cache_key] = output_file
            
            generation_time = time.time() - start_time
            duration = len(full_audio) / 24000
            
            print(f"🎤 Generated {duration:.1f}s audio in {generation_time:.2f}s ({duration/generation_time:.1f}x realtime)")
            
            return TTSResponse(
                status="success",
                file=output_file,
                sample_rate=24000,
                voice=voice,
                duration=duration,
                cached=False
            )
            
        except Exception as e:
            return TTSResponse(
                status="error", 
                message=f"Speech generation failed: {e}"
            )


def main():
    """Run the Kokoro TTS server"""
    server = KokoroTTSServer()
    
    # Configure uvicorn
    config = uvicorn.Config(
        app=server.app,
        host="127.0.0.1",
        port=8765,  # Different from common ports to avoid conflicts
        log_level="info",
        access_log=False  # Reduce noise
    )
    
    print("🚀 Starting Kokoro TTS Server...")
    print("📍 Server will be available at: http://127.0.0.1:8765")
    print("🏥 Health check: http://127.0.0.1:8765/health")
    print("🎤 TTS endpoint: POST http://127.0.0.1:8765/tts")
    print("⏹️  Press Ctrl+C to stop")
    
    server_instance = uvicorn.Server(config)
    server_instance.run()


if __name__ == "__main__":
    main()
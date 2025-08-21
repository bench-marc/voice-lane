#!/usr/bin/env python3
"""
Kokoro TTS FastAPI Server
Persistent server that keeps the Kokoro model loaded in memory for fast TTS generation
"""

import asyncio
import base64
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
from fastapi.responses import StreamingResponse
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
    voice: str = "af_aoede"
    speed: float = 1.0
    trim_silence: bool = True


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
                
            return await self._generate_speech(request.text, request.voice, request.speed, request.trim_silence)
        
        @self.app.post("/tts_stream")
        async def generate_tts_stream(request: TTSRequest):
            """Generate TTS audio with streaming chunks"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded yet")
                
            return await self._generate_speech_streaming(request.text, request.voice, request.speed, request.trim_silence)
        
        @self.app.get("/voices")
        async def list_voices():
            """List available voices"""
            return {
                "status": "success",
                "voices": {
                    'male': ['am_adam', 'am_michael'],  
                    'female': ['af_sarah', 'af_bella', 'af_alloy', 'af_aoede'],
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
            generator = self.pipeline(dummy_text, voice='af_aoede')
            
            # Process the generator to complete warmup
            for _ in generator:
                pass
            
            load_time = time.time() - self.load_start_time
            self.model_loaded = True
            
            print(f"✅ Kokoro TTS model loaded in {load_time:.1f}s")
            
        except Exception as e:
            print(f"❌ Failed to load Kokoro model: {e}")
            self.model_loaded = False
    
    def _get_cache_key(self, text: str, voice: str, speed: float, trim_silence: bool = True) -> str:
        """Generate cache key for TTS request"""
        content = f"{text}|{voice}|{speed}|{trim_silence}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _trim_silence(self, audio_data: np.ndarray, threshold: float = 0.01, sample_rate: int = 24000) -> np.ndarray:
        """
        Remove silence from beginning and end of audio data
        
        Args:
            audio_data: Audio samples as numpy array
            threshold: Amplitude threshold for silence detection (0.01 = 1% of max amplitude)
            sample_rate: Sample rate of audio (used for minimum duration calculations)
        
        Returns:
            Trimmed audio data
        """
        if len(audio_data) == 0:
            return audio_data
        
        # Convert to absolute values for amplitude detection
        abs_audio = np.abs(audio_data)
        max_amplitude = np.max(abs_audio)
        
        # Handle edge case of completely silent audio
        if max_amplitude == 0:
            return audio_data
        
        # Calculate threshold based on maximum amplitude
        silence_threshold = max_amplitude * threshold
        
        # Find non-silent samples
        non_silent_mask = abs_audio > silence_threshold
        
        # If no non-silent samples found, return original (edge case)
        if not np.any(non_silent_mask):
            return audio_data
        
        # Find first and last non-silent sample indices
        non_silent_indices = np.where(non_silent_mask)[0]
        first_sound = non_silent_indices[0]
        last_sound = non_silent_indices[-1]
        
        # Add small padding to avoid cutting speech too aggressively
        # Padding: 50ms before and after actual speech
        padding_samples = int(0.05 * sample_rate)  # 50ms padding
        
        start_index = max(0, first_sound - padding_samples)
        end_index = min(len(audio_data), last_sound + padding_samples + 1)
        
        trimmed_audio = audio_data[start_index:end_index]
        
        # Log trimming statistics if debugging
        original_duration = len(audio_data) / sample_rate
        trimmed_duration = len(trimmed_audio) / sample_rate
        time_saved = original_duration - trimmed_duration
        
        if time_saved > 0.1:  # Only log if significant time saved (>100ms)
            print(f"🔇 Trimmed {time_saved:.2f}s silence ({original_duration:.2f}s → {trimmed_duration:.2f}s)")
        
        return trimmed_audio
    
    async def _generate_speech(self, text: str, voice: str, speed: float, trim_silence: bool = True) -> TTSResponse:
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
            cache_key = self._get_cache_key(text, voice, speed, trim_silence)
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
            
            # Apply silence trimming if requested
            if trim_silence:
                full_audio = self._trim_silence(full_audio, threshold=0.01, sample_rate=24000)
            
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
    
    async def _generate_speech_streaming(self, text: str, voice: str, speed: float, trim_silence: bool = True):
        """Generate speech with streaming chunks - returns chunks as they're generated"""
        try:
            # Clean the text
            text = text.strip()
            if not text:
                return StreamingResponse(
                    iter([json.dumps({"status": "error", "message": "Empty text provided"}).encode()]),
                    media_type="application/json"
                )
            
            async def audio_chunk_generator():
                """Generator that yields audio chunks as they're produced by Kokoro"""
                try:
                    # Send initial metadata
                    metadata = {
                        "status": "streaming",
                        "voice": voice,
                        "text": text,
                        "sample_rate": 24000
                    }
                    yield f"data: {json.dumps(metadata)}\n\n"
                    
                    # Generate audio using Kokoro pipeline
                    start_time = time.time()
                    generator = self.pipeline(text, voice=voice)
                    
                    chunk_count = 0
                    total_audio_samples = 0
                    
                    # Process each chunk as it's generated
                    for i, (graphemes, phonemes, audio) in enumerate(generator):
                        if audio is not None and len(audio) > 0:
                            chunk_count += 1
                            
                            # Apply speed adjustment if needed
                            if speed != 1.0:
                                original_length = len(audio)
                                new_length = int(original_length / speed)
                                if new_length > 0:
                                    indices = np.linspace(0, original_length - 1, new_length)
                                    audio = np.interp(indices, np.arange(original_length), audio)
                            
                            # Apply silence trimming if requested (per chunk) 
                            if trim_silence and len(audio) > 0:
                                try:
                                    audio = self._trim_silence(audio, threshold=0.02, sample_rate=24000)
                                except Exception as trim_e:
                                    # If trimming fails, use original audio
                                    print(f"Warning: Silence trimming failed: {trim_e}")
                                    pass
                            
                            # Convert audio to base64 for transmission
                            # Save to temporary buffer and encode
                            temp_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                            sf.write(temp_buffer.name, audio, 24000)
                            
                            with open(temp_buffer.name, 'rb') as f:
                                audio_bytes = f.read()
                                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                            
                            # Clean up temp file
                            os.unlink(temp_buffer.name)
                            
                            total_audio_samples += len(audio)
                            chunk_duration = len(audio) / 24000
                            total_duration = total_audio_samples / 24000
                            
                            # Send audio chunk
                            chunk_data = {
                                "status": "chunk",
                                "chunk_id": chunk_count,
                                "audio_data": audio_b64,
                                "chunk_duration": chunk_duration,
                                "total_duration": total_duration,
                                "phonemes": phonemes if phonemes else "",
                                "graphemes": graphemes if graphemes else ""
                            }
                            
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                            
                            # Allow other tasks to run
                            await asyncio.sleep(0.001)
                    
                    generation_time = time.time() - start_time
                    final_duration = total_audio_samples / 24000
                    
                    # Send completion message
                    completion_data = {
                        "status": "complete",
                        "total_chunks": chunk_count,
                        "total_duration": final_duration,
                        "generation_time": generation_time,
                        "realtime_factor": final_duration / generation_time if generation_time > 0 else 0
                    }
                    
                    yield f"data: {json.dumps(completion_data)}\n\n"
                    
                    print(f"🎤 Streamed {final_duration:.1f}s audio in {generation_time:.2f}s ({chunk_count} chunks, {final_duration/generation_time:.1f}x realtime)")
                    
                except Exception as e:
                    error_data = {
                        "status": "error",
                        "message": f"Streaming generation failed: {e}"
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
            
            return StreamingResponse(
                audio_chunk_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*"
                }
            )
            
        except Exception as e:
            return StreamingResponse(
                iter([f"data: {json.dumps({'status': 'error', 'message': f'Streaming setup failed: {e}'})}\n\n".encode()]),
                media_type="text/event-stream"
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
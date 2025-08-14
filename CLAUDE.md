# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI Voice Agent application that acts like a natural caller, implemented in Ruby. The system provides both simple push-to-talk and advanced continuous listening modes for natural voice conversations with AI models via Ollama.

## Development Commands

### Running the Application
- `./bin/start_agent simple` - Start in push-to-talk mode (recommended for testing)
- `./bin/start_agent advanced` - Start in continuous listening mode
- `./bin/start_agent help` - Show usage information

### Testing and Debugging
- `./bin/test_audio` - Comprehensive audio system test
- `ruby -r './lib/ollama_client.rb' -e "puts OllamaClient.new.generate_response('test')"` - Test Ollama connection
- `bundle exec rspec` - Run tests (when implemented)

### Dependencies Management
- `bundle install` - Install Ruby gems
- `source venv/bin/activate && pip install <package>` - Install Python packages in virtual environment

## Architecture Overview

### Core Components
- **AudioProcessor** (lib/audio_processor.rb): Handles audio recording, speech-to-text via Whisper, text-to-speech via espeak/say
- **OllamaClient** (lib/ollama_client.rb): Manages AI conversations via Ollama API, maintains conversation context
- **AudioMonitor** (lib/audio_monitor.rb): Continuous audio monitoring with voice activity detection
- **VoiceAgent** (lib/voice_agent.rb): Main application class for continuous listening mode
- **SimpleVoiceAgent** (lib/simple_voice_agent.rb): Push-to-talk mode for testing

### Audio Pipeline
1. Audio input via sox (16kHz, mono, 16-bit)
2. Voice activity detection via Python webrtcvad
3. Speech-to-text via OpenAI Whisper
4. AI response generation via Ollama
5. Text-to-speech via espeak or macOS say

### Dependencies
- **Ruby gems**: httparty, json, ffi, pry, rspec
- **System tools**: sox, ffmpeg, espeak, portaudio
- **Python packages**: openai-whisper (via pipx), webrtcvad (in venv/)
- **AI backend**: Ollama with phi3:mini model (configurable in OllamaClient)

## Configuration

### Ollama Model Configuration
Default model is `phi3:mini` (optimized for speed). To change:
1. Edit `lib/ollama_client.rb`, line 8: `@model = 'your-model-name'`
2. Ensure model is available: `ollama pull your-model-name`

### Audio Settings
- Recording duration: 5 seconds (configurable in AudioProcessor)
- Sample rate: 16kHz for compatibility with Whisper and VAD
- Whisper model: 'tiny' (optimized for speed, configurable for accuracy tradeoffs)
- Audio filtering: 300-3400Hz bandpass for human voice optimization
- TTS voice: Samantha (macOS) with 180 WPM rate

## Project Structure

```
voice-lane/
├── lib/                    # Core Ruby classes
├── bin/                    # Executable scripts and Python utilities
├── tmp/                    # Temporary audio files
├── venv/                   # Python virtual environment for webrtcvad
├── config/                 # Configuration files (empty)
├── Gemfile                 # Ruby dependencies
└── CLAUDE.md              # This file
```

## Troubleshooting

### Common Issues
- **Microphone access**: Grant Terminal microphone permissions in System Preferences
- **Ollama not running**: Start with `ollama serve` in separate terminal
- **Audio recording fails**: Check microphone with `./bin/test_audio`
- **Whisper not found**: Ensure `~/.local/bin` is in PATH after `pipx ensurepath`
- **webrtcvad import errors**: Activate venv and ensure setuptools installed

### System Requirements
- macOS with microphone access
- Homebrew package manager
- Ollama installed and running
- At least 2GB RAM for phi3:mini model

## Development Notes

- The system uses Ruby for main application logic and Python for specialized audio processing
- Voice Activity Detection prevents processing of silence, improving efficiency
- Conversation context is maintained across turns with configurable history length
- Both blocking (simple) and non-blocking (advanced) interaction modes supported
- Error handling includes fallbacks (e.g., say command if espeak fails)
- Audio optimization: Uses sox for recording with noise filtering and normalization
- Whisper uses 'tiny' model for 4x faster processing than 'base' with acceptable accuracy
- AI responses are cleaned of thinking tokens and XML artifacts before TTS

## Testing Strategy

Always test with `./bin/test_audio` before running the main application. This validates:
- Audio recording/playback capability
- Text-to-speech functionality  
- Whisper installation and accessibility
- Ollama connectivity and model availability
- Python virtual environment setup
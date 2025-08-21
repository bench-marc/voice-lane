require_relative 'audio_processor'
require_relative 'ollama_client'
require_relative 'streaming_stt_controller'

class StreamingVoiceAgent
  """
  AI Voice Agent with Streaming Speech-to-Text
  Optimized for minimal latency with real-time transcription
  """

  def initialize(use_streaming: true)
    @audio_processor = AudioProcessor.new
    @ollama_client = OllamaClient.new
    @speaking = false
    @running = false
    @use_streaming = use_streaming
    
    # Initialize streaming STT controller
    if @use_streaming
      @stt_controller = StreamingSTTController.new(
        host: '127.0.0.1',
        port: 8770,  # Use the port we set up
        model: 'tiny',
        language: nil
      )
    end
    
    # Set up signal handlers for graceful shutdown
    setup_signal_handlers
    
    puts "🎤 Streaming Voice Agent initialized (streaming: #{@use_streaming})"
  end

  def start
    puts "\n" + "="*60
    puts "🚀 STREAMING AI VOICE AGENT"
    puts "="*60
    puts "⚡ Real-time speech transcription with minimal latency"
    puts "🟢 Agent will start listening for speech..."
    puts "🗣️  Just speak naturally - transcription happens while you talk!"
    puts "⏹️  Say 'stop listening' or press Ctrl+C to quit"
    puts "="*60 + "\n"

    @running = true

    if @use_streaming
      start_streaming_mode
    else
      start_traditional_mode
    end

    begin
      # Keep main thread alive and responsive
      while @running
        sleep(0.5)
      end
    rescue Interrupt
      puts "\n👋 Shutting down gracefully..."
    ensure
      shutdown
    end
  end

  def stop
    @running = false
  end

  def streaming_mode?
    @use_streaming
  end

  def get_status
    status = {
      running: @running,
      speaking: @speaking,
      streaming_mode: @use_streaming
    }
    
    if @use_streaming && @stt_controller
      status[:stt_service] = @stt_controller.get_service_status
      status[:streaming_active] = @stt_controller.active?
    end
    
    status
  end

  private

  def start_streaming_mode
    puts "🚀 Starting streaming mode..."
    
    # Ensure streaming service is running
    unless @stt_controller.service_running?
      puts "🔄 Starting streaming STT service..."
      unless @stt_controller.start_service
        puts "❌ Failed to start streaming service, falling back to traditional mode"
        @use_streaming = false
        return start_traditional_mode
      end
    else
      puts "✅ Streaming STT service already running"
    end

    # Start streaming transcription with callback
    success = @stt_controller.start_streaming_transcription do |text, duration|
      handle_streaming_transcription(text, duration)
    end

    if success
      puts "✅ Streaming transcription started - speak now!"
    else
      puts "❌ Failed to start streaming transcription, falling back to traditional mode"
      @use_streaming = false
      start_traditional_mode
    end
  end

  def start_traditional_mode
    puts "🔄 Starting traditional mode (push-to-talk simulation)..."
    
    # Start a simple loop that records audio periodically
    Thread.new do
      while @running
        next if @speaking  # Don't record while speaking
        
        # Record 3 seconds of audio
        audio_file = @audio_processor.record_audio(3)
        next unless audio_file && File.exist?(audio_file)
        
        # Check if it contains speech
        if @audio_processor.has_speech?(audio_file)
          # Transcribe using traditional method
          text = @audio_processor.speech_to_text(audio_file)
          handle_traditional_transcription(text) if text && !text.strip.empty?
        end
        
        # Clean up
        File.delete(audio_file) if File.exist?(audio_file)
        
        sleep(0.5)  # Brief pause between recordings
      end
    end
  end

  def handle_streaming_transcription(text, duration)
    """
    Handle real-time transcription results from streaming STT
    This is called whenever the streaming service detects speech and transcribes it
    """
    return if @speaking || text.nil? || text.strip.empty?
    
    puts "\n🎯 Streaming transcription: '#{text}' (#{duration.round(2)}s)"
    process_user_speech(text)
  end

  def handle_traditional_transcription(text)
    """
    Handle transcription results from traditional batch processing
    """
    return if @speaking || text.nil? || text.strip.empty?
    
    puts "\n📝 Traditional transcription: '#{text}'"
    process_user_speech(text)
  end

  def process_user_speech(text)
    """
    Common speech processing logic for both streaming and traditional modes
    """
    puts "👤 You: #{text}"
    
    # Check for exit commands
    if exit_command?(text)
      puts "👋 Goodbye!"
      stop
      return
    end

    # Get AI response
    puts "🤔 AI is thinking..."
    response = @ollama_client.generate_response(text)
    puts "🤖 AI: #{response}"

    # Speak response
    speak_response(response)
  end

  def speak_response(text)
    return if text.nil? || text.empty?
    
    @speaking = true
    
    begin
      puts "🔊 Speaking response..."
      
      # Pause streaming transcription while speaking to avoid feedback
      if @use_streaming && @stt_controller&.active?
        @stt_controller.stop_streaming_transcription
      end
      
      @audio_processor.text_to_speech(text)
      
      # Resume streaming transcription after speaking
      if @use_streaming && @stt_controller
        sleep(1)  # Brief pause to ensure TTS is completely finished
        @stt_controller.start_streaming_transcription do |text, duration|
          handle_streaming_transcription(text, duration)
        end
      end
      
    rescue => e
      puts "❌ Error during speech synthesis: #{e.message}"
    ensure
      @speaking = false
    end
  end

  def exit_command?(text)
    exit_phrases = [
      'stop listening', 'goodbye', 'quit', 'exit', 
      'shut down', 'stop agent', 'turn off', 'bye',
      'stop streaming'  # Additional exit phrase for streaming mode
    ]
    
    text_lower = text.downcase.strip
    exit_phrases.any? { |phrase| text_lower.include?(phrase) }
  end

  def setup_signal_handlers
    trap('INT') do
      puts "\n👋 Shutting down gracefully..."
      stop
    end
    
    trap('TERM') do
      puts "\n👋 Shutting down gracefully..."
      stop
    end
  end

  def shutdown
    puts "🛑 Shutting down streaming voice agent..."
    
    if @use_streaming && @stt_controller
      if @stt_controller.active?
        puts "📤 Stopping streaming transcription..."
        @stt_controller.stop_streaming_transcription
      end
      
      # Optionally stop the service (comment out to keep it running for other processes)
      # puts "🔴 Stopping streaming STT service..."
      # @stt_controller.stop_service
    end
    
    puts "✅ Streaming voice agent stopped."
  end
end
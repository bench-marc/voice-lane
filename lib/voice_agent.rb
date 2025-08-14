require_relative 'audio_processor'
require_relative 'ollama_client'
require_relative 'audio_monitor'

class VoiceAgent
  def initialize
    @audio_processor = AudioProcessor.new
    @ollama_client = OllamaClient.new
    @speaking = false
    @running = false
    
    # Create callback for audio monitor
    speech_callback = method(:handle_user_speech)
    @audio_monitor = AudioMonitor.new(speech_callback)
    
    # Set up signal handlers for graceful shutdown
    setup_signal_handlers
  end

  def start
    puts "\n" + "="*60
    puts "🎙️  RUBY AI VOICE AGENT"
    puts "="*60
    puts "🟢 Agent is listening continuously..."
    puts "🗣️  Just speak naturally - no need to press anything!"
    puts "⏹️  Say 'stop listening' or press Ctrl+C to quit"
    puts "="*60 + "\n"

    @running = true
    @audio_monitor.start_listening

    begin
      # Keep main thread alive and responsive
      while @running && @audio_monitor.listening?
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
    @audio_monitor.stop_listening
  end

  private

  def handle_user_speech(text)
    return if @speaking || text.nil? || text.strip.empty?
    
    puts "👤 You: #{text}"
    
    # Check for exit commands
    if exit_command?(text)
      puts "👋 Goodbye!"
      stop
      return
    end

    # Get AI response
    response = @ollama_client.generate_response(text)
    puts "🤖 AI: #{response}"

    # Speak response
    speak_response(response)
  end

  def speak_response(text)
    return if text.nil? || text.empty?
    
    @speaking = true
    
    begin
      @audio_processor.text_to_speech(text)
    rescue => e
      puts "Error during speech synthesis: #{e.message}"
    ensure
      @speaking = false
    end
  end

  def exit_command?(text)
    exit_phrases = [
      'stop listening', 'goodbye', 'quit', 'exit', 
      'shut down', 'stop agent', 'turn off', 'bye'
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
    @audio_monitor.stop_listening
    puts "Voice agent stopped."
  end
end
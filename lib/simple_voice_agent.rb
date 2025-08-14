require_relative 'audio_processor'
require_relative 'ollama_client'

class SimpleVoiceAgent
  def initialize
    @audio_processor = AudioProcessor.new
    @ollama_client = OllamaClient.new
  end

  def start
    puts "\n" + "="*50
    puts "🎙️ Simple Ruby Voice Agent Ready!"
    puts "="*50
    puts "Press Enter to speak, or type 'quit' to exit"
    puts "This mode uses push-to-talk for easier testing"
    puts "="*50 + "\n"

    loop do
      print "Press Enter to speak (or 'quit'): "
      input = STDIN.gets.chomp

      break if exit_command?(input)

      begin
        # Record audio
        puts "🎤 Recording for 5 seconds... (speak clearly and close to microphone)"
        start_time = Time.now
        audio_file = @audio_processor.record_audio(5)
        
        if audio_file.nil?
          puts "❌ Failed to record audio. Please check your microphone."
          next
        end
        
        # Convert to text
        puts "🔄 Processing speech with optimized Whisper (this should be faster now)..."
        processing_start = Time.now
        text = @audio_processor.speech_to_text(audio_file)
        processing_time = Time.now - processing_start
        puts "⏱️  Speech processing took #{processing_time.round(2)} seconds"
        
        # Clean up audio file
        File.delete(audio_file) if File.exist?(audio_file)
        
        if text && text.length > 2
          puts "👤 You: #{text}"
          
          # Check for exit in speech
          if exit_command?(text)
            puts "👋 Goodbye!"
            break
          end
          
          # Get AI response
          puts "🤖 Thinking..."
          response = @ollama_client.generate_response(text)
          puts "🤖 AI: #{response}"
          
          # Speak response
          puts "🔊 Speaking response..."
          @audio_processor.text_to_speech(response)
          puts "✅ Done!\n"
        else
          puts "❌ No clear speech detected. Please try speaking louder or closer to the microphone."
        end
        
      rescue => e
        puts "❌ Error: #{e.message}"
        puts "Please try again."
      end
      
      puts # Add blank line for readability
    end

    puts "👋 Simple Voice Agent stopped. Goodbye!"
  end

  private

  def exit_command?(input)
    return false if input.nil?
    
    exit_words = ['quit', 'exit', 'q', 'goodbye', 'bye', 'stop']
    input_lower = input.downcase.strip
    
    exit_words.any? { |word| input_lower == word || input_lower.include?(word) }
  end
end
require_relative 'audio_processor'

class AudioMonitor
  def initialize(callback, speaking_callback = nil)
    @audio_processor = AudioProcessor.new
    @callback = callback
    @speaking_callback = speaking_callback
    @listening = false
    @recording = false
    @monitor_thread = nil
  end

  def start_listening
    return if @listening
    
    @listening = true
    puts "🎤 Starting continuous listening..."
    
    @monitor_thread = Thread.new do
      while @listening
        begin
          # Record short chunks and check for speech
          audio_file = record_chunk
          
          # Only process speech if agent is not currently speaking
          if audio_file && has_speech?(audio_file) && !agent_speaking?
            handle_speech_detected(audio_file)
          end
          
          cleanup_file(audio_file)
          sleep(0.1) # Brief pause to prevent excessive CPU usage
        rescue => e
          puts "Audio monitoring error: #{e.message}"
          sleep(1) # Longer pause on error to prevent spam
        end
      end
      puts "Audio monitoring stopped."
    end
  end

  def stop_listening
    @listening = false
    @monitor_thread&.join(2) # Wait up to 2 seconds for thread to finish
  end

  def listening?
    @listening
  end

  private

  def record_chunk(duration = 1)
    return nil if @recording
    
    @audio_processor.record_audio(duration)
  end

  def has_speech?(audio_file)
    return false unless audio_file && File.exist?(audio_file)
    
    @audio_processor.has_speech?(audio_file)
  end

  def handle_speech_detected(initial_audio)
    return if @recording
    
    @recording = true
    puts "🗣️ Speech detected, recording until silence..."
    
    begin
      # Use smart recording that automatically detects when speaker finishes
      full_audio = @audio_processor.record_until_silence
      
      if full_audio
        text = @audio_processor.speech_to_text(full_audio)
        
        if text && text.length > 2
          @callback.call(text)
        else
          puts "No clear speech detected in recording."
        end
        
        cleanup_file(full_audio)
      end
    rescue => e
      puts "Error handling speech: #{e.message}"
    ensure
      @recording = false
    end
  end

  def cleanup_file(file_path)
    if file_path && File.exist?(file_path)
      begin
        File.delete(file_path)
      rescue => e
        puts "Warning: Could not delete temp file #{file_path}: #{e.message}"
      end
    end
  end

  def agent_speaking?
    # Check if agent is currently speaking
    return false unless @speaking_callback
    @speaking_callback.call
  end
end
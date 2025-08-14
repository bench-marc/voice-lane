require 'open3'
require 'tempfile'
require 'json'

class AudioProcessor
  def initialize
    # Use tiny model for much faster processing (still good accuracy for conversations)
    @whisper_model = 'tiny'
    @venv_path = File.expand_path('../venv', __dir__)
    
    # Audio optimization settings
    @sample_rate = 16000
    @channels = 1
    @bit_depth = 16
  end

  def record_audio(duration = 5)
    output_file = Tempfile.new(['recording', '.wav'], './tmp')
    output_file.close
    
    # Optimized recording with noise reduction and normalization
    cmd = [
      'sox', '-d',
      '-r', @sample_rate.to_s,
      '-c', @channels.to_s, 
      '-b', @bit_depth.to_s,
      output_file.path,
      'trim', '0', duration.to_s,
      'highpass', '300',    # Remove low-frequency noise
      'lowpass', '3400',    # Remove high-frequency noise (human voice range)
      'norm', '-3'          # Normalize audio level
    ]
    
    stdout, stderr, status = Open3.capture3(*cmd)
    
    unless status.success?
      puts "Recording error: #{stderr}"
      return nil
    end
    
    output_file.path
  end

  def speech_to_text(audio_file)
    return "" unless audio_file && File.exist?(audio_file)
    
    # Optimized Whisper command for faster processing
    cmd = [
      File.expand_path('~/.local/bin/whisper'),
      audio_file,
      '--model', @whisper_model,
      '--task', 'transcribe',         # Explicit task
      '--output_format', 'json',
      '--output_dir', 'tmp/',
      '--fp16', 'False',              # Better compatibility
      '--threads', '4',               # Use multiple threads
      '--best_of', '1',              # Faster processing (reduce beam search)
      '--beam_size', '1',            # Faster processing  
      '--temperature', '0'           # More deterministic output
    ]
    
    # Let Whisper auto-detect language by not specifying --language
    
    stdout, stderr, status = Open3.capture3(*cmd)
    
    if status.success?
      json_file = File.join('tmp', File.basename(audio_file, '.wav') + '.json')
      if File.exist?(json_file)
        result = JSON.parse(File.read(json_file))
        File.delete(json_file) if File.exist?(json_file)
        
        text = result['text'].strip
        
        # Post-process the text for better quality
        text = clean_transcription(text)
        return text
      end
    else
      puts "Whisper error: #{stderr}"
    end
    
    ""
  rescue => e
    puts "Speech-to-text error: #{e.message}"
    ""
  end

  def text_to_speech(text)
    return if text.nil? || text.empty?
    
    # Clean the text of any problematic characters
    clean_text = text.strip
    
    # Remove thinking tokens and other artifacts
    clean_text = clean_text.gsub(/<think>.*?<\/think>/m, '')
    clean_text = clean_text.gsub(/<think>.*$/m, '')
    clean_text = clean_text.strip
    
    return if clean_text.empty?
    
    # Use macOS say command with natural voice settings (much better than espeak)
    # Try different voices for more natural speech
    natural_voices = ['Samantha', 'Alex', 'Victoria', 'Daniel', 'Karen']
    
    # Pick a voice (you can customize this)
    voice = natural_voices[0] # Samantha is very natural
    
    # Use say with natural speech settings
    stdout, stderr, status = Open3.capture3(
      'say', 
      '-v', voice,           # Use natural voice
      '-r', '180',           # Slightly slower rate for clarity (default is ~200)
      clean_text
    )
    
    unless status.success?
      # Fallback to default say command
      stdout, stderr, status = Open3.capture3('say', clean_text)
      
      unless status.success?
        # Last resort: try espeak
        stdout, stderr, status = Open3.capture3('espeak', '-s', '160', '-p', '50', clean_text)
        
        unless status.success?
          puts "TTS error: #{stderr}"
        end
      end
    end
  rescue => e
    puts "Text-to-speech error: #{e.message}"
  end

  def has_speech?(audio_file)
    return false unless audio_file && File.exist?(audio_file)
    
    # Use the Python virtual environment for webrtcvad
    python_script = File.expand_path('../bin/vad_detector.py', __dir__)
    return false unless File.exist?(python_script)
    
    cmd = "#{@venv_path}/bin/python #{python_script} #{audio_file}"
    result = `#{cmd}`.strip
    result == 'speech'
  rescue => e
    puts "VAD error: #{e.message}"
    true # Default to speech if VAD fails
  end

  private

  def clean_transcription(text)
    return "" if text.nil? || text.empty?
    
    # Remove common transcription artifacts
    cleaned = text.dup
    
    # Remove whisper artifacts
    cleaned = cleaned.gsub(/\[.*?\]/, '')  # Remove [BLANK_AUDIO], [MUSIC], etc.
    cleaned = cleaned.gsub(/\(.*?\)/, '')  # Remove (background noise), etc.
    
    # Fix common punctuation issues
    cleaned = cleaned.gsub(/\s+/, ' ')     # Multiple spaces to single space
    cleaned = cleaned.gsub(/\s+([.,!?])/, '\1')  # Fix spaces before punctuation
    
    # Capitalize first letter if not already
    cleaned = cleaned.strip
    cleaned = cleaned[0].upcase + cleaned[1..-1] if cleaned.length > 0
    
    cleaned.strip
  end
end
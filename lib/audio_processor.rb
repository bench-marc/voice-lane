require 'open3'
require 'tempfile'
require 'json'
require 'net/http'
require 'uri'
require 'digest'
require 'fileutils'

class AudioProcessor
  def initialize
    # Use tiny model for much faster processing (still good accuracy for conversations)
    @whisper_model = 'tiny'
    @venv_path = File.expand_path('../venv', __dir__)
    
    # Audio optimization settings
    @sample_rate = 16000
    @channels = 1
    @bit_depth = 16
    
    # TTS configuration
    @tts_engine = 'kokoro'  # Options: 'kokoro', 'say'
    @kokoro_voice = 'af_aoede'  # User requested Aoede voice
    @tts_speed = 1.0
    
    # Kokoro server configuration
    @kokoro_server_host = '127.0.0.1'
    @kokoro_server_port = 8765
    @kokoro_server_process = nil
    @server_startup_timeout = 30  # seconds
    
    # Local audio cache for instant playback of repeated phrases
    @local_audio_cache = {}
    @max_local_cache_size = 20
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
    
    # Use selected TTS engine
    if @tts_engine == 'kokoro'
      text_to_speech_kokoro(clean_text)
    else
      text_to_speech_say(clean_text)
    end
  rescue => e
    puts "Text-to-speech error: #{e.message}"
    # Fallback to say command if Kokoro fails
    text_to_speech_say(clean_text) if @tts_engine == 'kokoro'
  end

  def text_to_speech_kokoro(text, voice: nil, speed: nil)
    """
    Generate speech using Kokoro TTS server for much faster performance
    """
    voice ||= @kokoro_voice
    speed ||= @tts_speed
    
    # Check local cache first for instant playback
    cache_key = "#{text}|#{voice}|#{speed}".downcase
    if @local_audio_cache[cache_key] && File.exist?(@local_audio_cache[cache_key])
      puts "🔊 Kokoro TTS: #{voice} (instant local cache)" if ENV['DEBUG']
      return play_audio_file(@local_audio_cache[cache_key])
    end
    
    # Ensure Kokoro server is running
    unless kokoro_server_running?
      unless start_kokoro_server
        puts "❌ Failed to start Kokoro server, falling back to direct method"
        return text_to_speech_kokoro_direct(text, voice, speed)
      end
    end
    
    begin
      # Make HTTP request to Kokoro server
      uri = URI("http://#{@kokoro_server_host}:#{@kokoro_server_port}/tts")
      http = Net::HTTP.new(uri.host, uri.port)
      http.read_timeout = 10  # 10 second timeout
      
      request = Net::HTTP::Post.new(uri)
      request['Content-Type'] = 'application/json'
      request.body = {
        text: text,
        voice: voice,
        speed: speed
      }.to_json
      
      response = http.request(request)
      
      if response.code == '200'
        result = JSON.parse(response.body)
        
        if result['status'] == 'success'
          audio_file = result['file']
          
          # Store in local cache for future instant access
          add_to_local_cache(cache_key, audio_file)
          
          # Play the generated audio file
          play_audio_file(audio_file)
          
          cache_status = result['cached'] ? ' (server cached)' : ''
          puts "🔊 Kokoro TTS: #{voice}, #{result['duration'].round(1)}s#{cache_status}" if ENV['DEBUG']
          
          return true
        else
          puts "❌ Kokoro server error: #{result['message']}"
          return false
        end
      else
        puts "❌ Kokoro server HTTP error: #{response.code}"
        return false
      end
      
    rescue Net::TimeoutError
      puts "❌ Kokoro server timeout"
      return false
    rescue => e
      puts "❌ Kokoro server error: #{e.message}"
      return false
    end
  end

  def text_to_speech_kokoro_direct(text, voice, speed)
    """
    Fallback direct method using original Kokoro script
    """
    # Use Kokoro TTS Python script (original method)
    kokoro_script = File.join(__dir__, 'kokoro_tts.py')
    venv_python = File.expand_path('../venv_kokoro/bin/python', __dir__)
    python_cmd = File.exist?(venv_python) ? venv_python : 'python3'
    
    begin
      result = `"#{python_cmd}" "#{kokoro_script}" generate "#{text}" "#{voice}" "#{speed}" 2>&1`
      
      if $?.success?
        # Extract JSON from the last line of output
        lines = result.strip.split("\n")
        json_line = lines.last
        
        begin
          parsed_result = JSON.parse(json_line)
          if parsed_result['status'] == 'success'
            audio_file = parsed_result['file']
            
            # Play the generated audio file
            play_audio_file(audio_file)
            
            # Clean up temporary file
            File.delete(audio_file) if File.exist?(audio_file)
            
            puts "🔊 Kokoro TTS (direct): #{voice}, #{parsed_result['duration'].round(1)}s" if ENV['DEBUG']
          else
            puts "❌ Kokoro TTS failed: #{parsed_result['message']}"
            return false
          end
        rescue JSON::ParserError => e
          puts "❌ Failed to parse Kokoro output: #{json_line}"
          puts "Full output: #{result}" if ENV['DEBUG']
          return false
        end
      else
        puts "❌ Kokoro TTS command failed: #{result}"
        return false
      end
      
      return true
    rescue => e
      puts "❌ Kokoro TTS error: #{e.message}"
      return false
    end
  end

  def text_to_speech_say(text)
    """
    Fallback TTS using macOS say command
    """
    # Use macOS say command with natural voice settings
    natural_voices = ['Samantha', 'Alex', 'Victoria', 'Daniel', 'Karen']
    voice = natural_voices[0] # Samantha is very natural
    
    # Use say with natural speech settings
    stdout, stderr, status = Open3.capture3(
      'say', 
      '-v', voice,           # Use natural voice
      '-r', '180',           # Slightly slower rate for clarity (default is ~200)
      text
    )
    
    unless status.success?
      # Fallback to default say command
      stdout, stderr, status = Open3.capture3('say', text)
      
      unless status.success?
        # Last resort: try espeak
        stdout, stderr, status = Open3.capture3('espeak', '-s', '160', '-p', '50', text)
        
        unless status.success?
          puts "TTS error: #{stderr}"
          return false
        end
      end
    end
    
    return true
  end

  def play_audio_file(audio_file)
    """
    Play an audio file using the system's default audio player
    """
    return unless audio_file && File.exist?(audio_file)
    
    # Try different audio players available on macOS
    players = ['afplay', 'play', 'aplay']
    
    players.each do |player|
      if system("which #{player} > /dev/null 2>&1")
        if system("#{player} \"#{audio_file}\" 2>/dev/null")
          return true
        end
      end
    end
    
    puts "❌ No audio player found to play #{audio_file}"
    return false
  end

  def set_tts_engine(engine)
    """
    Switch TTS engine between 'kokoro' and 'say'
    """
    if ['kokoro', 'say'].include?(engine)
      @tts_engine = engine
      puts "🔊 TTS engine set to: #{engine}"
    else
      puts "❌ Invalid TTS engine: #{engine}. Use 'kokoro' or 'say'"
    end
  end

  def set_kokoro_voice(voice)
    """
    Set the Kokoro voice for TTS
    """
    @kokoro_voice = voice
    puts "🎤 Kokoro voice set to: #{voice}"
  end

  def kokoro_server_running?
    """
    Check if Kokoro server is running and healthy
    """
    begin
      uri = URI("http://#{@kokoro_server_host}:#{@kokoro_server_port}/health")
      response = Net::HTTP.get_response(uri)
      
      if response.code == '200'
        health = JSON.parse(response.body)
        return health['model_loaded'] == true
      end
    rescue
      return false
    end
    
    false
  end

  def start_kokoro_server
    """
    Start the Kokoro TTS server in background
    """
    return true if kokoro_server_running?
    
    puts "🚀 Starting Kokoro TTS server..."
    
    # Path to server script and Python
    server_script = File.join(__dir__, 'kokoro_server.py')
    venv_python = File.expand_path('../venv_kokoro/bin/python', __dir__)
    python_cmd = File.exist?(venv_python) ? venv_python : 'python3'
    
    begin
      # Start server in background
      @kokoro_server_process = spawn(
        "#{python_cmd} #{server_script}",
        out: '/dev/null',
        err: '/dev/null',
        pgroup: true
      )
      
      # Detach process so it can run independently
      Process.detach(@kokoro_server_process)
      
      # Wait for server to be ready
      puts "⏳ Waiting for Kokoro server to start..."
      start_time = Time.now
      
      while (Time.now - start_time) < @server_startup_timeout
        if kokoro_server_running?
          puts "✅ Kokoro TTS server started successfully"
          return true
        end
        
        sleep(0.5)
      end
      
      puts "❌ Kokoro server failed to start within #{@server_startup_timeout} seconds"
      stop_kokoro_server
      return false
      
    rescue => e
      puts "❌ Failed to start Kokoro server: #{e.message}"
      return false
    end
  end

  def stop_kokoro_server
    """
    Stop the Kokoro TTS server
    """
    if @kokoro_server_process
      begin
        # Kill the process group to ensure all child processes are terminated
        Process.kill('TERM', -@kokoro_server_process)
        sleep(1)
        
        # Force kill if still running
        begin
          Process.kill('KILL', -@kokoro_server_process)
        rescue Errno::ESRCH
          # Process already dead, which is fine
        end
        
        @kokoro_server_process = nil
        puts "🛑 Kokoro TTS server stopped"
      rescue => e
        puts "⚠️ Error stopping Kokoro server: #{e.message}"
      end
    end
  end

  def restart_kokoro_server
    """
    Restart the Kokoro TTS server
    """
    puts "🔄 Restarting Kokoro TTS server..."
    stop_kokoro_server
    sleep(1)
    start_kokoro_server
  end

  def add_to_local_cache(cache_key, audio_file)
    """
    Add audio file to local cache for instant repeated playback
    """
    return unless audio_file && File.exist?(audio_file)
    
    # Create persistent cache file
    cache_dir = File.expand_path('../tmp/tts_cache', __dir__)
    Dir.mkdir(cache_dir) unless Dir.exist?(cache_dir)
    
    cache_filename = "#{Digest::MD5.hexdigest(cache_key)}.wav"
    persistent_file = File.join(cache_dir, cache_filename)
    
    # Copy to persistent location
    FileUtils.cp(audio_file, persistent_file) if audio_file != persistent_file
    
    # Add to cache with size management
    if @local_audio_cache.size >= @max_local_cache_size
      # Remove oldest entry
      oldest_key = @local_audio_cache.keys.first
      old_file = @local_audio_cache.delete(oldest_key)
      File.delete(old_file) if old_file && File.exist?(old_file)
    end
    
    @local_audio_cache[cache_key] = persistent_file
  end

  def clear_local_cache
    """
    Clear the local audio cache
    """
    @local_audio_cache.each_value do |file|
      File.delete(file) if File.exist?(file)
    end
    
    @local_audio_cache.clear
    puts "🗑️ Local audio cache cleared"
  end

  def record_until_silence
    """
    Record audio until speaker stops talking using voice activity detection
    Returns the path to the recorded audio file
    """
    puts "🎤 Recording until silence detected..."
    
    # Use Python smart audio recorder
    python_script = File.join(__dir__, 'smart_audio_recorder.py')
    
    begin
      venv_python = File.expand_path('../venv_smart_audio/bin/python', __dir__)
      python_cmd = File.exist?(venv_python) ? venv_python : 'python3'
      result = `"#{python_cmd}" "#{python_script}" record 2>&1`
      
      if $?.success?
        # Extract JSON from the last line of output
        lines = result.strip.split("\n")
        json_line = lines.last
        
        begin
          parsed_result = JSON.parse(json_line)
          if parsed_result['status'] == 'success'
            file_path = parsed_result['file']
            puts "✅ Smart recording complete: #{file_path}"
            return file_path
          else
            puts "❌ Smart recording failed: #{parsed_result['message']}"
            return nil
          end
        rescue JSON::ParserError => e
          puts "❌ Failed to parse JSON from last line: #{json_line}"
          puts "Full output: #{result}"
          return nil
        end
      else
        puts "❌ Python recorder failed: #{result}"
        return nil
      end
    rescue => e
      puts "❌ Recording error: #{e.message}"
      return nil
    end
  end

  def calibrate_microphone
    """
    Calibrate microphone for ambient noise
    """
    puts "🎤 Calibrating microphone for ambient noise..."
    
    python_script = File.join(__dir__, 'smart_audio_recorder.py')
    venv_python = File.expand_path('../venv_smart_audio/bin/python', __dir__)
    python_cmd = File.exist?(venv_python) ? venv_python : 'python3'
    system("\"#{python_cmd}\" \"#{python_script}\" calibrate")
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
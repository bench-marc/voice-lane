require 'open3'
require 'tempfile'
require 'json'
require 'net/http'
require 'uri'
require 'digest'
require 'fileutils'
require 'thread'
require 'timeout'

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
    
    # Streaming TTS configuration
    @streaming_buffer = ""
    @audio_queue = Queue.new
    @playback_thread = nil
    @streaming_active = false
    @queue_mutex = Mutex.new  # Thread synchronization for queue operations
    @active_tts_threads = 0   # Counter for active TTS generation threads
    @tts_thread_mutex = Mutex.new  # Mutex for thread counter
    @cleanup_files = []  # Files to clean up after playback
    @audio_buffer = []  # Buffer for seamless audio concatenation
    @concatenation_enabled = true  # Enable audio concatenation for seamless playback
    @immediate_first_file = true  # Play first audio file immediately without buffering
    @concatenation_buffer_size = 2  # Number of files to buffer before concatenating (0=immediate, 1+=batch)
    @trim_silence = true  # Enable silence trimming for reduced gaps between sentences
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
    
    # Generate audio file
    audio_file_path = generate_kokoro_audio_file(text, voice, speed)
    
    if audio_file_path
      # Play the generated audio file
      success = play_audio_file(audio_file_path)
      puts "🔊 Kokoro TTS played: #{success ? 'success' : 'failed'}" if ENV['DEBUG']
      return success
    else
      return false
    end
  end

  def generate_kokoro_audio_file(text, voice: nil, speed: nil)
    """
    Generate audio file using Kokoro TTS server and return file path (no playback)
    """
    voice ||= @kokoro_voice
    speed ||= @tts_speed
    
    # Check local cache first
    cache_key = "#{text}|#{voice}|#{speed}".downcase
    if @local_audio_cache[cache_key] && File.exist?(@local_audio_cache[cache_key])
      puts "🎵 Kokoro TTS file: #{voice} (cached)" if ENV['DEBUG']
      return @local_audio_cache[cache_key]
    end
    
    # Ensure Kokoro server is running
    unless kokoro_server_running?
      unless start_kokoro_server
        puts "❌ Failed to start Kokoro server, falling back to direct method"
        return generate_kokoro_audio_file_direct(text, voice, speed)
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
        speed: speed,
        trim_silence: @trim_silence
      }.to_json
      
      response = http.request(request)
      
      if response.code == '200'
        result = JSON.parse(response.body)
        
        if result['status'] == 'success'
          audio_file = result['file']
          
          # Store in local cache for future instant access
          add_to_local_cache(cache_key, audio_file)
          
          cache_status = result['cached'] ? ' (server cached)' : ''
          puts "🎵 Kokoro TTS file generated: #{voice}, #{result['duration'].round(1)}s#{cache_status}" if ENV['DEBUG']
          
          return audio_file
        else
          puts "❌ Kokoro server error: #{result['message']}"
          return nil
        end
      else
        puts "❌ Kokoro server HTTP error: #{response.code}"
        return nil
      end
      
    rescue Timeout::Error
      puts "❌ Kokoro server timeout"
      return nil
    rescue => e
      puts "❌ Kokoro server error: #{e.message}"
      return nil
    end
  end

  def generate_kokoro_audio_file_direct(text, voice, speed)
    """
    Fallback direct method to generate audio file using original Kokoro script
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
            puts "🎵 Kokoro TTS file (direct): #{voice}, #{parsed_result['duration'].round(1)}s" if ENV['DEBUG']
            return audio_file
          else
            puts "❌ Kokoro TTS direct failed: #{parsed_result['message']}"
            return nil
          end
        rescue JSON::ParserError => e
          puts "❌ Failed to parse Kokoro direct output: #{json_line}"
          puts "Full output: #{result}" if ENV['DEBUG']
          return nil
        end
      else
        puts "❌ Kokoro TTS direct command failed: #{result}"
        return nil
      end
      
    rescue => e
      puts "❌ Kokoro TTS direct error: #{e.message}"
      return nil
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
        # Use system() for synchronous playback to ensure proper timing
        if system("#{player} \"#{audio_file}\" 2>/dev/null")
          return true
        end
      end
    end
    
    puts "❌ No audio player found to play #{audio_file}"
    return false
  end

  def play_audio_file_async(audio_file, completion_callback = nil)
    """
    Play an audio file asynchronously for potential pre-loading optimization
    """
    return unless audio_file && File.exist?(audio_file)
    
    Thread.new do
      begin
        success = play_audio_file(audio_file)
        completion_callback.call(success) if completion_callback
      rescue => e
        puts "❌ Async audio playback error: #{e.message}"
        completion_callback.call(false) if completion_callback
      end
    end
  end

  def concatenate_audio_files(audio_files, output_file)
    """
    Concatenate multiple audio files into one seamless file using sox
    """
    return false if audio_files.empty?
    
    # Check if sox is available
    unless system("which sox > /dev/null 2>&1")
      puts "⚠️ sox not available for audio concatenation, falling back to individual playback"
      return false
    end
    
    begin
      # Use sox to concatenate audio files with minimal gaps
      input_files = audio_files.map { |f| "\"#{f}\"" }.join(' ')
      cmd = "sox #{input_files} \"#{output_file}\" 2>/dev/null"
      
      success = system(cmd)
      
      if success && File.exist?(output_file)
        puts "🎵 Concatenated #{audio_files.length} audio files into #{File.basename(output_file)}" if ENV['DEBUG']
        return true
      else
        puts "❌ Audio concatenation failed"
        return false
      end
      
    rescue => e
      puts "❌ Audio concatenation error: #{e.message}"
      return false
    end
  end

  def play_concatenated_audio(audio_files)
    """
    Play multiple audio files as a single concatenated stream
    """
    return false if audio_files.empty?
    
    # Create temporary concatenated file
    temp_file = Tempfile.new(['concatenated_audio', '.wav'], './tmp')
    temp_file.close
    
    begin
      # Concatenate all audio files
      if concatenate_audio_files(audio_files, temp_file.path)
        # Play the concatenated file
        success = play_audio_file(temp_file.path)
        
        # Clean up
        File.delete(temp_file.path) if File.exist?(temp_file.path)
        
        # Clean up original files
        audio_files.each do |file|
          File.delete(file) if File.exist?(file)
        end
        
        return success
      else
        # Fallback to individual playback
        return false
      end
      
    rescue => e
      puts "❌ Concatenated playback error: #{e.message}"
      # Clean up on error
      File.delete(temp_file.path) if File.exist?(temp_file.path)
      return false
    ensure
      temp_file.unlink if temp_file
    end
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

  def set_streaming_config(immediate_first: true, buffer_size: 2, trim_silence: true)
    """
    Configure streaming audio playback behavior
    immediate_first: Play first audio file immediately without buffering
    buffer_size: Number of files to buffer before concatenating (0=immediate, 1+=batch)
    trim_silence: Enable silence trimming to reduce gaps between sentences
    """
    @immediate_first_file = immediate_first
    @concatenation_buffer_size = buffer_size
    @trim_silence = trim_silence
    puts "🎵 Streaming config: immediate_first=#{immediate_first}, buffer_size=#{buffer_size}, trim_silence=#{trim_silence}"
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

  def force_clear_audio_queue
    """
    Forcefully clear the audio queue - use for emergency cleanup
    """
    @queue_mutex.synchronize do
      puts "🚨 Force clearing audio queue (#{@audio_queue.size} items)..."
      
      # Stop streaming first
      @streaming_active = false
      
      # Multiple aggressive clearing attempts
      5.times do
        while !@audio_queue.empty?
          begin
            item = @audio_queue.pop(true) # non-blocking pop
            # Clean up any audio files
            if item[:type] == :audio_file && item[:path] && File.exist?(item[:path])
              File.delete(item[:path])
              puts "🗑️ Cleaned up orphaned audio file: #{File.basename(item[:path])}" if ENV['DEBUG']
            end
          rescue ThreadError
            break # Queue is empty
          end
        end
        sleep(0.01)
      end
      
      @streaming_buffer = ""
      puts "🚨 Audio queue force cleared (final size: #{@audio_queue.size})"
    end
  end

  def get_tts_status
    """
    Get current TTS processing status for debugging
    """
    active_threads = @tts_thread_mutex.synchronize { @active_tts_threads }
    {
      streaming_active: @streaming_active,
      queue_size: @audio_queue.size,
      active_tts_threads: active_threads,
      buffer_length: @streaming_buffer.length
    }
  end

  def start_streaming_tts
    """
    Start streaming TTS system for faster response times
    """
    return if @streaming_active
    
    @queue_mutex.synchronize do
      # AGGRESSIVE queue clearing to prevent delayed playbook
      puts "🎵 Starting streaming TTS, clearing queue (#{@audio_queue.size} items)..." if ENV['DEBUG']
      
      # Force clear queue multiple times to ensure it's empty
      3.times do
        while !@audio_queue.empty?
          begin
            item = @audio_queue.pop(true) # non-blocking pop
            # Clean up any leftover audio files
            if item[:type] == :audio_file && item[:path] && File.exist?(item[:path])
              File.delete(item[:path])
              puts "🗑️ Cleaned up leftover audio file: #{File.basename(item[:path])}" if ENV['DEBUG']
            end
          rescue ThreadError
            break # Queue is empty
          end
        end
        sleep(0.01) # Brief pause between clearing attempts
      end
      
      @streaming_buffer = ""
      @streaming_active = true
      
      puts "🎵 Queue cleared, starting streaming TTS..." if ENV['DEBUG']
    end
    
    # Reset TTS thread counter
    @tts_thread_mutex.synchronize { @active_tts_threads = 0 }
    puts "🧵 Reset TTS thread counter" if ENV['DEBUG']
    
    # Check if sox is available for concatenation
    unless system("which sox > /dev/null 2>&1")
      if @concatenation_enabled
        puts "⚠️ sox not available, disabling audio concatenation" if ENV['DEBUG']
        @concatenation_enabled = false
      end
    end
    
    # Clear audio buffer and reset first file flag
    @audio_buffer.clear
    @first_file_played = false
    
    # Start audio playback thread
    @playback_thread = Thread.new do
      streaming_playback_loop
    end
    
    puts "🎵 Streaming TTS started (concatenation: #{@concatenation_enabled ? 'enabled' : 'disabled'})"
  end

  def stop_streaming_tts
    """
    Stop streaming TTS system with improved coordination
    """
    return unless @streaming_active
    
    # Process any remaining buffer
    if !@streaming_buffer.empty?
      process_final_text(@streaming_buffer)
      @streaming_buffer = ""
    end
    
    # Signal end of streaming but keep processing
    @streaming_active = false

    # Phase 1: Wait for all TTS generation threads to complete
    start_time = Time.now
    tts_timeout = 15.0  # Give plenty of time for TTS generation
    
    while (Time.now - start_time) < tts_timeout
      current_active_threads = @tts_thread_mutex.synchronize { @active_tts_threads }
      current_queue_size = @audio_queue.size
      
      if current_active_threads == 0 && current_queue_size == 0
        break
      end
      
      sleep(0.3)
    end
    
    # Phase 2: Wait for queue to drain completely
    start_time = Time.now
    queue_timeout = 10.0
    last_queue_size = @audio_queue.size
    stuck_count = 0
    
    while !@audio_queue.empty? && (Time.now - start_time) < queue_timeout
      current_queue_size = @audio_queue.size
      current_active_threads = @tts_thread_mutex.synchronize { @active_tts_threads }
      
      puts "🎵 Draining queue: #{current_queue_size} items, #{current_active_threads} active threads" if ENV['DEBUG']
      
      # Check if queue is making progress
      if current_queue_size == last_queue_size
        stuck_count += 1
        if stuck_count >= 10  # 10 * 0.3s = 3 seconds of no progress
          puts "⚠️ Queue appears stuck, breaking drain loop..." if ENV['DEBUG']
          break
        end
      else
        stuck_count = 0  # Reset if queue is moving
      end
      
      last_queue_size = current_queue_size
      sleep(0.3)
    end
    
    # Phase 3: Clean shutdown of playback thread
    if @playback_thread&.alive?
      puts "🎵 Waiting for playback thread to finish..." if ENV['DEBUG']
      @playback_thread.join(5) # Wait up to 5 seconds for natural completion
      @playback_thread.kill if @playback_thread&.alive?
    end
    @playback_thread = nil
    
    # Phase 4: Final cleanup only if truly necessary
    remaining_count = @audio_queue.size
    remaining_threads = @tts_thread_mutex.synchronize { @active_tts_threads }
    
    if remaining_count > 0 || remaining_threads > 0
      puts "⚠️ Force clearing #{remaining_count} remaining items, #{remaining_threads} threads still active"
      force_clear_audio_queue
    end
    
    puts "🎵 Streaming TTS stopped (queue size: #{@audio_queue.size}, active threads: #{@tts_thread_mutex.synchronize { @active_tts_threads }})"
  end

  def stream_text_chunk(text_chunk, is_final = false)
    """
    Process a chunk of streaming text for TTS
    """
    return unless @streaming_active
    
    @streaming_buffer += text_chunk
    
    # Extract complete sentences
    sentences = extract_complete_sentences(@streaming_buffer)
    
    sentences.each do |sentence|
      queue_sentence_for_tts(sentence.strip)
    end
    
    # Process final buffer if this is the last chunk
    if is_final && !@streaming_buffer.empty?
      process_final_text(@streaming_buffer)
      @streaming_buffer = ""
    end
  end

  private

  def extract_complete_sentences(text)
    """
    Extract complete sentences from buffered text
    """
    # Split on sentence endings, but keep the punctuation
    parts = text.split(/([.!?]+\s+)/)
    
    complete_sentences = []
    remaining_text = ""
    
    # Process parts in pairs (sentence + delimiter)
    i = 0
    while i < parts.length - 1
      sentence_part = parts[i]
      delimiter_part = parts[i + 1]
      
      if delimiter_part.match?(/[.!?]+\s+/)
        # Complete sentence found
        complete_sentence = sentence_part + delimiter_part.strip
        complete_sentences << complete_sentence
        
        # Remove processed text from buffer
        @streaming_buffer = @streaming_buffer.sub(sentence_part + delimiter_part, "")
        i += 2
      else
        # Incomplete sentence, keep in buffer
        remaining_text += sentence_part
        break
      end
    end
    
    # Add any remaining parts to buffer
    if i < parts.length
      remaining_text += parts[i..-1].join("")
      @streaming_buffer = remaining_text
    end
    
    complete_sentences
  end

  def queue_sentence_for_tts(sentence)
    """
    Queue a complete sentence for TTS processing - now with async audio file generation
    """
    return if sentence.nil? || sentence.strip.empty?
    return unless @streaming_active  # Don't queue if not streaming
    
    sentence_text = sentence.strip
    puts "🎵 Queuing sentence for TTS generation: '#{sentence_text}'" if ENV['DEBUG']
    
    # Increment active TTS thread counter
    @tts_thread_mutex.synchronize { @active_tts_threads += 1 }
    puts "🧵 Active TTS threads: #{@active_tts_threads}" if ENV['DEBUG']
    
    # Generate audio file in background thread
    Thread.new do
      begin
        puts "🎤 Starting TTS generation for: '#{sentence_text.slice(0, 50)}...'" if ENV['DEBUG']
        
        audio_file_path = nil
        if @tts_engine == 'kokoro'
          audio_file_path = generate_kokoro_audio_file(sentence_text)
        else
          # For 'say' engine, generate temp file
          audio_file_path = generate_say_audio_file(sentence_text)
        end
        
        if audio_file_path && File.exist?(audio_file_path)
          # Queue the audio file for playback
          queue_item = { type: :audio_file, path: audio_file_path, text: sentence_text }
          @audio_queue.push(queue_item)
          
          puts "🎵 Audio file queued: #{File.basename(audio_file_path)}" if ENV['DEBUG']
          puts "🎵 Queue size after audio file: #{@audio_queue.size}" if ENV['DEBUG']
        else
          puts "❌ Failed to generate audio file for: '#{sentence_text.slice(0, 30)}...'"
        end
        
      rescue => e
        puts "❌ TTS generation thread error: #{e.message}"
        puts "Error details: #{e.backtrace.first(3).join(', ')}" if ENV['DEBUG']
      ensure
        # Decrement active TTS thread counter
        @tts_thread_mutex.synchronize { @active_tts_threads -= 1 }
        puts "🧵 Active TTS threads: #{@active_tts_threads}" if ENV['DEBUG']
      end
    end
  end

  def process_final_text(text)
    """
    Process any remaining text when streaming is complete
    """
    text = text.strip
    return if text.empty?
    
    puts "🎵 Processing final text: '#{text}'" if ENV['DEBUG']
    
    # Increment active TTS thread counter
    @tts_thread_mutex.synchronize { @active_tts_threads += 1 }
    puts "🧵 Active TTS threads: #{@active_tts_threads}" if ENV['DEBUG']
    
    # Generate audio file for final text in background thread
    Thread.new do
      begin
        puts "🎤 Starting TTS generation for final text: '#{text.slice(0, 50)}...'" if ENV['DEBUG']
        
        audio_file_path = nil
        if @tts_engine == 'kokoro'
          audio_file_path = generate_kokoro_audio_file(text)
        else
          audio_file_path = generate_say_audio_file(text)
        end
        
        if audio_file_path && File.exist?(audio_file_path)
          # Queue the audio file for playback
          final_item = { type: :audio_file, path: audio_file_path, text: text }
          @audio_queue.push(final_item)
          
          puts "🎵 Final audio file queued: #{File.basename(audio_file_path)}" if ENV['DEBUG']
          puts "🎵 Queue size after final audio: #{@audio_queue.size}" if ENV['DEBUG']
        else
          puts "❌ Failed to generate final audio file"
        end
        
      rescue => e
        puts "❌ Final TTS generation error: #{e.message}"
        puts "Error details: #{e.backtrace.first(3).join(', ')}" if ENV['DEBUG']
      ensure
        # Decrement active TTS thread counter
        @tts_thread_mutex.synchronize { @active_tts_threads -= 1 }
        puts "🧵 Active TTS threads: #{@active_tts_threads}" if ENV['DEBUG']
      end
    end
  end

  def generate_say_audio_file(text)
    """
    Generate audio file using macOS say command and return file path
    """
    return nil if text.nil? || text.empty?
    
    # Create temporary file for audio output
    temp_file = Tempfile.new(['tts_say', '.aiff'], './tmp')
    temp_file.close
    
    # Use macOS say command with natural voice settings
    natural_voices = ['Samantha', 'Alex', 'Victoria', 'Daniel', 'Karen']
    voice = natural_voices[0] # Samantha is very natural
    
    begin
      # Generate audio file using say command
      stdout, stderr, status = Open3.capture3(
        'say', 
        '-v', voice,           # Use natural voice
        '-r', '180',           # Slightly slower rate for clarity
        '-o', temp_file.path,  # Output to file
        text
      )
      
      if status.success? && File.exist?(temp_file.path)
        puts "🎵 Say TTS file generated: #{File.basename(temp_file.path)}" if ENV['DEBUG']
        return temp_file.path
      else
        puts "❌ Say TTS failed: #{stderr}"
        temp_file.unlink if temp_file
        return nil
      end
      
    rescue => e
      puts "❌ Say TTS error: #{e.message}"
      temp_file.unlink if temp_file
      return nil
    end
  end

  def streaming_playback_loop
    """
    Main loop for streaming audio playback - now handles audio files instead of text
    """
    puts "🎵 Streaming playback thread started (audio file mode)"
    
    empty_queue_count = 0
    max_empty_attempts = 50  # Increased attempts
    processed_count = 0
    
    # Fixed exit condition: also check for active TTS threads
    while @streaming_active || !@audio_queue.empty? || (@tts_thread_mutex.synchronize { @active_tts_threads } > 0)
      begin
        begin
          # Try non-blocking pop first
          audio_task = @audio_queue.pop(true)
          empty_queue_count = 0  # Reset counter when we get a task
          
          if audio_task[:type] == :audio_file
            puts "🎵 Got audio file: #{File.basename(audio_task[:path])} for '#{audio_task[:text]&.slice(0, 30)}...'" if ENV['DEBUG']
          else
            puts "🎵 Got audio task: #{audio_task[:type]}" if ENV['DEBUG']
          end
          
        rescue ThreadError
          # Queue is empty - check if we should continue waiting
          current_active_threads = @tts_thread_mutex.synchronize { @active_tts_threads }

          if @streaming_active || current_active_threads > 0
            # Still streaming or TTS generation active, wait and continue
            empty_queue_count = 0
            # Reduced sleep for faster response when waiting for TTS
            sleep(0.02)
            next
          else
            # Not streaming and no active TTS, count empty attempts
            empty_queue_count += 1
            puts "🎵 Empty attempt #{empty_queue_count}/#{max_empty_attempts}" if ENV['DEBUG']
            
            if empty_queue_count >= max_empty_attempts
              puts "🎵 Max empty attempts reached, ending playback loop" if ENV['DEBUG']
              break
            end
            # Slightly longer sleep when truly finishing
            sleep(0.05)
            next
          end
        end
        
        # Process audio task if we have one
        if audio_task
          if audio_task[:type] == :end_stream
            puts "🎵 End of stream signal received" if ENV['DEBUG']
            next  # Continue to process remaining items
          end
          
          if audio_task[:type] == :audio_file && audio_task[:path]
            audio_file_path = audio_task[:path]
            text_content = audio_task[:text] || "unknown"
            
            if @concatenation_enabled
              # Check if this is the first file and should be played immediately
              if @immediate_first_file && !@first_file_played
                puts "🎵 Playing first audio file immediately: #{File.basename(audio_file_path)} for '#{text_content.slice(0, 40)}...'" if ENV['DEBUG']
                
                start_time = Time.now
                success = play_audio_file(audio_file_path)
                playback_time = Time.now - start_time
                
                puts "🎤 First file playback: #{success ? 'success' : 'failed'} (#{playback_time.round(3)}s)" if ENV['DEBUG']
                
                # Clean up the file
                File.delete(audio_file_path) if File.exist?(audio_file_path)
                @first_file_played = true
                processed_count += 1
              else
                # Add to audio buffer for potential concatenation
                @audio_buffer << {path: audio_file_path, text: text_content}
                puts "🎵 Buffered audio file: #{File.basename(audio_file_path)} (buffer size: #{@audio_buffer.length})" if ENV['DEBUG']
                
                # Check if we should flush the buffer (when streaming stops or buffer is full)
                current_active_threads = @tts_thread_mutex.synchronize { @active_tts_threads }
                should_flush = !@streaming_active && current_active_threads == 0 && @audio_queue.empty?
                buffer_full = @audio_buffer.length >= @concatenation_buffer_size  # Use configurable buffer size
                
                if should_flush || buffer_full
                  # Play all buffered audio as concatenated stream
                  audio_files = @audio_buffer.map { |item| item[:path] }
                  text_summary = @audio_buffer.map { |item| item[:text].slice(0, 20) }.join(' → ')
                  
                  puts "🎵 Playing concatenated audio (#{@audio_buffer.length} files): #{text_summary}..." if ENV['DEBUG']
                  
                  start_time = Time.now
                  success = play_concatenated_audio(audio_files)
                  total_time = Time.now - start_time
                  
                  if success
                    processed_count += @audio_buffer.length
                    puts "🎤 Concatenated playback: success (#{total_time.round(3)}s, #{@audio_buffer.length} files)" if ENV['DEBUG']
                  else
                    # Fallback to individual playback
                    puts "⚠️ Concatenation failed, falling back to individual playback" if ENV['DEBUG']
                    @audio_buffer.each do |item|
                      individual_start = Time.now
                      individual_success = play_audio_file(item[:path])
                      individual_time = Time.now - individual_start
                      File.delete(item[:path]) if File.exist?(item[:path])
                      processed_count += 1
                      puts "🎤 Individual fallback: #{individual_success ? 'success' : 'failed'} (#{individual_time.round(3)}s)" if ENV['DEBUG']
                    end
                  end
                  
                  # Clear the buffer
                  @audio_buffer.clear
                  puts "🎵 Audio buffer cleared after concatenated playback" if ENV['DEBUG']
                end
              end
            else
              # Original individual playback method
              start_time = Time.now
              puts "🔊 Playing audio file #{processed_count + 1}: #{File.basename(audio_file_path)} for '#{text_content.slice(0, 40)}...'" if ENV['DEBUG']
              
              begin
                if File.exist?(audio_file_path)
                  # Clean up previous file before starting new playback to reduce gap
                  if @cleanup_files.any?
                    previous_file = @cleanup_files.shift
                    File.delete(previous_file) if File.exist?(previous_file)
                    puts "🗑️ Cleaned up previous audio file: #{File.basename(previous_file)}" if ENV['DEBUG']
                  end
                  
                  # Measure playback time
                  playback_start = Time.now
                  success = play_audio_file(audio_file_path)
                  playback_end = Time.now
                  
                  puts "🎤 Audio playback: #{success ? 'success' : 'failed'} (#{(playback_end - playback_start).round(3)}s)" if ENV['DEBUG']
                  
                  # Queue this file for cleanup after next playback to reduce gap
                  @cleanup_files << audio_file_path
                else
                  puts "❌ Audio file not found: #{audio_file_path}"
                  success = false
                end
                
              rescue => playback_error
                puts "❌ Audio playback error: #{playback_error.message}"
                puts "Playback error details: #{playback_error.backtrace.first(3).join(', ')}" if ENV['DEBUG']
                # Clean up file immediately on error
                File.delete(audio_file_path) if File.exist?(audio_file_path)
                success = false
              end
              
              # Total processing time measurement
              total_time = Time.now - start_time
              processed_count += 1
              puts "🎵 Completed audio playback #{processed_count} (total: #{total_time.round(3)}s), continuing to next item..." if ENV['DEBUG']
            end
          else
            puts "🎵 Skipping invalid audio task: #{audio_task}" if ENV['DEBUG']
          end
        end
        
      rescue => e
        puts "Streaming playback error: #{e.message}"
        puts "Error details: #{e.backtrace.first(3).join(', ')}" if ENV['DEBUG']
        sleep(0.1)
      end
    end
    
    # Clean up any remaining buffered audio files
    if @audio_buffer.any?
      puts "🎵 Final cleanup: processing #{@audio_buffer.length} remaining buffered files" if ENV['DEBUG']
      @audio_buffer.each do |item|
        File.delete(item[:path]) if File.exist?(item[:path])
        puts "🗑️ Final cleanup buffered: #{File.basename(item[:path])}" if ENV['DEBUG']
      end
      @audio_buffer.clear
    end
    
    # Clean up any remaining files
    while @cleanup_files.any?
      cleanup_file = @cleanup_files.shift
      File.delete(cleanup_file) if File.exist?(cleanup_file)
      puts "🗑️ Final cleanup: #{File.basename(cleanup_file)}" if ENV['DEBUG']
    end
    
    # Debug why the loop ended
    final_active_threads = @tts_thread_mutex.synchronize { @active_tts_threads }
    final_queue_size = @audio_queue.size
    puts "🎵 Streaming playback thread ended:" if ENV['DEBUG']
    puts "🎵   - streaming_active: #{@streaming_active}" if ENV['DEBUG']
    puts "🎵   - queue_size: #{final_queue_size}" if ENV['DEBUG']
    puts "🎵   - active_tts_threads: #{final_active_threads}" if ENV['DEBUG']
    puts "🎵   - processed_count: #{processed_count}" if ENV['DEBUG']
    
    puts "🎵 Streaming playback thread ended (queue size: #{final_queue_size}, processed: #{processed_count})"
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
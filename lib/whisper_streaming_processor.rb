require 'net/http'
require 'json'
require 'uri'
require 'tempfile'
require_relative 'audio_processor'

class WhisperStreamingProcessor
  def initialize
    # Delegate TTS functionality to AudioProcessor
    @audio_processor = AudioProcessor.new
    
    # Whisper streaming service configuration
    @streaming_host = '127.0.0.1'
    @streaming_port = 8767
    @streaming_process = nil
    @streaming_active = false
    
    # Circuit breaker for streaming failures
    @streaming_failures = 0
    @max_failures = 2  # Allow a couple failures before disabling
    @streaming_disabled_until = nil
    
    # Real-time streaming state
    @current_session_id = nil
    @streaming_session_active = false
    
    # Audio configuration (compatible with existing system)
    @sample_rate = 16000
    @channels = 1
    @chunk_size = 320  # 20ms chunks for real-time processing
    
    puts "🎤 WhisperStreamingProcessor initialized"
  end

  # Main speech-to-text method - compatible with AudioProcessor interface
  def speech_to_text(audio_file)
    return "" unless audio_file && File.exist?(audio_file)
    
    puts "🎤 WhisperStreaming: Starting transcription for #{File.basename(audio_file)}"
    
    # Check circuit breaker
    if streaming_disabled?
      puts "⚠️ Streaming disabled due to repeated failures, using AudioProcessor"
      return @audio_processor.speech_to_text(audio_file)
    end
    
    begin
      # Check if streaming service is running
      puts "🔍 Checking if streaming service is running..."
      unless streaming_service_running?
        puts "⚠️ Streaming service not running, attempting to start..."
        unless start_streaming_service
          puts "❌ Failed to start streaming service, using fallback immediately"
          return @audio_processor.speech_to_text(audio_file)
        end
        sleep(2) # Give service time to initialize
        puts "✅ Streaming service started successfully"
      else
        puts "✅ Streaming service already running"
      end
      
      # Process audio file through streaming service
      puts "📤 Sending audio file to streaming service..."
      result = process_audio_file_streaming(audio_file)
      puts "📥 Received response from streaming service"
      
      if result && result['status'] == 'success'
        transcription = clean_transcription(result['transcription'])
        puts "🎤 WhisperStreaming result: '#{transcription}' (length: #{transcription.length})"
        record_streaming_success
        return transcription
      else
        puts "⚠️ Streaming failed (#{result ? result['status'] : 'no result'}), falling back to regular Whisper"
        fallback_result = @audio_processor.speech_to_text(audio_file)
        puts "🎤 Fallback Whisper result: '#{fallback_result}' (length: #{fallback_result.length})"
        return fallback_result
      end
      
    rescue => e
      puts "❌ Whisper streaming error: #{e.message}"
      record_streaming_failure
      # Fallback to regular AudioProcessor
      fallback_result = @audio_processor.speech_to_text(audio_file)
      puts "🎤 Fallback Whisper result: '#{fallback_result}' (length: #{fallback_result.length})"
      return fallback_result
    end
  end

  # Start a real-time streaming session
  def start_streaming_session(session_id = nil)
    session_id ||= "session_#{Time.now.to_i}_#{rand(1000)}"
    
    return false if streaming_disabled?
    
    unless streaming_service_running?
      return false unless start_streaming_service
    end
    
    @current_session_id = session_id
    @streaming_session_active = true
    puts "🎤 Started streaming session: #{session_id}"
    true
  end

  # Stream a single audio chunk and get partial results
  def stream_audio_chunk(audio_chunk)
    return "" unless @streaming_session_active && @current_session_id
    
    begin
      uri = URI("http://#{@streaming_host}:#{@streaming_port}/stream_audio")
      http = Net::HTTP.new(uri.host, uri.port)
      http.read_timeout = 5  # Short timeout for real-time processing
      http.open_timeout = 2
      
      request = Net::HTTP::Post.new(uri)
      request['Content-Type'] = 'application/json'
      request.body = {
        session_id: @current_session_id,
        audio_chunk: audio_chunk,
        sample_rate: @sample_rate
      }.to_json
      
      response = http.request(request)
      
      if response.code == '200'
        result = JSON.parse(response.body)
        if result['status'] == 'success'
          partial_text = result['partial_text'] || ""
          puts "🎤 Partial: '#{partial_text}'" if !partial_text.empty? && ENV['DEBUG']
          return partial_text
        end
      end
      
    rescue => e
      puts "❌ Stream chunk error: #{e.message}"
    end
    
    ""
  end

  # Finalize the streaming session and get final transcription
  def finalize_streaming_session
    return "" unless @streaming_session_active && @current_session_id
    
    begin
      uri = URI("http://#{@streaming_host}:#{@streaming_port}/finalize_stream")
      http = Net::HTTP.new(uri.host, uri.port)
      http.read_timeout = 10
      http.open_timeout = 5
      
      request = Net::HTTP::Post.new(uri)
      request['Content-Type'] = 'application/json'
      request.body = {
        session_id: @current_session_id
      }.to_json
      
      response = http.request(request)
      
      if response.code == '200'
        result = JSON.parse(response.body)
        if result['status'] == 'success'
          final_text = result['final_transcription'] || ""
          puts "🎤 Final transcription: '#{final_text}'" if ENV['DEBUG']
          record_streaming_success
          return clean_transcription(final_text)
        end
      end
      
    rescue => e
      puts "❌ Finalize error: #{e.message}"
      record_streaming_failure
    ensure
      @streaming_session_active = false
      @current_session_id = nil
    end
    
    ""
  end

  # Real-time streaming transcription for continuous audio
  def speech_to_text_streaming(audio_chunks)
    return "" if audio_chunks.empty?
    
    begin
      # Start streaming session
      unless start_streaming_session
        puts "⚠️ Failed to start streaming session, falling back to regular Whisper"
        return @audio_processor.speech_to_text_streaming(audio_chunks) if @audio_processor.respond_to?(:speech_to_text_streaming)
        return ""
      end
      
      # Stream chunks and collect partial results
      partial_texts = []
      audio_chunks.each do |chunk|
        partial = stream_audio_chunk(chunk)
        partial_texts << partial unless partial.empty?
      end
      
      # Get final result
      final_result = finalize_streaming_session
      
      return final_result.empty? ? partial_texts.last : final_result
      
    rescue => e
      puts "❌ Real-time streaming error: #{e.message}"
      record_streaming_failure
      return ""
    end
  end

  # Start streaming STT service
  def start_streaming_stt
    unless streaming_service_running?
      start_streaming_service
    end
    @streaming_active = true
  end

  # Stop streaming STT service
  def stop_streaming_stt
    @streaming_active = false
    # Keep service running for reuse, just mark as inactive
  end

  # Delegate TTS methods to AudioProcessor
  def text_to_speech(text)
    @audio_processor.text_to_speech(text)
  end

  def start_streaming_tts
    @audio_processor.start_streaming_tts
  end

  def stop_streaming_tts
    @audio_processor.stop_streaming_tts
  end

  def stream_text_chunk(text_chunk, is_final = false)
    @audio_processor.stream_text_chunk(text_chunk, is_final)
  end

  # Delegate other AudioProcessor methods
  def record_audio(duration = 5)
    @audio_processor.record_audio(duration)
  end

  def record_until_silence
    @audio_processor.record_until_silence
  end

  def has_speech?(audio_file)
    @audio_processor.has_speech?(audio_file)
  end

  def set_tts_engine(engine)
    @audio_processor.set_tts_engine(engine)
  end

  def start_kokoro_server
    @audio_processor.start_kokoro_server
  end

  def stop_kokoro_server
    @audio_processor.stop_kokoro_server
  end

  private

  def streaming_disabled?
    return false unless @streaming_disabled_until
    
    if Time.now < @streaming_disabled_until
      return true
    else
      # Re-enable streaming after timeout
      @streaming_disabled_until = nil
      @streaming_failures = 0
      puts "🔄 Re-enabling streaming service after timeout"
      return false
    end
  end

  def record_streaming_failure
    @streaming_failures += 1
    puts "⚠️ Streaming failure #{@streaming_failures}/#{@max_failures}"
    
    if @streaming_failures >= @max_failures
      @streaming_disabled_until = Time.now + 300  # Disable for 5 minutes
      puts "🚫 Disabling streaming service for 5 minutes due to repeated failures"
    end
  end

  def record_streaming_success
    if @streaming_failures > 0
      puts "✅ Streaming working again, resetting failure counter"
      @streaming_failures = 0
    end
  end

  def streaming_service_running?
    begin
      uri = URI("http://#{@streaming_host}:#{@streaming_port}/health")
      response = Net::HTTP.get_response(uri)
      response.code == '200'
    rescue
      false
    end
  end

  def start_streaming_service
    return if streaming_service_running?
    
    puts "🚀 Starting Whisper streaming service..."
    
    # Path to streaming service script
    service_script = File.join(__dir__, 'whisper_streaming_service.py')
    venv_python = File.expand_path('../venv_whisper_streaming/bin/python', __dir__)
    python_cmd = File.exist?(venv_python) ? venv_python : 'python3'
    
    begin
      # Start service in background
      @streaming_process = spawn(
        "#{python_cmd} #{service_script} server #{@streaming_port}",
        out: '/dev/null',
        err: '/dev/null',
        pgroup: true
      )
      
      Process.detach(@streaming_process)
      
      # Wait for service to be ready
      start_time = Time.now
      timeout = 30
      
      while (Time.now - start_time) < timeout
        if streaming_service_running?
          puts "✅ Whisper streaming service started"
          return true
        end
        sleep(0.5)
      end
      
      puts "❌ Whisper streaming service failed to start within #{timeout} seconds"
      stop_streaming_service
      return false
      
    rescue => e
      puts "❌ Failed to start Whisper streaming service: #{e.message}"
      return false
    end
  end

  def stop_streaming_service
    if @streaming_process
      begin
        Process.kill('TERM', -@streaming_process)
        sleep(1)
        
        begin
          Process.kill('KILL', -@streaming_process)
        rescue Errno::ESRCH
          # Process already dead
        end
        
        @streaming_process = nil
        puts "🛑 Whisper streaming service stopped"
      rescue => e
        puts "⚠️ Error stopping streaming service: #{e.message}"
      end
    end
  end

  def process_audio_file_streaming(audio_file)
    uri = URI("http://#{@streaming_host}:#{@streaming_port}/transcribe_file")
    http = Net::HTTP.new(uri.host, uri.port)
    http.read_timeout = 10  # Shorter timeout to prevent hanging
    http.open_timeout = 5   # Timeout for connection establishment
    
    request = Net::HTTP::Post.new(uri)
    request['Content-Type'] = 'application/json'
    request.body = {
      audio_file: audio_file,
      streaming: true
    }.to_json
    
    response = http.request(request)
    
    if response.code == '200'
      JSON.parse(response.body)
    else
      puts "❌ Streaming service error: #{response.code}"
      nil
    end
  end

  def process_audio_chunks_streaming(audio_chunks)
    uri = URI("http://#{@streaming_host}:#{@streaming_port}/transcribe_chunks")
    http = Net::HTTP.new(uri.host, uri.port)
    http.read_timeout = 10  # Shorter timeout to prevent hanging
    http.open_timeout = 5   # Timeout for connection establishment
    
    request = Net::HTTP::Post.new(uri)
    request['Content-Type'] = 'application/json'
    request.body = {
      audio_chunks: audio_chunks,
      sample_rate: @sample_rate,
      channels: @channels
    }.to_json
    
    response = http.request(request)
    
    if response.code == '200'
      JSON.parse(response.body)
    else
      puts "❌ Streaming chunks error: #{response.code}"
      nil
    end
  end

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
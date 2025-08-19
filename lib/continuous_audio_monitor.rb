require 'open3'
require 'json'
require 'tempfile'
require 'thread'
require 'net/http'
require 'uri'
require 'socket'

class ContinuousAudioMonitor
  def initialize(speech_callback, speaking_callback = nil)
    @speech_callback = speech_callback
    @speaking_callback = speaking_callback
    @listening = false
    @service_process = nil
    @monitoring_thread = nil
    @processing_thread = nil
    
    # Audio service configuration
    @service_script = File.join(__dir__, 'continuous_audio_service.py')
    @venv_python = File.expand_path('../venv_smart_audio/bin/python', __dir__)
    @python_cmd = File.exist?(@venv_python) ? @venv_python : 'python3'
    @service_host = '127.0.0.1'
    @service_port = 8768
    @service_startup_timeout = 10
    
    # Speech processing queue
    @speech_queue = Queue.new
    @last_agent_speaking_state = false
    
    puts "🎤 ContinuousAudioMonitor initialized"
  end

  def start_listening
    return true if @listening

    @listening = true
    puts "🎤 Starting continuous audio monitoring..."

    begin
      # Start the Python audio service
      start_audio_service
      
      # Start monitoring threads
      start_monitoring_threads
      
      puts "✅ Continuous audio monitoring started"
      return true
      
    rescue => e
      puts "❌ Failed to start continuous audio monitoring: #{e.message}"
      stop_listening
      return false
    end
  end

  def stop_listening
    return unless @listening

    @listening = false
    puts "🛑 Stopping continuous audio monitoring..."

    # Stop threads
    stop_monitoring_threads

    # Stop audio service
    stop_audio_service

    puts "✅ Continuous audio monitoring stopped"
  end

  def listening?
    @listening && @service_process && process_alive?(@service_process)
  end

  def set_agent_speaking(speaking)
    return unless listening?
    
    # Only update if state changed
    if speaking != @last_agent_speaking_state
      @last_agent_speaking_state = speaking
      
      begin
        # Send command to Python service
        http_post('/agent_speaking', { speaking: speaking })
        
        if speaking
          puts "🤖 Notified service: Agent started speaking"
        else
          puts "🎤 Notified service: Agent finished speaking"
        end
        
      rescue => e
        puts "⚠️ Failed to update agent speaking state: #{e.message}"
      end
    end
  end

  def get_status
    return { listening: false, error: "Service not running" } unless listening?
    
    begin
      response = http_get('/status')
      response
    rescue => e
      { listening: false, error: e.message }
    end
  end

  private

  def start_audio_service
    puts "🚀 Starting Python audio service HTTP server..."
    
    # Kill any existing service on this port first
    system("lsof -ti:#{@service_port} | xargs kill -9 2>/dev/null") rescue nil
    sleep(1)
    
    # Start service process in server mode
    @service_process = spawn(
      "#{@python_cmd} #{@service_script} server #{@service_port}",
      out: '/dev/null',
      err: '/dev/null'
    )

    # Wait for service to be ready
    puts "⏳ Waiting for audio service to start..."
    start_time = Time.now
    
    while (Time.now - start_time) < @service_startup_timeout
      if service_running?
        # Send start command to begin audio capture
        http_post('/start', {})
        puts "✅ Python audio service started (PID: #{@service_process})"
        return
      end
      sleep(0.5)
    end

    raise "Audio service failed to start within #{@service_startup_timeout} seconds"
  end

  def stop_audio_service
    return unless @service_process

    begin
      # Try graceful shutdown first
      http_post('/stop', {}) rescue nil
      sleep(1)

      # Force kill if still running
      if process_alive?(@service_process)
        Process.kill('TERM', @service_process)
        sleep(1)
        
        if process_alive?(@service_process)
          Process.kill('KILL', @service_process)
        end
      end

      Process.waitpid(@service_process) rescue nil
      puts "🛑 Python audio service stopped"
      
    rescue => e
      puts "⚠️ Error stopping audio service: #{e.message}"
    ensure
      @service_process = nil
    end
  end

  def start_monitoring_threads
    # Thread to monitor service health and get speech
    @monitoring_thread = Thread.new do
      monitor_service_loop
    end

    # Thread to process speech queue
    @processing_thread = Thread.new do
      process_speech_loop
    end
  end

  def stop_monitoring_threads
    # Stop monitoring thread
    if @monitoring_thread&.alive?
      @monitoring_thread.join(2) # Wait up to 2 seconds
      @monitoring_thread.kill if @monitoring_thread.alive?
    end

    # Stop processing thread
    if @processing_thread&.alive?
      @processing_thread.join(2) # Wait up to 2 seconds  
      @processing_thread.kill if @processing_thread.alive?
    end
  end

  def monitor_service_loop
    puts "🔄 Service monitoring thread started"
    
    while @listening
      begin
        # Check service health
        unless listening?
          puts "❌ Audio service died - stopping monitoring"
          @listening = false
          break
        end

        # Check for new speech from service
        check_for_speech

        sleep(0.1) # Check every 100ms

      rescue => e
        puts "Service monitoring error: #{e.message}"
        sleep(1) # Longer pause on error
      end
    end
    
    puts "🔄 Service monitoring thread ended"
  end

  def process_speech_loop
    puts "🔄 Speech processing thread started"
    
    while @listening
      begin
        # Process speech from queue
        speech_data = @speech_queue.pop

        break unless @listening # Exit if we're shutting down

        if speech_data
          process_speech_data(speech_data)
        end

      rescue => e
        puts "Speech processing error: #{e.message}"
        sleep(0.5)
      end
    end
    
    puts "🔄 Speech processing thread ended"
  end

  def check_for_speech
    begin
      response = http_get('/speech')
      
      if response && response['status'] == 'success' && response['speech']
        @speech_queue.push(response['speech'])
      end
      
    rescue => e
      # Silently ignore errors to avoid spam in monitoring loop
      # puts "Error checking for speech: #{e.message}" if ENV['DEBUG']
    end
  end

  def process_speech_data(speech_data)
    return unless speech_data && speech_data['file']

    audio_file = speech_data['file']
    
    begin
      # Only process if agent is not currently speaking
      if agent_speaking?
        puts "🎤 Queueing speech (agent is speaking): #{speech_data['duration']&.round(1)}s"
        # Re-queue for later processing
        Thread.new do
          sleep(0.5) # Brief delay
          @speech_queue.push(speech_data) if @listening
        end
        return
      end

      # Transcribe the audio
      if File.exist?(audio_file)
        puts "🔄 Processing speech: #{speech_data['duration']&.round(1)}s"
        
        # Use existing AudioProcessor for transcription
        audio_processor = AudioProcessor.new
        text = audio_processor.speech_to_text(audio_file)
        
        # Clean up temporary file
        File.delete(audio_file) if File.exist?(audio_file)
        
        # Call the speech callback if we have valid text
        if text && text.length > 2
          @speech_callback.call(text)
        else
          puts "❌ No clear speech detected in processed audio"
        end
      else
        puts "❌ Audio file not found: #{audio_file}"
      end

    rescue => e
      puts "❌ Error processing speech: #{e.message}"
      # Clean up file on error
      File.delete(audio_file) if audio_file && File.exist?(audio_file)
    end
  end

  def service_running?
    begin
      response = http_get('/status')
      # Just check if we get a valid response, not if it's listening yet
      response && response.key?('listening')
    rescue
      false
    end
  end

  def http_get(path)
    uri = URI("http://#{@service_host}:#{@service_port}#{path}")
    response = Net::HTTP.get_response(uri)
    
    if response.code == '200'
      JSON.parse(response.body)
    else
      nil
    end
  end

  def http_post(path, data)
    uri = URI("http://#{@service_host}:#{@service_port}#{path}")
    http = Net::HTTP.new(uri.host, uri.port)
    request = Net::HTTP::Post.new(uri)
    request['Content-Type'] = 'application/json'
    request.body = data.to_json
    
    response = http.request(request)
    
    if response.code == '200'
      JSON.parse(response.body)
    else
      nil
    end
  end

  def process_alive?(pid)
    return false unless pid
    
    begin
      Process.getpgid(pid)
      true
    rescue Errno::ESRCH
      false
    end
  end

  def agent_speaking?
    return false unless @speaking_callback
    @speaking_callback.call
  end
end
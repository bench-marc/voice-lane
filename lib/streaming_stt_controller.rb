#!/usr/bin/env ruby

require 'net/http'
require 'uri'
require 'json'

class StreamingSTTController
  """
  Ruby controller for managing streaming speech-to-text services
  Provides a clean interface for starting/stopping streaming services
  and handling real-time transcription callbacks
  """

  def initialize(host: '127.0.0.1', port: 8769, model: 'tiny', language: nil)
    @host = host
    @port = port
    @model = model
    @language = language
    @service_process = nil
    @callback = nil
    @transcription_thread = nil
    @active = false
    
    # HTTP connection pooling for better performance
    @http_connection = nil
    @connection_mutex = Mutex.new
    @max_retries = 3
    @retry_delay = 0.5
  end

  def start_service
    """
    Start the Python streaming audio service
    """
    return true if service_running?

    puts "🚀 Starting streaming STT service on #{@host}:#{@port}"

    venv_python = File.expand_path('../venv_whisper_streaming/bin/python', __dir__)
    service_script = File.join(__dir__, 'streaming_audio_service.py')

    unless File.exist?(venv_python)
      puts "❌ Python virtual environment not found: #{venv_python}"
      return false
    end

    unless File.exist?(service_script)
      puts "❌ Service script not found: #{service_script}"
      return false
    end

    begin
      # Build command with parameters
      cmd_parts = [
        venv_python,
        service_script,
        'server',
        @port.to_s,
        @model
      ]
      cmd_parts << @language if @language

      # Start service in background with process group
      @service_process = spawn(
        cmd_parts.join(' '),
        out: File.expand_path('../tmp/streaming_stt.log', __dir__),
        err: File.expand_path('../tmp/streaming_stt_error.log', __dir__),
        pgroup: true  # Create new process group to avoid parent signals
      )

      # Don't detach so we can manage the process properly
      # Process.detach(@service_process)

      # Wait for service to start
      puts "⏳ Waiting for service to start..."
      start_time = Time.now
      timeout = 15

      while (Time.now - start_time) < timeout
        if service_running?
          puts "✅ Streaming STT service started successfully (PID: #{@service_process})"
          return true
        end
        sleep(0.5)
      end

      puts "❌ Service failed to start within #{timeout} seconds"
      stop_service
      return false

    rescue => e
      puts "❌ Failed to start service: #{e.message}"
      return false
    end
  end

  def stop_service
    """
    Stop the Python streaming audio service
    """
    if @service_process
      begin
        # First try graceful shutdown
        Process.kill('TERM', -@service_process)
        sleep(2)

        # Force kill if still running
        begin
          Process.kill('KILL', -@service_process)
        rescue Errno::ESRCH
          # Process already dead
        end

        puts "🛑 Streaming STT service stopped"
        @service_process = nil

      rescue => e
        puts "⚠️ Error stopping service: #{e.message}"
      end
    end
  end

  def service_running?
    """
    Check if the streaming service is running and healthy
    """
    perform_http_request("/health", :get) do |response|
      response.code == '200'
    end || false
  end

  def get_service_status
    """
    Get detailed service status and statistics
    """
    perform_http_request("/status", :get) do |response|
      if response.code == '200'
        JSON.parse(response.body)
      else
        {"status" => "error", "message" => "Service not available"}
      end
    end || {"status" => "error", "message" => "Connection failed"}
  end

  def transcribe_file(audio_file)
    """
    Transcribe an audio file using the streaming service
    """
    return {"status" => "error", "message" => "File not found"} unless File.exist?(audio_file)
    return {"status" => "error", "message" => "Service not running"} unless service_running?

    begin
      uri = URI("http://#{@host}:#{@port}/transcribe_file")
      http = Net::HTTP.new(uri.host, uri.port)
      http.read_timeout = 30

      request = Net::HTTP::Post.new(uri)
      request['Content-Type'] = 'application/json'
      request.body = {
        audio_file: File.expand_path(audio_file)
      }.to_json

      response = http.request(request)

      if response.code == '200'
        result = JSON.parse(response.body)
        puts "📝 Transcribed: '#{result['transcription']}' (#{result['processing_time']&.round(2)}s)"
        return result
      else
        return {"status" => "error", "message" => "HTTP error: #{response.code}"}
      end

    rescue => e
      return {"status" => "error", "message" => e.message}
    end
  end

  def clear_transcription_results
    """
    Clear any pending transcription results from previous sessions
    """
    return false unless service_running?
    
    perform_http_request("/results", :get) do |response|
      if response.code == '200'
        data = JSON.parse(response.body)
        results = data['results'] || []
        if results.any?
          puts "🧹 Cleared #{results.length} stale STT results from previous session" if ENV['DEBUG']
        else
          puts "✅ No stale STT results to clear" if ENV['DEBUG']
        end
        true
      else
        puts "⚠️ Failed to clear STT results: HTTP #{response.code}" if ENV['DEBUG']
        false
      end
    end || false
  end
  
  def reset_audio_stream
    """
    Reset the audio stream to clear any buffered audio from previous sessions
    This stops and restarts the audio capture to ensure clean state
    """
    return false unless service_running?
    
    begin
      puts "🔄 Resetting STT audio stream..." if ENV['DEBUG']
      
      # First, stop any active streaming
      if @active
        puts "🛑 Stopping active streaming before reset..." if ENV['DEBUG']
        stop_streaming_transcription
        
        # Wait for streaming to actually stop
        timeout_count = 0
        while @active && timeout_count < 10
          sleep(0.1)
          timeout_count += 1
        end
        
        if @active
          puts "⚠️ Streaming did not stop cleanly, forcing reset..." if ENV['DEBUG']
          @active = false
        end
      end
      
      # Make HTTP request to reset audio stream
      result = perform_http_request("/reset_audio", :post, {}.to_json) do |response|
        if response.code == '200'
          puts "✅ Audio stream reset successful" if ENV['DEBUG']
          true
        else
          puts "⚠️ Audio stream reset failed: HTTP #{response.code}" if ENV['DEBUG']
          false
        end
      end
      
      if result
        return true
      else
        # Fallback: try restarting the entire service
        puts "🔄 Attempting service restart as fallback..." if ENV['DEBUG']
        restart_service
        return true
      end
      
    rescue => e
      puts "❌ Error resetting audio stream: #{e.message}" if ENV['DEBUG']
      # Fallback: restart service
      puts "🔄 Attempting service restart as fallback..." if ENV['DEBUG']
      restart_service
      return true
    end
  end

  def start_streaming_transcription(callback = nil, &block)
    """
    Start real-time streaming transcription with callback
    """
    @callback = callback || block

    unless @callback
      puts "❌ No callback provided for streaming transcription"
      return false
    end

    return false if @active
    return false unless service_running?

    puts "🎤 Starting streaming transcription..."

    result = perform_http_request("/start_streaming", :post, {}.to_json) do |response|
      if response.code == '200'
        @active = true
        puts "✅ Streaming transcription started"

        # Start monitoring thread
        start_monitoring_thread

        true
      else
        puts "❌ Failed to start streaming: HTTP #{response.code}"
        false
      end
    end

    result || false
  end

  def stop_streaming_transcription
    """
    Stop real-time streaming transcription
    """
    return unless @active

    puts "🛑 Stopping streaming transcription..."

    perform_http_request("/stop_streaming", :post, {}.to_json) do |response|
      if response.code == '200'
        puts "✅ Streaming transcription stopped"
      else
        puts "⚠️ HTTP error stopping streaming: #{response.code}"
      end
    end

    @active = false
    @callback = nil

    # Wait for monitoring thread to finish
    if @transcription_thread && @transcription_thread.alive?
      @transcription_thread.join(2)  # 2 second timeout
    end
    
    # Close HTTP connection when streaming stops to free resources
    close_http_connection
  end

  def active?
    """
    Check if streaming transcription is currently active
    """
    @active
  end

  def restart_service
    """
    Restart the streaming service
    """
    puts "🔄 Restarting streaming STT service..."
    stop_service
    sleep(2)
    start_service
  end

  private

  def get_http_connection
    """
    Get or create a reusable HTTP connection with keep-alive
    """
    @connection_mutex.synchronize do
      if @http_connection.nil? || !@http_connection.started?
        @http_connection = Net::HTTP.new(@host, @port)
        @http_connection.keep_alive_timeout = 30
        @http_connection.read_timeout = 10
        @http_connection.open_timeout = 5
        @http_connection.start
      end
      @http_connection
    end
  rescue => e
    puts "⚠️ HTTP connection error: #{e.message}" if ENV['DEBUG']
    @http_connection = nil
    nil
  end

  def close_http_connection
    """
    Close the HTTP connection
    """
    @connection_mutex.synchronize do
      if @http_connection&.started?
        @http_connection.finish
      end
      @http_connection = nil
    end
  rescue => e
    puts "⚠️ Error closing HTTP connection: #{e.message}" if ENV['DEBUG']
  end

  def perform_http_request(path, method = :get, body = nil, headers = {})
    """
    Perform HTTP request with connection pooling and retry logic
    """
    retries = 0
    
    while retries < @max_retries
      begin
        http = get_http_connection
        return nil unless http
        
        request = case method
                 when :get
                   Net::HTTP::Get.new(path)
                 when :post
                   req = Net::HTTP::Post.new(path)
                   req['Content-Type'] = 'application/json'
                   req.body = body if body
                   req
                 else
                   raise "Unsupported HTTP method: #{method}"
                 end
        
        headers.each { |k, v| request[k] = v }
        
        response = http.request(request)
        
        if block_given?
          return yield(response)
        else
          return response
        end
        
      rescue Net::ReadTimeout, Net::OpenTimeout, Errno::ECONNRESET, Errno::ECONNREFUSED => e
        retries += 1
        puts "⚠️ HTTP request failed (attempt #{retries}/#{@max_retries}): #{e.message}" if ENV['DEBUG']
        
        # Close connection on error to force reconnect
        close_http_connection
        
        if retries < @max_retries
          sleep(@retry_delay * retries)  # Exponential backoff
        else
          puts "❌ HTTP request failed after #{@max_retries} attempts"
          return nil
        end
      rescue => e
        puts "❌ Unexpected HTTP error: #{e.message}" if ENV['DEBUG']
        close_http_connection
        return nil
      end
    end
    
    nil
  end

  def start_monitoring_thread
    """
    Start background thread to monitor for transcription results
    Polls the streaming service for real-time transcription results
    """
    @transcription_thread = Thread.new do
      puts "🔄 Transcription monitoring thread started"

      while @active
        begin
          # Poll for transcription results
          results = get_transcription_results
          
          # Process each result
          results.each do |result|
            if @callback && result['text'] && !result['text'].empty?
              puts "📝 Received transcription: '#{result['text']}' (#{result['duration']&.round(2)}s)" if ENV['DEBUG']
              @callback.call(result['text'], result['duration'])
            end
          end

          # Check service health periodically, but only stop if consistently unhealthy
          if results.empty?
            status = get_service_status
            unless status["status"] == "running"
              puts "⚠️ Service not healthy: #{status['status']} - attempting restart..." if ENV['DEBUG']
              
              # Try to restart the service instead of just stopping
              if restart_service
                puts "✅ Service restarted successfully" if ENV['DEBUG']
                next  # Continue monitoring
              else
                puts "❌ Failed to restart service - stopping monitoring"
                @active = false
                break
              end
            end
          end

          # Poll frequently for low latency (100ms)
          sleep(0.1)

        rescue => e
          puts "❌ Monitoring thread error: #{e.message}"
          @active = false
          break
        end
      end

      puts "🔄 Transcription monitoring thread ended"
    end
  end

  def get_transcription_results
    """
    Poll the streaming service for pending transcription results
    """
    perform_http_request("/results", :get) do |response|
      if response.code == '200'
        data = JSON.parse(response.body)
        data['results'] || []
      else
        []
      end
    end || []
  end
end
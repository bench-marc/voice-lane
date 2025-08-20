require 'open3'
require 'json'
require 'tempfile'
require 'thread'
require 'net/http'
require 'uri'
require 'socket'
require 'base64'

class KrokoStreamingAudioMonitor
  def initialize(speech_callback, speaking_callback = nil)
    @speech_callback = speech_callback
    @speaking_callback = speaking_callback
    @listening = false
    @streaming_active = false
    
    # Kroko STT service configuration
    @kroko_host = '127.0.0.1'
    @kroko_port = 8769
    @kroko_startup_timeout = 30
    
    # Audio capture configuration
    @sample_rate = 16000
    @chunk_size = 320  # 20ms at 16kHz
    @format = 'paInt16'
    
    # Speech processing
    @speech_buffer = ""
    @partial_transcripts = []
    @last_final_time = 0
    @silence_timeout = 2.0  # Seconds of silence before finalizing speech
    
    # Threads and processes
    @audio_capture_thread = nil
    @kroko_stt_process = nil
    
    puts "🎤 KrokoStreamingAudioMonitor initialized"
  end

  def start_listening
    return true if @listening
    
    puts "🚀 Starting Kroko streaming audio monitoring..."
    
    begin
      # Start Kroko STT server if not running
      unless kroko_server_running?
        puts "🔄 Starting Kroko STT server..."
        start_kroko_server
      end
      
      # Start streaming mode
      start_kroko_streaming
      
      # Start audio capture
      start_audio_capture
      
      @listening = true
      puts "✅ Kroko streaming audio monitoring started"
      return true
      
    rescue => e
      puts "❌ Failed to start Kroko streaming monitoring: #{e.message}"
      cleanup
      return false
    end
  end

  def stop_listening
    return unless @listening
    
    puts "🛑 Stopping Kroko streaming audio monitoring..."
    
    @listening = false
    @streaming_active = false
    
    # Stop streaming mode
    stop_kroko_streaming
    
    # Stop audio capture
    if @audio_capture_thread
      @audio_capture_thread.kill
      @audio_capture_thread.join(1)
      @audio_capture_thread = nil
    end
    
    cleanup
    puts "✅ Kroko streaming monitoring stopped"
  end

  def agent_speaking?
    return false unless @speaking_callback
    @speaking_callback.call
  end

  def listening?
    @listening
  end

  def set_agent_speaking(speaking)
    # Method to match ContinuousAudioMonitor interface
    # In Kroko streaming, we don't need to notify a separate service
    # since speech detection is handled directly in the audio monitor
    puts "🤖 Agent speaking state: #{speaking}" if ENV['DEBUG']
  end

  def get_status
    {
      listening: @listening,
      streaming_active: @streaming_active,
      kroko_server_running: kroko_server_running?,
      partial_transcripts: @partial_transcripts.size,
      last_final_time: @last_final_time
    }
  end

  private

  def kroko_server_running?
    begin
      uri = URI("http://#{@kroko_host}:#{@kroko_port}/health")
      http = Net::HTTP.new(uri.host, uri.port)
      http.read_timeout = 2
      response = http.get(uri.path)
      return response.code == '200'
    rescue
      return false
    end
  end

  def start_kroko_server
    return if kroko_server_running?
    
    service_path = File.expand_path('../lib/kroko_stt_service.py', __dir__)
    unless File.exist?(service_path)
      raise "Kroko STT service not found: #{service_path}"
    end
    
    # Kill any existing processes on the port first
    kill_existing_server
    
    # Use virtual environment python if available
    venv_python = File.expand_path('../venv_kroko/bin/python', __dir__)
    python_cmd = File.exist?(venv_python) ? venv_python : 'python3'
    
    # Start server in background with visible output for debugging
    puts "🔄 Starting server with: #{python_cmd} #{service_path} --host #{@kroko_host} --port #{@kroko_port}"
    @kroko_stt_process = spawn(
      python_cmd, service_path,
      '--host', @kroko_host,
      '--port', @kroko_port.to_s,
      pgroup: true,
      out: ENV['DEBUG'] ? $stdout : '/dev/null',
      err: ENV['DEBUG'] ? $stderr : '/dev/null'
    )
    
    # Wait for server to be ready with progress indicators
    ready = false
    attempts = 0
    max_attempts = @kroko_startup_timeout * 2
    
    puts "⏳ Waiting for Kroko server to be ready (max #{@kroko_startup_timeout}s)..."
    
    max_attempts.times do |i|
      attempts = i + 1
      sleep(0.5)
      
      # Show progress every 2 seconds (4 attempts)
      if attempts % 4 == 0
        puts "⏳ Still waiting... (#{attempts / 2}s elapsed)"
      end
      
      if kroko_server_running?
        ready = true
        break
      end
    end
    
    if ready
      puts "✅ Kroko STT server ready after #{attempts / 2}s"
    else
      puts "❌ Kroko STT server failed to start within #{@kroko_startup_timeout} seconds"
      
      # Try to get process status for debugging
      if @kroko_stt_process
        begin
          Process.kill(0, @kroko_stt_process)
          puts "⚠️ Process is still running but not responding on port #{@kroko_port}"
        rescue Errno::ESRCH
          puts "⚠️ Process appears to have died"
        end
      end
      
      raise "Kroko STT server startup failed"
    end
    
    puts "✅ Kroko STT server started"
  end

  def kill_existing_server
    begin
      # Use lsof to find process using the port
      output = `lsof -ti:#{@kroko_port} 2>/dev/null`.strip
      if output && !output.empty?
        pids = output.split("\n")
        pids.each do |pid|
          puts "🔄 Killing existing server process #{pid} on port #{@kroko_port}"
          Process.kill('TERM', pid.to_i)
          sleep(0.5)
          # Force kill if still running
          begin
            Process.kill('KILL', pid.to_i)
          rescue Errno::ESRCH
            # Process already dead
          end
        end
        sleep(1)  # Give the port time to be released
      end
    rescue => e
      puts "⚠️ Error cleaning up port #{@kroko_port}: #{e.message}"
    end
  end

  def start_kroko_streaming
    uri = URI("http://#{@kroko_host}:#{@kroko_port}/start_streaming")
    http = Net::HTTP.new(uri.host, uri.port)
    response = http.post(uri.path, '')
    
    if response.code == '200'
      result = JSON.parse(response.body)
      if result['status'] == 'success'
        @streaming_active = true
        puts "✅ Kroko streaming started"
        return true
      end
    end
    
    raise "Failed to start Kroko streaming: #{response.body}"
  end

  def stop_kroko_streaming
    return unless @streaming_active
    
    begin
      uri = URI("http://#{@kroko_host}:#{@kroko_port}/stop_streaming")
      http = Net::HTTP.new(uri.host, uri.port)
      response = http.post(uri.path, '')
      
      if response.code == '200'
        puts "✅ Kroko streaming stopped"
      end
    rescue => e
      puts "⚠️ Error stopping Kroko streaming: #{e.message}"
    end
    
    @streaming_active = false
  end

  def start_audio_capture
    @audio_capture_thread = Thread.new do
      begin
        capture_audio_with_pyaudio
      rescue => e
        puts "❌ Audio capture error: #{e.message}" if @listening
      end
    end
  end

  def capture_audio_with_pyaudio
    # Use a Python script for audio capture since Ruby doesn't have good real-time audio libs
    python_script = <<~PYTHON
import pyaudio
import sys
import time
import json
import base64

# Audio configuration
SAMPLE_RATE = #{@sample_rate}
CHUNK_SIZE = #{@chunk_size}
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Initialize PyAudio
p = pyaudio.PyAudio()

try:
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    
    print("AUDIO_READY", flush=True)
    
    while True:
        # Read audio chunk
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        
        # Encode as base64 and send to Ruby
        encoded_data = base64.b64encode(data).decode('utf-8')
        chunk_info = {
            'audio_data': encoded_data,
            'timestamp': time.time(),
            'chunk_size': CHUNK_SIZE,
            'sample_rate': SAMPLE_RATE
        }
        
        print("AUDIO_CHUNK:" + json.dumps(chunk_info), flush=True)
        
except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"AUDIO_ERROR:{e}", flush=True, file=sys.stderr)
finally:
    if 'stream' in locals():
        stream.stop_stream()
        stream.close()
    p.terminate()
    PYTHON
    
    # Start Python audio capture process (use same python as Kroko server)
    venv_python = File.expand_path('../venv_kroko/bin/python', __dir__)
    python_cmd = File.exist?(venv_python) ? venv_python : 'python3'
    
    puts "🎤 Starting Python audio capture with: #{python_cmd}" if ENV['DEBUG']
    stdin, stdout, stderr, @audio_process = Open3.popen3(python_cmd, '-c', python_script)
    stdin.close
    
    # Read audio chunks and process them
    puts "🎤 Reading audio chunks from Python process..." if ENV['DEBUG']
    stdout.each_line do |line|
      line = line.strip
      next if line.empty? || !@listening
      
      if line == "AUDIO_READY"
        puts "🎤 Audio capture ready"
        next
      end
      
      if line.start_with?("AUDIO_CHUNK:")
        chunk_json = line.sub("AUDIO_CHUNK:", "")
        begin
          chunk_data = JSON.parse(chunk_json)
          process_audio_chunk(chunk_data) if @streaming_active
        rescue JSON::ParserError => e
          puts "⚠️ Failed to parse audio chunk: #{e.message}"
        end
      elsif line.start_with?("AUDIO_ERROR:")
        error = line.sub("AUDIO_ERROR:", "")
        puts "❌ Audio capture error: #{error}"
        break
      else
        puts "🔍 Unknown audio output: #{line}" if ENV['DEBUG']
      end
    end
    
    # Also read stderr for Python errors
    if stderr
      Thread.new do
        stderr.each_line do |error_line|
          puts "❌ Python audio error: #{error_line.strip}" unless error_line.strip.empty?
        end
      end
    end
  rescue => e
    puts "❌ Audio capture thread error: #{e.message}" if @listening
  ensure
    # Clean up audio process
    if @audio_process
      begin
        Process.kill('TERM', @audio_process.pid)
        Process.wait(@audio_process.pid)
      rescue
        # Process already terminated
      end
      @audio_process = nil
    end
  end

  def process_audio_chunk(chunk_data)
    return unless @streaming_active && !agent_speaking?
    
    puts "🎤 Processing audio chunk: #{chunk_data['chunk_size']} samples" if ENV['DEBUG']
    
    begin
      # Send audio chunk to Kroko streaming endpoint
      uri = URI("http://#{@kroko_host}:#{@kroko_port}/stream_audio")
      http = Net::HTTP.new(uri.host, uri.port)
      http.read_timeout = 1
      
      request = Net::HTTP::Post.new(uri)
      request['Content-Type'] = 'application/json'
      request.body = chunk_data.to_json
      
      response = http.request(request)
      
      if response.code == '200'
        result = JSON.parse(response.body)
        puts "🎤 Kroko response: #{result}" if ENV['DEBUG']
        handle_kroko_streaming_response(result)
      else
        puts "⚠️ Kroko streaming error: #{response.code} #{response.message}"
        puts "Response body: #{response.body}" if ENV['DEBUG']
      end
      
    rescue Net::ReadTimeout
      # Ignore timeouts - they're expected with streaming
    rescue => e
      puts "❌ Error processing audio chunk: #{e.message}"
      puts "Backtrace: #{e.backtrace.first(3).join("\n")}" if ENV['DEBUG']
    end
  end

  def handle_kroko_streaming_response(result)
    return unless result['status'] == 'success'
    
    is_speech = result['is_speech']
    timestamp = result['timestamp']
    
    # Handle transcription if available
    if result['transcript']
      transcript = result['transcript'].strip
      
      if result['partial']
        # Partial transcription - accumulate
        unless @partial_transcripts.include?(transcript)
          @partial_transcripts << transcript
          puts "🎤 Partial: #{transcript}"
        end
      elsif result['final']
        # Final transcription - process
        @last_final_time = timestamp
        
        # If we have a final transcript from the service, use it
        # Otherwise, use accumulated partial transcripts
        if transcript && !transcript.empty?
          full_transcript = transcript
        else
          # Use accumulated partial transcripts
          full_transcript = @partial_transcripts.join(' ').strip
        end
        
        unless full_transcript.empty?
          puts "🎤 Hotel (Kroko, FINAL): #{full_transcript}"
          
          # Call the speech callback - this triggers the hotel agent response
          puts "📞 Calling speech callback with: '#{full_transcript}'"
          @speech_callback.call(full_transcript) if @speech_callback
        else
          puts "⚠️ Final transcript is empty, but finalizing anyway"
          # Even if empty, finalize any accumulated partial transcripts
          if @partial_transcripts.any?
            full_transcript = @partial_transcripts.join(' ').strip
            unless full_transcript.empty?
              puts "🎤 Hotel (Kroko, FINAL from partials): #{full_transcript}"
              puts "📞 Calling speech callback with: '#{full_transcript}'"
              @speech_callback.call(full_transcript) if @speech_callback
            end
          end
        end
        
        # Clear partial transcripts after finalizing
        @partial_transcripts.clear
      end
    end
    
    # Handle silence detection
    unless is_speech
      # If we have partial transcripts and enough silence time has passed
      if @partial_transcripts.any? && (timestamp - @last_final_time) > @silence_timeout
        # Finalize partial transcripts
        full_transcript = @partial_transcripts.join(' ').strip
        
        unless full_transcript.empty?
          puts "🎤 Hotel (Kroko, timeout): #{full_transcript}"
          @speech_callback.call(full_transcript) if @speech_callback
        end
        
        @partial_transcripts.clear
        @last_final_time = timestamp
      end
    end
  end

  def cleanup
    # Stop Kroko streaming
    stop_kroko_streaming
    
    # Stop audio capture
    if @audio_capture_thread
      @audio_capture_thread.kill
      @audio_capture_thread = nil
    end
    
    # Clean up audio process
    if @audio_process
      begin
        Process.kill('TERM', @audio_process.pid)
        Process.wait(@audio_process.pid)
      rescue
        # Process already terminated
      end
      @audio_process = nil
    end
  end
end
require_relative 'audio_processor'
require_relative 'hotel_booking_agent'
require_relative 'streaming_stt_controller'

class HotelAutoVoiceAgent
  def initialize
    @audio_processor = AudioProcessor.new
    @ollama_client = HotelBookingOllamaClient.new
    @speaking = false
    @call_active = false
    
    # Start Kokoro TTS server for fast speech generation
    puts "🚀 Initializing TTS engine..."
    @audio_processor.start_kokoro_server
    
    # Initialize streaming STT controller for real-time transcription
    @streaming_stt = StreamingSTTController.new(
      host: '127.0.0.1',
      port: 8770,  # Use streaming service port
      model: 'tiny',  # Fast model for hotel calls
      language: nil   # Auto-detect language
    )
    
    # Set up signal handlers for graceful shutdown
    setup_signal_handlers
  end

  def start_auto_call(guest_name: nil, booking_reference: nil, hotel_name: nil, checkin_date: nil, checkout_date: nil)
    puts "\n" + "="*60
    puts "📞 LANES&PLANES AUTO VOICE HOTEL AGENT"
    puts "="*60
    puts "🎤 Fully automatic voice conversation"
    puts "👤 Guest: #{guest_name}" if guest_name
    puts "📋 Booking: #{booking_reference}" if booking_reference
    puts "🏨 Hotel: #{hotel_name}" if hotel_name
    puts "📅 Check-in: #{checkin_date}" if checkin_date
    puts "📅 Check-out: #{checkout_date}" if checkout_date
    puts "="*60
    puts "🎤 Agent will listen first for hotel greeting, then respond"
    puts "⏹️  Say 'end call' or press Ctrl+C to stop"
    puts "="*60 + "\n"

    # Store booking details for context
    @ollama_client.set_booking_details(
      guest_name: guest_name,
      booking_reference: booking_reference,
      hotel_name: hotel_name,
      checkin_date: checkin_date,
      checkout_date: checkout_date
    )

    @call_active = true
    
    puts "🎤 Starting streaming transcription for hotel staff..."
    
    # Start streaming STT service if needed
    unless @streaming_stt.service_running?
      puts "🚀 Starting streaming STT service..."
      unless @streaming_stt.start_service
        puts "❌ Failed to start streaming STT service"
        return false
      end
    end
    
    # Clear any stale transcription results and reset audio stream from previous sessions
    puts "🧹 Clearing STT results and resetting audio stream..." if ENV['DEBUG']
    @streaming_stt.clear_transcription_results
    @streaming_stt.reset_audio_stream
    
    # Allow time for reset to complete before starting new session
    sleep(0.5)
    
    # Start streaming transcription with callback
    streaming_callback = method(:handle_streaming_hotel_speech)
    unless @streaming_stt.start_streaming_transcription(&streaming_callback)
      puts "❌ Failed to start streaming transcription"
      return false
    end
    
    puts "✅ Real-time streaming transcription active"
    puts "🎤 System is now actively listening with minimal latency..."
    puts "💬 Transcription happens while hotel staff speaks (streaming mode)"
    puts "⏹️  Press Ctrl+C to stop the call at any time"
    puts ""
    
    # Keep call active
    begin
      loop_count = 0
      while @call_active && @streaming_stt.active?
        sleep(0.5)
        loop_count += 1
        
        # Show activity indicator every 10 seconds
        if loop_count % 20 == 0  # Every 10 seconds (20 * 0.5s)
          status = @streaming_stt.get_service_status
          if status && status['statistics']
            stats = status['statistics']
            puts "🎤 Streaming... (#{stats['total_processed']} utterances processed, avg #{(stats['avg_processing_time'] * 1000)&.round(0)}ms latency)" if ENV['DEBUG']
          else
            puts "🎤 Streaming... (waiting for hotel staff to speak)" if ENV['DEBUG']
          end
        end
        
        # Check if objectives are met periodically
        if call_objectives_met? && @ollama_client.conversation_length >= 4
          puts "\n✅ Call objectives completed successfully!"
          
          # Generate confirmation statement
          confirmation = "Perfect, Lanes&Planes will be billed directly. Thank you!"
          puts "🤖 Agent: #{confirmation}"
          speak_response(confirmation)
          
          summary = @ollama_client.generate_call_summary
          puts "\n📋 Call Summary:"
          puts summary
          
          stop_call
          break
        end
      end
    rescue Interrupt
      puts "\n📞 Call interrupted by user"
    ensure
      stop_call
    end
    
    puts "\n📞 Call completed!"
  end

  def stop_call
    @call_active = false
    if @streaming_stt.active?
      puts "🛑 Stopping streaming transcription..."
      @streaming_stt.stop_streaming_transcription
    end
    @audio_processor.stop_kokoro_server
    puts "📞 Call stopped"
  end

  private

  def handle_streaming_hotel_speech(text, duration)
    """
    Handle real-time transcription from streaming STT service
    Called immediately when silence is detected and speech is transcribed
    """
    return if @speaking || !@call_active || text.nil? || text.strip.empty?
    
    # Debug timing: Reset all timing variables for this interaction
    if ENV['DEBUG']
      @debug_speech_received_at = Time.now
      @debug_llm_start = nil
      @debug_first_chunk_time = nil
      @debug_llm_end = nil
      @debug_first_audio_time = nil
      @debug_first_audio_played = false
      @debug_tts_init_start = nil
      @debug_tts_init_end = nil
      @debug_tts_complete_start = nil
      @debug_tts_complete_end = nil
      @debug_pipeline_end = nil
      @debug_first_chunk_received = false
      @debug_first_audio_sent = false
      # Reset audio processor debug timing as well
      @audio_processor.instance_variable_set(:@debug_first_playback_time, nil)
      @audio_processor.instance_variable_set(:@debug_track_first_playback, nil)
      puts "📥 [DEBUG] Speech received: '#{text}' (#{duration&.round(2)}s audio) at #{@debug_speech_received_at.strftime('%H:%M:%S.%3N')}"
    end
    
    puts "🏨 Hotel (streaming #{duration&.round(2)}s): #{text}"
    
    # Process the speech using existing logic
    handle_hotel_speech(text)
  end

  def handle_hotel_speech(text)
    return if @speaking || !@call_active || text.nil? || text.strip.empty?
    
    # If this is the first message (hotel greeting), generate opening response
    # if @ollama_client.conversation_length == 0
    #   opening = @ollama_client.generate_opening_statement
    #   puts "🤖 Agent: #{opening}"
    #   speak_response(opening)
    #
    #   # Add opening statement to conversation history
    #   @ollama_client.add_assistant_message(opening)
    #   return
    # end
    
    # Check for call end signals
    if text.downcase.match?(/(goodbye|thank you.*day|have a good|bye|end.*call)/)
      puts "📞 Hotel is ending the call..."
      
      # Generate brief closing response
      closing = "Thank you, goodbye!"
      puts "🤖 Agent: #{closing}"
      speak_response(closing)
      
      stop_call
      return
    end

    # Generate agent response using streaming
    puts "🤖 Agent: " # Start the line
    
    # Debug timing: LLM processing start
    if ENV['DEBUG']
      @debug_llm_start = Time.now
      puts "🧠 [DEBUG] Starting LLM processing at #{@debug_llm_start.strftime('%H:%M:%S.%3N')}"
    end
    
    @speaking = true
    # Streaming service handles agent speaking state internally
    
    # Ensure clean session start by stopping any previous streaming
    @audio_processor.stop_streaming_tts
    sleep(0.1) # Brief pause to ensure cleanup
    
    # Debug timing: TTS initialization
    if ENV['DEBUG']
      @debug_tts_init_start = Time.now
      puts "🔊 [DEBUG] Initializing TTS at #{@debug_tts_init_start.strftime('%H:%M:%S.%3N')}"
    end
    
    @audio_processor.start_streaming_tts
    
    if ENV['DEBUG']
      @debug_tts_init_end = Time.now
      puts "🔊 [DEBUG] TTS initialized in #{((@debug_tts_init_end - @debug_tts_init_start) * 1000).round(0)}ms"
    end
    
    agent_reply = ""
    final_response = nil
    @debug_first_chunk_received = false
    @debug_first_audio_sent = false
    
    begin
      final_response = @ollama_client.generate_response_stream(text) do |chunk, is_done|
        if !is_done && !chunk.empty?
          # Debug timing: First LLM chunk received
          if ENV['DEBUG'] && !@debug_first_chunk_received
            @debug_first_chunk_received = true
            @debug_first_chunk_time = Time.now
            puts "🧠 [DEBUG] First LLM chunk received in #{((@debug_first_chunk_time - @debug_llm_start) * 1000).round(0)}ms"
          end
          
          # Stream chunk to TTS and display
          print chunk
          STDOUT.flush
          agent_reply += chunk
          
          # Debug timing: First audio sent to TTS
          if ENV['DEBUG'] && !@debug_first_audio_sent
            @debug_first_audio_sent = true
            @debug_first_audio_time = Time.now
            puts "🔊 [DEBUG] First text sent to TTS in #{((@debug_first_audio_time - @debug_llm_start) * 1000).round(0)}ms"
          end
          
          @audio_processor.stream_text_chunk(chunk, false)
          
          # Set up callback for first audio playback tracking
          if ENV['DEBUG'] && !@debug_first_audio_played
            @debug_first_audio_played = true  # Mark as tracked to avoid duplicates
            # Set a flag in audio processor to track when first playback actually starts
            @audio_processor.instance_variable_set(:@debug_first_audio_time, @debug_first_audio_time)
            @audio_processor.instance_variable_set(:@debug_track_first_playback, true)
          end
        elsif is_done
          # Debug timing: LLM processing completed
          if ENV['DEBUG']
            @debug_llm_end = Time.now
            puts "🧠 [DEBUG] LLM processing completed in #{((@debug_llm_end - @debug_llm_start) * 1000).round(0)}ms"
          end
          
          # Final processing
          @audio_processor.stream_text_chunk("", true)
          
          # Debug timing: TTS completion
          if ENV['DEBUG']
            @debug_tts_complete_start = Time.now
            puts "🔊 [DEBUG] Finalizing TTS at #{@debug_tts_complete_start.strftime('%H:%M:%S.%3N')}"
          end
          
          @audio_processor.stop_streaming_tts
          
          if ENV['DEBUG']
            @debug_tts_complete_end = Time.now
            puts "🔊 [DEBUG] TTS finalized in #{((@debug_tts_complete_end - @debug_tts_complete_start) * 1000).round(0)}ms"
          end
          
          puts "" # New line after streaming
        end
      end
    rescue => e
      puts "\n❌ Streaming error: #{e.message}"
      # Fallback to regular response
      @audio_processor.stop_streaming_tts
      response = @ollama_client.generate_response(text)
      agent_reply = response[:message]
      final_response = response
      puts "🤖 Agent: #{agent_reply}"
      speak_response(agent_reply)
    ensure
      @speaking = false
      # Streaming service handles agent speaking state internally
      
      # Debug timing: Total pipeline summary
      if ENV['DEBUG'] && @debug_speech_received_at
        @debug_pipeline_end = Time.now
        total_pipeline_ms = ((@debug_pipeline_end - @debug_speech_received_at) * 1000).round(0)
        puts "⏱️  [DEBUG] TOTAL PIPELINE: #{total_pipeline_ms}ms (Speech→Response complete)"
        
        # Additional timing breakdown if available
        if @debug_llm_start && @debug_first_chunk_time
          stt_to_llm_ms = ((@debug_llm_start - @debug_speech_received_at) * 1000).round(0)
          llm_first_response_ms = ((@debug_first_chunk_time - @debug_llm_start) * 1000).round(0)
          puts "📊 [DEBUG] BREAKDOWN: STT→LLM: #{stt_to_llm_ms}ms, LLM first chunk: #{llm_first_response_ms}ms"
        end
        
        # TTS-specific timing breakdown - get actual playback time from audio processor
        audio_processor_playback_time = @audio_processor.instance_variable_get(:@debug_first_playback_time)
        if @debug_first_audio_time && audio_processor_playback_time
          text_to_audio_ms = ((audio_processor_playback_time - @debug_first_audio_time) * 1000).round(0)
          puts "📊 [DEBUG] TTS BREAKDOWN: Text→Audio playback: #{text_to_audio_ms}ms"
          
          # End-to-end user experience timing
          speech_to_audio_ms = ((audio_processor_playback_time - @debug_speech_received_at) * 1000).round(0)
          puts "🎯 [DEBUG] USER EXPERIENCE: Speech→Audio playing: #{speech_to_audio_ms}ms"
        else
          puts "📊 [DEBUG] TTS BREAKDOWN: First audio playback timing not captured"
        end
      end
    end
    
    # Use the response from streaming or fallback
    response = final_response || { message: agent_reply, coverage_confirmed: nil, end_conversation: false }
    
    # Debug output
    puts "DEBUG: Raw agent reply: '#{response[:message]}'" if ENV['DEBUG']
    puts "DEBUG: Coverage confirmed: #{response[:coverage_confirmed]}" if ENV['DEBUG']
    puts "DEBUG: End conversation: #{response[:end_conversation]}" if ENV['DEBUG']
    
    # Check if LLM wants to end conversation or has confirmed coverage
    if false # response[:end_conversation] || response[:coverage_confirmed] == true
      puts "\n✅ Call objectives completed based on LLM assessment!"
      confirmation = "Perfect, Lanes&Planes will be billed directly. Thank you!"
      puts "🤖 Agent: #{confirmation}"
      speak_response(confirmation)
      stop_call
      return
    end
  end

  def speak_response(text)
    return if text.nil? || text.empty?
    
    # Debug timing: Non-streaming TTS
    if ENV['DEBUG']
      @debug_tts_batch_start = Time.now
      puts "🔊 [DEBUG] Starting batch TTS at #{@debug_tts_batch_start.strftime('%H:%M:%S.%3N')}"
    end
    
    @speaking = true
    
    # Streaming service will pause transcription during TTS automatically
    
    begin
      @audio_processor.text_to_speech(text)
      
      if ENV['DEBUG']
        @debug_tts_batch_end = Time.now
        puts "🔊 [DEBUG] Batch TTS completed in #{((@debug_tts_batch_end - @debug_tts_batch_start) * 1000).round(0)}ms"
      end
      
    rescue => e
      puts "TTS error: #{e.message}"
    ensure
      @speaking = false
      # Streaming service will resume transcription automatically
    end
  end

  def speak_response_streaming(text)
    """
    Stream text to speech as it's generated by LLM
    """
    return if text.nil? || text.empty?
    
    @speaking = true
    
    # Streaming service will pause transcription during TTS automatically
    
    begin
      # Start streaming TTS
      @audio_processor.start_streaming_tts
      
      # Send the complete text for streaming processing
      @audio_processor.stream_text_chunk(text, true)
      
      # Wait for streaming to complete
      @audio_processor.stop_streaming_tts
      
    rescue => e
      puts "Streaming TTS error: #{e.message}"
    ensure
      @speaking = false
      # Streaming service will resume transcription automatically
    end
  end

  def cleanup_response(text)
    return "" if text.nil?
    
    # Remove any remaining artifacts and keep it very short
    clean = text.strip
    clean = clean.split('.')[0] + '.' if clean.include?('.')  # Take only first sentence
    clean = clean.split('?')[0] + '?' if clean.include?('?')  # Take only first question
    
    # Limit to very short responses
    words = clean.split
    if words.length > 10
      clean = words[0..9].join(' ') + '.'
    end
    
    clean.strip
  end

  def self.call_objectives_met?(ollama_client)
    return false # do a real check here

    conversation = ollama_client.conversation_history
    return false if conversation.length < 6  # Need at least 3 real exchanges
    
    conversation_text = conversation.map { |msg| msg[:content] }.join(' ').downcase

    # Require EXPLICIT confirmations from hotel (much stricter)
    direct_payment_confirmed = conversation_text.match?(/(yes.*direct|accept.*direct|bill.*lanes|lanes.*bill|direct.*payment.*yes|we.*accept)/)
    no_guest_charge_confirmed = conversation_text.match?(/(no.*charge.*guest|guest.*not.*charge|won.*charge.*guest|no.*guest.*charge)/)
    
    # Need BOTH confirmations
    both_confirmed = direct_payment_confirmed && no_guest_charge_confirmed
    sufficient_exchange = conversation.length >= 6
    
    puts "Debug: Direct payment confirmed: #{!!direct_payment_confirmed}" if ENV['DEBUG']
    puts "Debug: No guest charge confirmed: #{!!no_guest_charge_confirmed}" if ENV['DEBUG']
    puts "Debug: Conversation length: #{conversation.length}" if ENV['DEBUG']
    
    both_confirmed && sufficient_exchange
  end

  def call_objectives_met?
    self.class.call_objectives_met?(@ollama_client)
  end

  def agent_speaking?
    @speaking
  end

  def setup_signal_handlers
    trap('INT') do
      puts "\n📞 Stopping call..."
      stop_call
    end
    
    trap('TERM') do
      puts "\n📞 Stopping call..."
      stop_call
    end
  end
end
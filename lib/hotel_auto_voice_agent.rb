require_relative 'audio_processor'
require_relative 'hotel_booking_agent'
require_relative 'continuous_audio_monitor'

class HotelAutoVoiceAgent
  def initialize
    @audio_processor = AudioProcessor.new
    @ollama_client = HotelBookingOllamaClient.new
    @speaking = false
    @call_active = false
    
    # Start Kokoro TTS server for fast speech generation
    puts "🚀 Initializing TTS engine..."
    @audio_processor.start_kokoro_server
    
    # Create callbacks for continuous audio monitor
    speech_callback = method(:handle_hotel_speech)
    speaking_callback = method(:agent_speaking?)
    @audio_monitor = ContinuousAudioMonitor.new(speech_callback, speaking_callback)
    
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
    
    puts "🎤 Starting continuous listening for hotel staff..."
    
    # Start continuous audio monitoring
    unless @audio_monitor.start_listening
      puts "❌ Failed to start continuous audio monitoring"
      return false
    end
    
    puts "🎤 System is now actively listening for hotel staff speech..."
    puts "💬 The system will automatically respond when hotel staff speaks"
    puts "⏹️  Press Ctrl+C to stop the call at any time"
    puts ""
    
    # Keep call active
    begin
      loop_count = 0
      while @call_active && @audio_monitor.listening?
        sleep(0.5)
        loop_count += 1
        
        # Show activity indicator every 10 seconds
        if loop_count % 20 == 0  # Every 10 seconds (20 * 0.5s)
          status = @audio_monitor.get_status
          if status
            puts "🎤 Listening... (#{status['total_chunks']} audio chunks processed, #{status['processed_utterances']} speech events detected)"
          else
            puts "🎤 Listening... (waiting for hotel staff to speak)"
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
    @audio_monitor.stop_listening
    @audio_processor.stop_kokoro_server
  end

  private

  def handle_hotel_speech(text)
    return if @speaking || !@call_active || text.nil? || text.strip.empty?
    
    puts "🏨 Hotel: #{text}"
    
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

    # Generate agent response
    response = @ollama_client.generate_response(text)
    agent_reply = response[:message]
    
    # Debug output
    puts "DEBUG: Raw agent reply: '#{agent_reply}'" if ENV['DEBUG']
    puts "DEBUG: Coverage confirmed: #{response[:coverage_confirmed]}" if ENV['DEBUG']
    puts "DEBUG: End conversation: #{response[:end_conversation]}" if ENV['DEBUG']
    
    if agent_reply && !agent_reply.empty? && agent_reply != "I apologize, I need to think about that a bit more."
      puts "🤖 Agent: #{agent_reply}"
      speak_response(agent_reply)
      
      # Check if LLM wants to end conversation or has confirmed coverage
      if response[:end_conversation] || response[:coverage_confirmed] == true
        puts "\n✅ Call objectives completed based on LLM assessment!"
        confirmation = "Perfect, Lanes&Planes will be billed directly. Thank you!"
        puts "🤖 Agent: #{confirmation}"
        speak_response(confirmation)
        stop_call
        return
      end
    else
      # Better fallback based on context
      if text.downcase.include?('hello') || text.downcase.include?('there')
        fallback = "Hello, this is Alex from Lanes&Planes. I need to confirm payment arrangements for your booking."
      elsif text.downcase.include?('want') || text.downcase.include?('about')
        fallback = "I need to confirm that you'll accept direct payment from Lanes&Planes for this booking."
      else
        fallback = "Will you accept direct payment from Lanes&Planes for this booking?"
      end
      
      puts "🤖 Agent: #{fallback}"
      speak_response(fallback)
    end
  end

  def speak_response(text)
    return if text.nil? || text.empty?
    
    @speaking = true
    
    # Notify audio monitor that agent is speaking
    @audio_monitor.set_agent_speaking(true)
    
    begin
      @audio_processor.text_to_speech(text)
    rescue => e
      puts "TTS error: #{e.message}"
    ensure
      @speaking = false
      # Notify audio monitor that agent finished speaking
      @audio_monitor.set_agent_speaking(false)
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
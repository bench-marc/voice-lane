require_relative 'audio_processor'
require_relative 'hotel_booking_agent'

class HotelVoiceAgent
  def initialize
    @audio_processor = AudioProcessor.new
    @ollama_client = HotelBookingOllamaClient.new
    @speaking = false
  end

  def start_voice_call(guest_name: nil, booking_reference: nil, hotel_name: nil)
    puts "\n" + "="*60
    puts "📞 LANES&PLANES VOICE HOTEL BOOKING AGENT"
    puts "="*60
    puts "🎤 Voice-enabled hotel cost coverage confirmation"
    puts "👤 Guest: #{guest_name}" if guest_name
    puts "📋 Booking: #{booking_reference}" if booking_reference
    puts "🏨 Hotel: #{hotel_name}" if hotel_name
    puts "="*60
    puts "Press Enter when hotel staff answers, or 'quit' to exit"
    puts "="*60 + "\n"

    # Store booking details for context
    @ollama_client.set_booking_details(
      guest_name: guest_name,
      booking_reference: booking_reference,
      hotel_name: hotel_name
    )

    # Calibrate microphone for better voice detection
    @audio_processor.calibrate_microphone
    
    # Wait for user to indicate call is connected
    print "Press Enter when hotel answers the phone: "
    STDIN.gets

    # Generate and speak opening statement
    opening = @ollama_client.generate_opening_statement
    puts "🤖 Agent: #{opening}"
    speak_response(opening)
    
    # Add opening statement to conversation history
    @ollama_client.add_assistant_message(opening)

    # Start voice conversation loop
    voice_conversation_loop
  end

  private

  def voice_conversation_loop
    loop do
      puts "\n🎤 Ready to listen for hotel response... (Press Enter to start)"
      STDIN.gets

      # Record hotel response using smart recording
      puts "🎤 Recording hotel response (will stop when you finish talking)..."
      audio_file = @audio_processor.record_until_silence
      
      if audio_file
        puts "🔄 Processing speech..."
        hotel_response = @audio_processor.speech_to_text(audio_file)
        File.delete(audio_file) if File.exist?(audio_file)
        
        if hotel_response && hotel_response.length > 2
          puts "🏨 Hotel: #{hotel_response}"
          
          # Check for call end
          if hotel_response.downcase.match?(/(goodbye|thank you|have a|good day|bye)/)
            puts "📞 Hotel seems to be ending the call..."
            
            # Generate closing response
            closing = @ollama_client.generate_response("Thank you for confirming the cost coverage. Have a great day!")
            puts "🤖 Agent: #{closing}"
            speak_response(closing)
            break
          end
          
          # Generate agent response
          response = @ollama_client.generate_response(hotel_response)
          agent_reply = response[:message]
          puts "🤖 Agent: #{agent_reply}"
          speak_response(agent_reply)
          
          # Check if objectives are met or if LLM wants to end conversation
          if response[:end_conversation] || response[:coverage_confirmed] == true
            puts "\n✅ Call objectives completed successfully!"
            
            # Generate confirmation statement
            confirmation = "Perfect, I have confirmation that Lanes&Planes will be billed directly and the guest will not be charged. Thank you for your time!"
            puts "🤖 Agent: #{confirmation}"
            speak_response(confirmation)
            
            summary = @ollama_client.generate_call_summary
            puts "\n📋 Call Summary:"
            puts summary
            break
          end
        else
          puts "❌ No clear speech detected. Please try again."
        end
      else
        puts "❌ Recording failed. Please try again."
      end
    end
    
    puts "\n📞 Call completed successfully!"
  end

  def speak_response(text)
    return if text.nil? || text.empty?
    
    @speaking = true
    @audio_processor.text_to_speech(text)
    @speaking = false
  end

end
require_relative 'audio_processor'
require_relative 'ollama_client'

class HotelBookingAgent
  def initialize
    @audio_processor = AudioProcessor.new
    # Use specialized Ollama client for hotel booking context
    @ollama_client = HotelBookingOllamaClient.new
  end

  def start_call(guest_name: nil, booking_reference: nil, hotel_name: nil)
    puts "\n" + "="*60
    puts "📞 LANES&PLANES HOTEL BOOKING AGENT"
    puts "="*60
    puts "🏨 Calling hotel to confirm cost coverage"
    puts "👤 Guest: #{guest_name}" if guest_name
    puts "📋 Booking: #{booking_reference}" if booking_reference
    puts "🏨 Hotel: #{hotel_name}" if hotel_name
    puts "="*60 + "\n"

    # Store booking details for context
    @ollama_client.set_booking_details(
      guest_name: guest_name,
      booking_reference: booking_reference,
      hotel_name: hotel_name
    )

    # Generate opening statement
    opening = @ollama_client.generate_opening_statement
    puts "🤖 Agent: #{opening}"
    @audio_processor.text_to_speech(opening)

    # Start conversation loop
    conversation_loop
  end

  private

  def conversation_loop
    loop do
      print "\n📞 Hotel Response (or 'end call' to finish): "
      hotel_response = STDIN.gets.chomp

      break if hotel_response.downcase.include?('end call') || hotel_response.downcase.include?('goodbye')

      # Process hotel response and generate agent reply
      agent_reply = @ollama_client.generate_response(hotel_response)
      puts "🤖 Agent: #{agent_reply}"
      @audio_processor.text_to_speech(agent_reply)

      # Check if objectives are met
      if HotelAutoVoiceAgent.call_objectives_met?(@ollama_client)
        puts "\n✅ Call objectives completed successfully!"
        summary = @ollama_client.generate_call_summary
        puts "\n📋 Call Summary:"
        puts summary
        break
      end
    end

    puts "\n📞 Call ended. Thank you!"
  end

end

class HotelBookingOllamaClient < OllamaClient
  def initialize
    super('hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q2_K')  # Use more capable model for better instruction following
    @booking_details = {}
  end

  def set_booking_details(guest_name: nil, booking_reference: nil, hotel_name: nil)
    @booking_details = {
      guest_name: guest_name,
      booking_reference: booking_reference,
      hotel_name: hotel_name
    }
  end

  def generate_opening_statement
    details = format_booking_details
    context = build_opening_context(details)
    
    response = self.class.post('/api/generate', {
      body: {
        model: @model,
        prompt: context,
        stream: false,
        options: {
          temperature: 0.3, # Lower temperature for more consistent professional responses
          num_predict: 100  # Increased to prevent truncation
        }
      }.to_json,
      headers: { 'Content-Type' => 'application/json' },
      timeout: 30
    })

    if response.success?
      result = JSON.parse(response.body)
      clean_ai_response(result['response'].strip)
    else
      "Hello, this is Alex from Lanes&Planes. I need to confirm cost coverage arrangements for booking #{@booking_details[:booking_reference]}."
    end
  end

  def generate_call_summary
    conversation_text = @conversation_history.map do |msg|
      "#{msg[:role] == 'user' ? 'Hotel' : 'Agent'}: #{msg[:content]}"
    end.join("\n")

    context = "Summarize this hotel booking confirmation call. Focus on whether cost coverage was confirmed:\n\n#{conversation_text}\n\nSummary:"

    response = self.class.post('/api/generate', {
      body: {
        model: @model,
        prompt: context,
        stream: false,
        options: { temperature: 0.2, num_predict: 100 }
      }.to_json,
      headers: { 'Content-Type' => 'application/json' }
    })

    if response.success?
      result = JSON.parse(response.body)
      clean_ai_response(result['response'].strip)
    else
      "Call completed - please review conversation for cost coverage confirmation."
    end
  end

  private

  def build_context
    details = format_booking_details
    
    context = build_hotel_context(details)

    context += "History of conversation: \nCONVERSATION START\n"
    # Use recent conversation history
    recent_history = @conversation_history.last(6)
    recent_history.each do |msg|
      role = msg[:role] == 'user' ? 'Hotel Staff' : 'You'
      context += "#{role}: #{msg[:content]}\n"
    end
    context += "CONVERSATION END\n"
    context += "Please react only to the last message from Hotel staff now!"
    context
  end

  def build_opening_context(details)
    "You are Alex from Lanes&Planes making your opening call to a hotel.#{details}

YOUR TASK: Generate Alex's professional opening statement to introduce the call.

REQUIREMENTS:
- Identify yourself as Alex from Lanes&Planes
- State you're calling about a guest booking  
- Mention you need to confirm payment arrangements
- Be brief and professional (maximum 2 sentences)
- Don't ask detailed questions yet, just introduce the purpose

Alex's opening statement:"
  end

  def build_hotel_context(details)
    "You ARE Alex from Lanes&Planes currently in conversation with hotel staff.#{details}

YOUR OBJECTIVES:
- Confirm hotel will accept DIRECT PAYMENT from Lanes&Planes 
- Ensure guest will NOT be charged directly at the hotel
- Get clear YES/NO answers to these questions

CONVERSATION STYLE:
- Professional and focused
- One question at a time
- Maximum 2 sentences per response  
- React directly to what hotel staff just said
- Stay persistent until you get clear confirmation

"
  end

  def format_booking_details
    return "" if @booking_details.empty?
    
    details = "\n\nBOOKING DETAILS:"
    details += "\n- Guest: #{@booking_details[:guest_name]}" if @booking_details[:guest_name]
    details += "\n- Reference: #{@booking_details[:booking_reference]}" if @booking_details[:booking_reference]  
    details += "\n- Hotel: #{@booking_details[:hotel_name]}" if @booking_details[:hotel_name]
    details
  end
end
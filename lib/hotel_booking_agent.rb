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
    puts "🏨 Calling hotel to confirm cost coverage (Hotel speaks first)"
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

    # Start conversation loop (hotel speaks first)
    conversation_loop
  end

  private

  def conversation_loop
    loop do
      print "\n📞 Hotel Response (or 'end call' to finish): "
      hotel_response = STDIN.gets.chomp

      break if hotel_response.downcase.include?('end call') || hotel_response.downcase.include?('goodbye')

      # # If this is the first message (hotel greeting), generate opening response
      # if @ollama_client.conversation_length == 0
      #   opening = @ollama_client.generate_opening_statement
      #   puts "🤖 Agent: #{opening}"
      #   @audio_processor.text_to_speech(opening)
      #
      #   # Add opening statement to conversation history
      #   @ollama_client.add_assistant_message(opening)
      #   next
      # end

      # Process hotel response and generate agent reply
      response = @ollama_client.generate_response(hotel_response)
      agent_reply = response[:message]
      puts "🤖 Agent: #{agent_reply}"
      @audio_processor.text_to_speech(agent_reply)

      # Check if objectives are met or if LLM wants to end conversation
      if response[:end_conversation] || response[:coverage_confirmed] == true
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
    super('hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q4_K_M')  # Use more capable model for better instruction following
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
        format: {
          type: "object",
          properties: {
            message: {
              type: "string",
              description: "Your professional opening statement (maximum 2 sentences)"
            },
            coverage_confirmed: {
              type: ["boolean", "null"],
              description: "Always null for opening statement"
            },
            end_conversation: {
              type: "boolean",
              description: "Always false for opening statement"
            }
          },
          required: ["message", "coverage_confirmed", "end_conversation"]
        },
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
      ai_response = result['response'].strip
      
      begin
        # Parse JSON response for opening statement
        json_response = JSON.parse(ai_response)
        json_response['message'] || "Hello, this is Alex from Lanes&Planes. I need to confirm cost coverage arrangements for booking #{@booking_details[:booking_reference]}."
      rescue JSON::ParserError
        # Fallback if not JSON
        clean_ai_response(ai_response)
      end
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

    context += "CONVERSATION HISTORY\n"
    # Use recent conversation history
    recent_history = @conversation_history.last(6)
    recent_history.each do |msg|
      role = msg[:role] == 'user' ? 'Hotel Staff' : 'You'
      context += "#{role}: \"#{msg[:content]}\"\n"
    end
    context += "\n\n"
    context += "You: "
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
    "You ARE Chasey from Lanes&Planes currently in a phone call with hotel staff.#{details}

BACKGROUND:
- Lanes&Planes is a travel agency for business travel
- You are an employee of Lanes&Planes
- The stay is already booked and confirmed
- Lanes&Planes wants to cover all charges for the booking
- Lanes&Planes wants to prevent that the hotel charges the traveler directly
- Lanes&Planes does not cover charges for the minibar
- The answers from hotel staff might have spelling errors or word may have been misunderstood. Guess what hotel staff wanted to say if their answers dont make sense.

YOUR OBJECTIVES:
- Confirm hotel will accept DIRECT PAYMENT from Lanes&Planes 
- Ensure guest will NOT be charged directly at the hotel
- Get clear YES/NO answers to these questions

CONVERSATION STYLE:
- Professional and focused
- One question at a time
- Maximum 2 sentences per response  
- Read the conversation history for context
- Please react ONLY to the last message from Hotel staff
- Stay persistent until you get clear confirmation or rejection about the cost coverage
- After you got clear confirmation or rejection, thank the hotel employee and say goodbye. Do not repeat the outcome of the phone call again.
- keep your answers as natural as possible
- make the conversation feel like a usual phone call

"
  end

  def format_booking_details
    return "" if @booking_details.empty?
    
    details = "\n\nBOOKING DETAILS:"
    details += "\n- Guest: #{@booking_details[:guest_name]}" if @booking_details[:guest_name]
    details += "\n- Reference: #{@booking_details[:booking_reference]}" if @booking_details[:booking_reference]  
    details += "\n- Hotel: #{@booking_details[:hotel_name]}" if @booking_details[:hotel_name]
    details += "\n- Dates: #{@booking_details[:check_in]}" if @booking_details[:check_in]
    details
  end
end
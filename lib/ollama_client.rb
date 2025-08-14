require 'httparty'
require 'json'

class OllamaClient
  include HTTParty
  base_uri 'http://localhost:11434'

  def initialize(model = 'phi3:mini')
    @model = model
    @conversation_history = []
  end

  def generate_response(user_input)
    return "Sorry, I didn't catch that." if user_input.nil? || user_input.strip.empty?
    
    @conversation_history << { role: 'user', content: user_input }
    
    # Build context from conversation history
    context = build_context
    
    begin
      response = self.class.post('/api/generate', {
        body: {
          model: @model,
          prompt: context,
          stream: false,
          options: {
            temperature: 0.7,
            num_predict: 100
          }
        }.to_json,
        headers: { 'Content-Type' => 'application/json' },
        timeout: 30
      })

      if response.success?
        result = JSON.parse(response.body)
        ai_response = result['response'].strip
        
        # Clean up thinking tokens and other artifacts
        clean_response = clean_ai_response(ai_response)
        
        unless clean_response.empty?
          @conversation_history << { role: 'assistant', content: clean_response }
          
          # Keep conversation history manageable
          if @conversation_history.length > 12
            @conversation_history = @conversation_history.last(10)
          end
        end
        
        return clean_response.empty? ? "I apologize, I need to think about that a bit more." : clean_response
      else
        puts "Ollama HTTP error: #{response.code} - #{response.message}"
        return "Sorry, I'm having trouble connecting to the AI service."
      end
    rescue Net::TimeoutError
      return "Sorry, the request timed out. Please try again."
    rescue => e
      puts "Ollama error: #{e.message}"
      return "Sorry, there was an error processing your request."
    end
  end

  def clear_conversation
    @conversation_history.clear
  end

  def conversation_length
    @conversation_history.length
  end

  private

  def clean_ai_response(response)
    return "" if response.nil?
    
    # Remove thinking tokens and other artifacts
    clean = response.dup
    
    # Remove thinking blocks
    clean = clean.gsub(/<think>.*?<\/think>/m, '')
    clean = clean.gsub(/<think>.*$/m, '')
    
    # Remove any other XML-like tags that might appear
    clean = clean.gsub(/<[^>]*>/, '')
    
    # Clean up extra whitespace
    clean = clean.strip.squeeze(' ')
    
    # If the response is too short or seems incomplete, return empty
    if clean.length < 3 || clean.match?(/^\W*$/)
      return ""
    end
    
    clean
  end

  def build_context
    context = "You are a helpful AI assistant having a natural conversation over the phone. Keep responses concise, conversational, and under 2 sentences. Speak naturally as if talking to a friend.\n\n"
    
    # Use last 6 messages for context to avoid overwhelming the model
    recent_history = @conversation_history.last(6)
    recent_history.each do |msg|
      role = msg[:role] == 'user' ? 'Human' : 'Assistant'
      context += "#{role}: #{msg[:content]}\n"
    end
    
    context += "Assistant: "
    context
  end
end
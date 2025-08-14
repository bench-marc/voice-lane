require 'httparty'
require 'json'

class OllamaClient
  include HTTParty
  base_uri 'http://localhost:11434'

  def initialize(model = 'hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q2_K')
    @model = model
    @conversation_history = []
  end

  def generate_response(user_input)
    return "Sorry, I didn't catch that." if user_input.nil? || user_input.strip.empty?
    
    @conversation_history << { role: 'user', content: user_input }
    
    # Build context from conversation history
    context = build_context
    pp context
    begin
      response = self.class.post('/api/generate', {
        body: {
          model: @model,
          prompt: context,
          stream: false,
          options: {
            temperature: 0.7,
            num_predict: 100  # Increased to prevent truncation
          }
        }.to_json,
        headers: { 'Content-Type' => 'application/json' },
        timeout: 30
      })

      if response.success?
        result = JSON.parse(response.body)
        ai_response = result['response'].strip
        
        puts "DEBUG: Raw LLM response: '#{ai_response}'" if ENV['DEBUG']
        
        # Clean up thinking tokens and other artifacts
        clean_response = clean_ai_response(ai_response)
        
        puts "DEBUG: After cleaning: '#{clean_response}'" if ENV['DEBUG']
        puts "DEBUG: Clean response empty? #{clean_response.empty?}" if ENV['DEBUG']
        
        unless clean_response.empty?
          @conversation_history << { role: 'assistant', content: clean_response }
          
          # Keep conversation history manageable
          if @conversation_history.length > 12
            @conversation_history = @conversation_history.last(10)
          end
        end
        
        final_response = clean_response.empty? ? "I apologize, I need to think about that a bit more." : clean_response
        puts "DEBUG: Final response: '#{final_response}'" if ENV['DEBUG']
        return final_response
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

  def conversation_history
    @conversation_history
  end

  private

  def clean_ai_response(response)
    return "" if response.nil?
    
    # Remove thinking tokens and other artifacts
    clean = response.dup

    # Clean up extra whitespace
    clean = clean.strip.squeeze(' ')
    
    # If response starts with actual dialogue or action, keep it
    # Only filter if completely empty, just meta-commentary, or just punctuation
    if clean.empty? || clean.match?(/^\s*[^\w]*\s*$/) || clean.match?(/^(Okay|Let's|The user|I need|As \w+)/i)
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
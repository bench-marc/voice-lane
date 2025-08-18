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
          format: {
            type: "object",
            properties: {
              message: {
                type: "string",
                description: "Your professional response to the hotel staff (maximum 2 sentences)"
              },
              coverage_confirmed: {
                type: "boolean",
                description: "false unless hotel staff confirmed cost coverage in conversation history"
              },
              end_conversation: {
                type: "boolean",
                description: "true only if the conversation should end after this message"
              }
            },
            required: ["message", "coverage_confirmed", "end_conversation"]
          },
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
        
        begin
          # Parse the JSON response from LLM
          json_response = JSON.parse(ai_response)
          message = json_response['message']
          coverage_confirmed = json_response['coverage_confirmed']
          end_conversation = json_response['end_conversation']
          
          puts "DEBUG: Parsed JSON - message: '#{message}', coverage: #{coverage_confirmed}, end: #{end_conversation}" if ENV['DEBUG']
          
          unless message.nil? || message.empty?
            @conversation_history << { 
              role: 'assistant', 
              content: message,
              metadata: {
                coverage_confirmed: coverage_confirmed,
                end_conversation: end_conversation
              }
            }
            
            # Keep conversation history manageable
            if @conversation_history.length > 12
              @conversation_history = @conversation_history.last(10)
            end
          end
          
          return {
            message: message || "I apologize, I need to think about that a bit more.",
            coverage_confirmed: coverage_confirmed,
            end_conversation: end_conversation || false
          }
        rescue JSON::ParserError => e
          puts "DEBUG: JSON parse error: #{e.message}" if ENV['DEBUG']
          # Fallback to old behavior if JSON parsing fails
          clean_response = clean_ai_response(ai_response)
          @conversation_history << { role: 'assistant', content: clean_response } unless clean_response.empty?
          return {
            message: clean_response.empty? ? "I apologize, I need to think about that a bit more." : clean_response,
            coverage_confirmed: nil,
            end_conversation: false
          }
        end
      else
        puts "Ollama HTTP error: #{response.code} - #{response.message}"
        return {
          message: "Sorry, I'm having trouble connecting to the AI service.",
          coverage_confirmed: nil,
          end_conversation: false
        }
      end
    rescue Net::TimeoutError
      return {
        message: "Sorry, the request timed out. Please try again.",
        coverage_confirmed: nil,
        end_conversation: false
      }
    rescue => e
      puts "Ollama error: #{e.message}"
      return {
        message: "Sorry, there was an error processing your request.",
        coverage_confirmed: nil,
        end_conversation: false
      }
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

  def add_assistant_message(message)
    @conversation_history << { role: 'assistant', content: message }
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
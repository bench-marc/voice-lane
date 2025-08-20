require 'httparty'
require 'net/http'
require 'uri'
require 'json'

class OllamaClient
  include HTTParty
  base_uri 'http://localhost:11434'

  def initialize(model = 'hf.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q4_K_M')
    @model = model
    @conversation_history = []
  end

  def generate_response_stream(user_input, &block)
    return unless block_given?
    return if user_input.nil? || user_input.strip.empty?
    
    @conversation_history << { role: 'user', content: user_input }
    
    # Build context from conversation history
    context = build_context
    
    begin
      # Use Net::HTTP for streaming support
      uri = URI("http://localhost:11434/api/generate")  # Ollama default
      http = Net::HTTP.new(uri.host, uri.port)
      
      request = Net::HTTP::Post.new(uri.path)
      request['Content-Type'] = 'application/json'
      request.body = {
        model: @model,
        prompt: context,
        stream: true,  # Enable streaming
        # No JSON format for streaming - use plain text for clean TTS
        options: {
          temperature: 0.7,
          num_predict: 100
        }
      }.to_json
      
      full_response = ""
      final_result = {}
      
      # Stream the response
      http.request(request) do |response|
        if response.code == '200'
          response.read_body do |chunk|
            chunk.each_line do |line|
              next if line.strip.empty?
              
              begin
                chunk_data = JSON.parse(line)
                
                if chunk_data['response']
                  full_response += chunk_data['response']
                  
                  # Try to yield partial text for streaming TTS
                  block.call(chunk_data['response'], false) # false = not done
                end
                
                # Check if this is the final chunk
                if chunk_data['done']
                  # Process the complete plain text response
                  clean_response = clean_ai_response(full_response)
                  
                  # Analyze text for metadata using keyword detection
                  metadata = analyze_response_intent(clean_response)
                  
                  final_result = {
                    message: clean_response,
                    coverage_confirmed: metadata[:coverage_confirmed],
                    end_conversation: metadata[:end_conversation]
                  }
                  
                  # Add to conversation history
                  @conversation_history << { 
                    role: 'assistant', 
                    content: clean_response,
                    metadata: {
                      coverage_confirmed: metadata[:coverage_confirmed],
                      end_conversation: metadata[:end_conversation]
                    }
                  }
                  
                  # Keep conversation history manageable
                  if @conversation_history.length > 12
                    @conversation_history = @conversation_history.last(10)
                  end
                  
                  # Signal completion
                  block.call("", true) # true = done
                  break
                end
                
              rescue JSON::ParserError => e
                puts "Streaming parse error: #{e.message}" if ENV['DEBUG']
                next
              end
            end
          end
        else
          # Fallback to non-streaming on error
          puts "Streaming failed, falling back to regular response"
          return generate_response(user_input)
        end
      end
      
      return final_result
      
    rescue => e
      puts "Streaming error: #{e.message}"
      # Fallback to regular response
      return generate_response(user_input)
    end
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

  def analyze_response_intent(text)
    """
    Analyze response text to extract metadata using keyword detection
    """
    return { coverage_confirmed: nil, end_conversation: false } if text.nil? || text.empty?
    
    text_lower = text.downcase
    
    # Detect coverage confirmation
    coverage_confirmed = text_lower.match?(
      /(confirmed?|accept.*payment|bill.*lanes|direct.*payment|yes.*direct|
        lanes.*will.*bill|we.*accept|bill.*directly|charge.*lanes)/ix
    )
    
    # Detect conversation ending
    end_conversation = text_lower.match?(
      /(goodbye|thank you.*day|have.*good|bye|complete|finished|
        that.*all|anything.*else|help.*today)/ix
    )
    
    {
      coverage_confirmed: coverage_confirmed,
      end_conversation: end_conversation
    }
  end

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
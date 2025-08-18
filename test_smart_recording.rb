#!/usr/bin/env ruby

require_relative 'lib/audio_processor'

puts "🧪 Testing Smart Audio Recording"
puts "================================"

audio_processor = AudioProcessor.new

puts "\n1. Calibrating microphone..."
audio_processor.calibrate_microphone

puts "\n2. Testing smart recording..."
puts "   Say something and the recording will automatically stop when you finish talking."

audio_file = audio_processor.record_until_silence

if audio_file
  puts "\n3. Testing speech-to-text..."
  text = audio_processor.speech_to_text(audio_file)
  
  puts "\n📝 Transcribed text: '#{text}'"
  
  puts "\n4. Testing text-to-speech..."
  audio_processor.text_to_speech("I heard you say: #{text}")
  
  # Cleanup
  File.delete(audio_file) if File.exist?(audio_file)
  
  puts "\n✅ Smart recording test complete!"
else
  puts "\n❌ Smart recording failed!"
end
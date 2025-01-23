import os
import pyttsx3
import speech_recognition as sr
import datetime
import smtplib
from email.message import EmailMessage
import json
import logging
import threading
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalLanguageModel:
    def __init__(self, model_name="tiiuae/falcon-7b-instruct"):
        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Move model to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            
            # Create generation pipeline
            self.generator = pipeline(
                "text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            raise

    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        try:
            # Generate response with context
            full_prompt = f"User: {prompt}\nAssistant:"
            responses = self.generator(
                full_prompt, 
                max_length=max_length, 
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
            
            # Extract and clean response
            response = responses[0]['generated_text'].split("Assistant:")[-1].strip()
            return response
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I'm having trouble understanding that request."

class VoiceEngine:
    def __init__(self, voice_id: int = 0):
        self.engine = pyttsx3.init()
        self.setup_voice(voice_id)
        
    def setup_voice(self, voice_id: int):
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[voice_id].id)
        self.engine.setProperty('rate', 175)
        
    def speak(self, text: str):
        logger.info(f"Speaking: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        
    def listen(self, timeout=5, phrase_time_limit=5):
        try:
            with sr.Microphone() as source:
                print("Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                
            text = self.recognizer.recognize_google(audio)
            return text.lower()
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return None

class VoiceAssistant:
    def __init__(self):
        # Initialize components
        self.voice_engine = VoiceEngine()
        self.speech_recognizer = SpeechRecognizer()
        self.llm = LocalLanguageModel()
        self.running = True
        
    def process_generic_query(self, query):
        """Use LLM for generic queries not handled by specific methods"""
        try:
            response = self.llm.generate_response(query)
            self.voice_engine.speak(response)
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            self.voice_engine.speak("I encountered an error processing your request.")
    
    def process_time_command(self):
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        self.voice_engine.speak(f"The current time is {current_time}")
        
    def process_date_command(self):
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        self.voice_engine.speak(f"Today is {current_date}")
        
    def greet(self):
        hour = datetime.datetime.now().hour
        greeting = (
            "Good morning!" if 6 <= hour < 12 else
            "Good afternoon!" if 12 <= hour < 18 else
            "Good evening!"
        )
        self.voice_engine.speak(f"{greeting} I'm your AI assistant. How can I help you?")
        
    def run(self):
        self.greet()
        
        while self.running:
            try:
                command = self.speech_recognizer.listen()
                
                if not command:
                    continue
                
                print(f"You said: {command}")
                
                # Predefined commands
                if 'time' in command:
                    self.process_time_command()
                elif 'date' in command:
                    self.process_date_command()
                elif any(exit_word in command for exit_word in ['exit', 'quit', 'stop', 'goodbye']):
                    self.voice_engine.speak("Goodbye!")
                    self.running = False
                else:
                    # Use LLM for complex or undefined queries
                    self.process_generic_query(command)
                    
            except Exception as e:
                logger.error(f"Assistant runtime error: {e}")
                self.voice_engine.speak("I encountered an unexpected error.")

def main():
    try:
        assistant = VoiceAssistant()
        assistant.run()
    except KeyboardInterrupt:
        print("\nAssistant stopped.")
    except Exception as e:
        logger.error(f"Critical error: {e}")

if __name__ == "__main__":
    main()

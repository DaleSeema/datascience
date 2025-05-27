import pyttsx3
from typing import List, Dict, Any # For potential future use with voice properties

class TextToSpeech:
    def __init__(self, rate: int = 150, volume: float = 1.0, voice_id: str = None):
        """
        Initializes the Text-to-Speech engine using pyttsx3.

        Args:
            rate (int): Speaking rate (words per minute). Default is 150.
            volume (float): Speaking volume (0.0 to 1.0). Default is 1.0.
            voice_id (str, optional): Specific voice ID to use. If None, system default is used.
        """
        try:
            self.engine = pyttsx3.init()
        except Exception as e:
            print(f"Error initializing pyttsx3 engine: {e}")
            print("Please ensure that you have a text-to-speech engine installed on your system.")
            print("On Linux, this might require installing 'espeak' or 'festival':")
            print("  sudo apt-get update && sudo apt-get install espeak festival")
            print("On Windows or macOS, TTS engines are usually pre-installed.")
            self.engine = None # Set engine to None if initialization fails
            return # Exit constructor if engine fails to init

        if self.engine:
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            
            if voice_id:
                # Ensure voice_id is valid before setting, or pyttsx3 might error or ignore
                available_voices = self.engine.getProperty('voices')
                is_valid_voice = any(v.id == voice_id for v in available_voices)
                if is_valid_voice:
                    self.engine.setProperty('voice', voice_id)
                else:
                    print(f"Warning: Voice ID '{voice_id}' not found. Using default voice.")
            
            print(f"TextToSpeech engine initialized. Rate: {self.engine.getProperty('rate')}, Volume: {self.engine.getProperty('volume')}, Voice: {self.engine.getProperty('voice')}")
        else:
            print("TextToSpeech engine could not be initialized.")

    def speak(self, text: str) -> bool:
        """
        Speaks the given text using the initialized TTS engine.

        Args:
            text (str): The text to be spoken.

        Returns:
            bool: True if speech was attempted (engine exists and `say` was called), 
                  False otherwise (e.g., engine not initialized).
        """
        if not self.engine:
            print("Error: TTS engine not initialized. Cannot speak.")
            return False
        
        if not text:
            print("Warning: No text provided to speak.")
            return False # Or True, depending on desired behavior for empty text

        try:
            self.engine.say(text)
            self.engine.runAndWait() # Blocks until speaking is complete
            return True
        except RuntimeError as e:
            # This can happen if the event loop is busy or pyttsx3 has issues.
            print(f"RuntimeError during tts.speak(): {e}")
            print("This might be due to issues with the TTS engine or event loop.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred during tts.speak(): {e}")
            return False

    def set_rate(self, rate: int) -> bool:
        """Sets the speaking rate (words per minute)."""
        if not self.engine: 
            print("Error: TTS engine not initialized. Cannot set rate.")
            return False
        try:
            self.engine.setProperty('rate', rate)
            # print(f"TTS rate set to: {rate}") # Optional debug
            return True
        except Exception as e:
            print(f"Error setting rate: {e}")
            return False

    def get_rate(self) -> Optional[int]:
        """Gets the current speaking rate."""
        if not self.engine: 
            print("Error: TTS engine not initialized. Cannot get rate.")
            return None
        try:
            return self.engine.getProperty('rate')
        except Exception as e:
            print(f"Error getting rate: {e}")
            return None

    def set_volume(self, volume: float) -> bool:
        """Sets the speaking volume (0.0 to 1.0)."""
        if not self.engine: 
            print("Error: TTS engine not initialized. Cannot set volume.")
            return False
        if not (0.0 <= volume <= 1.0):
            print("Error: Volume must be between 0.0 and 1.0")
            return False
        try:
            self.engine.setProperty('volume', volume)
            # print(f"TTS volume set to: {volume}") # Optional debug
            return True
        except Exception as e:
            print(f"Error setting volume: {e}")
            return False

    def get_volume(self) -> Optional[float]:
        """Gets the current speaking volume."""
        if not self.engine: 
            print("Error: TTS engine not initialized. Cannot get volume.")
            return None
        try:
            return self.engine.getProperty('volume')
        except Exception as e:
            print(f"Error getting volume: {e}")
            return None

    def list_voices(self) -> List[Dict[str, Any]]:
        """Lists available voices and their attributes."""
        voices_info = []
        if not self.engine: 
            print("Error: TTS engine not initialized. Cannot list voices.")
            return voices_info
        
        try:
            available_voices = self.engine.getProperty('voices')
            for voice in available_voices:
                voices_info.append({
                    'id': voice.id,
                    'name': voice.name,
                    # These attributes might be empty or not well-populated by all engines
                    'languages': getattr(voice, 'languages', []), 
                    'gender': getattr(voice, 'gender', None),    
                    'age': getattr(voice, 'age', None)           
                })
            return voices_info
        except Exception as e:
            print(f"Error listing voices: {e}")
            return voices_info # Return empty list on error

    def set_voice(self, voice_id: str) -> bool:
        """Sets the voice to be used by its ID."""
        if not self.engine: 
            print("Error: TTS engine not initialized. Cannot set voice.")
            return False
        try:
            available_voices = self.engine.getProperty('voices')
            if any(v.id == voice_id for v in available_voices):
                self.engine.setProperty('voice', voice_id)
                # print(f"TTS voice set to ID: {voice_id}") # Optional debug
                return True
            else:
                print(f"Error: Voice ID '{voice_id}' not found.")
                return False
        except Exception as e:
            print(f"Error setting voice: {e}")
            return False
            
    def get_current_voice_id(self) -> Optional[str]:
        """Gets the ID of the current voice being used."""
        if not self.engine: 
            print("Error: TTS engine not initialized. Cannot get current voice ID.")
            return None
        try:
            return self.engine.getProperty('voice')
        except Exception as e:
            print(f"Error getting current voice ID: {e}")
            return None


if __name__ == '__main__':
    print("Comprehensive Test for TextToSpeech Module")
    print("="*40)

    try:
        print("\nInitializing TTS engine with default settings...")
        tts = TextToSpeech() # Default rate 150
        
        if not tts.engine:
            print("\nTTS engine initialization failed. Aborting further tests.")
            # Exit if engine couldn't be initialized, as other calls will fail
            import sys
            sys.exit(1)

        print("\n--- Testing speak() method ---")
        tts.speak("Hello, this is a test of the Text to Speech engine.")
        
        print("\n--- Testing rate property ---")
        initial_rate = tts.get_rate()
        print(f"Initial rate: {initial_rate}")
        print("Setting rate to 200...")
        tts.set_rate(200)
        tts.speak("My speaking rate is now 200 words per minute.")
        print(f"Rate after set: {tts.get_rate()}")
        print("Setting rate back to 150...")
        tts.set_rate(150) # Reset for further tests if needed
        tts.speak("My speaking rate is back to 150.")


        print("\n--- Testing volume property ---")
        initial_volume = tts.get_volume()
        print(f"Initial volume: {initial_volume}")
        if initial_volume is not None and initial_volume > 0.5:
            print("Setting volume to 0.5...")
            tts.set_volume(0.5)
            tts.speak("My speaking volume is now 0.5.")
            print(f"Volume after set: {tts.get_volume()}")
            tts.set_volume(initial_volume) # Reset
        else:
            print("Setting volume to 1.0 (max)...") # If initial volume was low
            tts.set_volume(1.0)
            tts.speak("My speaking volume is now 1.0.")
            print(f"Volume after set: {tts.get_volume()}")
            if initial_volume is not None: tts.set_volume(initial_volume) # Reset if possible

        print("\n--- Testing voice listing and selection ---")
        voices = tts.list_voices()
        if voices:
            print("Available voices:")
            for i, voice_info in enumerate(voices):
                print(f"  {i+1}. ID: {voice_info['id']}")
                print(f"     Name: {voice_info['name']}")
                # print(f"     Languages: {voice_info.get('languages', 'N/A')}") # May be missing
                # print(f"     Gender: {voice_info.get('gender', 'N/A')}")       # May be missing
            
            current_voice_id = tts.get_current_voice_id()
            print(f"\nCurrent voice ID: {current_voice_id}")

            if len(voices) > 1:
                # Attempt to set to a different voice (e.g., the second voice if available)
                # This is highly dependent on the system's installed voices.
                new_voice_id_to_test = voices[1]['id'] # Try the second voice
                if new_voice_id_to_test != current_voice_id:
                    print(f"Attempting to set voice to ID: {new_voice_id_to_test} ({voices[1]['name']})")
                    success_set_voice = tts.set_voice(new_voice_id_to_test)
                    if success_set_voice:
                        print(f"Successfully set voice. New current voice ID: {tts.get_current_voice_id()}")
                        tts.speak(f"I am now speaking with the voice of {voices[1]['name']}.")
                        # Set back to original voice if it was different
                        if current_voice_id:
                             print(f"Setting voice back to original ID: {current_voice_id}")
                             tts.set_voice(current_voice_id)
                    else:
                        print(f"Failed to set voice to ID: {new_voice_id_to_test}")
                else:
                    print("Only one distinct voice available, or new voice is same as current. Skipping voice change test.")
            else:
                print("Only one voice available. Skipping voice change test.")
        else:
            print("No voices found or error listing voices.")

        tts.speak("All tests complete. Goodbye!")
        print("\n--- TTS Module Test Suite Finished ---")

    except ImportError:
        print("Error: pyttsx3 library not found. Please install it using 'pip install pyttsx3'.")
    except Exception as e:
        print(f"An unexpected error occurred during the TTS test suite: {e}")
        print("This might indicate issues with your TTS engine installation or configuration.")

# Real-Time Sign Language Translator Prototype

This project aims to develop a prototype system capable of translating sign language (specifically focusing on American Sign Language - ASL) into text and speech in real-time. The system is designed to facilitate communication between sign language users and non-signers.

## Project Goals
- Capture video input in real-time.
- Extract key features from hands, body pose, and facial expressions using computer vision.
- Recognize sign language gestures and sequences using machine learning models.
- Translate recognized signs into text.
- Utilize a Large Language Model (LLM) to refine the translated text for grammatical correctness and fluency.
- Convert the final text into audible speech.

## Modules
The project is structured into several core modules located in the `src/` directory:
- `main.py`: Main application entry point.
- `video_capture.py`: Handles real-time video input.
- `feature_extraction.py`: Extracts features from video frames.
- `sign_recognition.py`: Performs sign language recognition.
- `nlp_llm.py`: Handles NLP tasks and LLM integration for text refinement.
- `tts.py`: Converts text to speech.

Further details on the technical design can be found in the project's design documentation.

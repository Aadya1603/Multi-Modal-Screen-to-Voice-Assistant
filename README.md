# Multi-Modal Screen-to-Voice Assistant

A powerful and versatile AI-powered assistant that combines screen capture, voice interaction, and multi-modal processing for an enhanced user experience.

## Description

This project is a significant upgrade to a low-latency screen-to-voice assistant, now powered by a multi-modal system using Llama, OpenAI, and Gemini models. It offers a hands-free, intelligent interface for interacting with your computer and accessing information.

## Features

- **Multi-Modal Processing**: Utilizes Llama, OpenAI, and Gemini for comprehensive analysis
- **Voice Query**: Accepts voice commands for hands-free operation
- **Image Grabbing**: Captures and analyzes relevant images based on voice queries
- **RAG Architecture**: Implemented for more accurate, context-aware responses
- **Clipboard Function**: Easy information copying and pasting
- **Camera Access**: Expanded visual input capabilities
- **Low Latency**: Delivers quick and efficient performance
- **Versatile Responses**: Assists with various domains and topics, offering expert-level answers

## Why Use It?

- **Hands-Free Productivity**: Interact with your computer and access information without typing
- **Multi-Modal Understanding**: Get more accurate and context-aware responses by combining text, voice, and image inputs
- **Flexible AI Models**: Leverage the strengths of multiple AI systems (Llama, OpenAI, Gemini) for optimal performance
- **Enhanced Accessibility**: Great for users with mobility impairments or those who prefer voice interaction
- **Learning and Research**: Quickly gather information and insights on various topics

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/multi-modal-screen-to-voice-assistant.git
   cd multi-modal-screen-to-voice-assistant
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the environment variables for API keys:
   ```bash
   export GROQ_API_KEY='your_groq_api_key'
   export GENAI_API_KEY='your_genai_api_key'
   export OPENAI_API_KEY='your_openai_api_key'
   ```

## Usage

Run the main script to start the assistant:
```bash
python Fox.py
```

Follow the on-screen prompts to interact with the assistant using voice commands.

## Code Overview

The main components of the system include:
- Voice recognition and transcription (using Whisper)
- Multi-modal processing (Llama, OpenAI, Gemini)
- Screen capture and webcam integration
- Text-to-speech output (OpenAI TTS API)

## Pro Tip

If you're low on API credits, you can use the free pyttsx3 library for text-to-speech functionality!

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).

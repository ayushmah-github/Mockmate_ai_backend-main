import os
import uuid
import asyncio
import google.generativeai as genai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google.cloud import texttospeech
from google.cloud import speech
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import base64

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Configure Google AI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize FastAPI app
app = FastAPI(
    title="Voice AI Assistant API",
    description="API for a voice-based conversational AI assistant using Google's Gemini model",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize clients
tts_client = texttospeech.TextToSpeechClient()
stt_client = speech.SpeechClient()

# System prompt to guide the AI behavior
SYSTEM_PROMPT = """
You are an AI assistant engaged in a seamless voice conversation. 
Keep your responses conversational, concise and engaging. 
Ask follow-up questions to maintain the natural flow of conversation.
If the person seems to be done with the current topic, you can suggest a new one.
"""

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = {}
        self.conversation_active = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        # Initialize a new chat session for this client
        model = genai.GenerativeModel('gemini-1.5-pro')
        chat = model.start_chat(history=[
            {"role": "user", "parts": ["Hi, I'd like to have a conversation with you."]},
            {"role": "model", "parts": ["Hello! I'm here and ready to chat with you. What would you like to talk about today?"]}
        ])
        
        # Apply the system prompt to guide the conversation style
        chat.send_message(SYSTEM_PROMPT)
        
        self.active_connections[client_id] = {"websocket": websocket, "chat": chat}
        self.conversation_active[client_id] = False
        
        # Send initial greeting
        await self.send_initial_greeting(client_id)

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.conversation_active:
            del self.conversation_active[client_id]

    async def send_initial_greeting(self, client_id: str):
        """Send an initial greeting to start the conversation"""
        greeting = "Hello! I'm ready to chat with you. What would you like to talk about today?"
        audio_path = await asyncio.to_thread(text_to_speech, greeting)
        
        websocket = self.active_connections[client_id]["websocket"]
        await websocket.send_json({
            "audio_url": audio_path, 
            "text": greeting,
            "status": "speaking"
        })
        
        # Set the conversation as active only after the greeting completes
        await asyncio.sleep(3)  # Approximate time for greeting to play
        await websocket.send_json({"status": "listening"})

    async def process_audio(self, audio_data: bytes, client_id: str):
        """Process audio data using Speech-to-Text"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]["websocket"]
            
            try:
                # Transcribe audio using Google Cloud Speech-to-Text
                transcript = await asyncio.to_thread(speech_to_text, audio_data)
                
                if transcript:
                    # Send transcript back to client for display
                    await websocket.send_json({
                        "transcript": transcript,
                        "status": "thinking"
                    })
                    
                    # Process the transcribed text with Gemini
                    await self.send_message(transcript, client_id)
                else:
                    # If no transcript was returned, go back to listening
                    await websocket.send_json({"status": "listening"})
                    
            except Exception as e:
                await websocket.send_json({
                    "error": f"Speech recognition error: {str(e)}",
                    "status": "listening"
                })

    async def send_message(self, message: str, client_id: str):
        """Process Gemini response and convert to speech"""
        if client_id in self.active_connections and message.strip():
            # Mark conversation as active to prevent multiple parallel responses
            self.conversation_active[client_id] = True
            
            websocket = self.active_connections[client_id]["websocket"]
            chat = self.active_connections[client_id]["chat"]
            
            await websocket.send_json({"status": "thinking"})
            
            try:
                # Get streaming response from Gemini
                response = chat.send_message(message, stream=True)
                
                # Collect the response chunks
                full_response = ""
                for chunk in response:
                    full_response += chunk.text
                
                # Let the frontend know AI is speaking
                await websocket.send_json({"status": "speaking"})
                
                # Convert the full response to speech
                audio_path = await asyncio.to_thread(text_to_speech, full_response)
                
                # Send the audio file path
                await websocket.send_json({
                    "audio_url": audio_path, 
                    "text": full_response,
                    "status": "speaking"
                })
                
                # Approximate time to speak the response based on word count
                # Roughly 150 words per minute speaking rate
                speak_time = len(full_response.split()) / 2.5
                await asyncio.sleep(speak_time)
                
                # When done speaking, switch back to listening mode
                await websocket.send_json({"status": "listening"})
                
            except Exception as e:
                await websocket.send_json({
                    "error": str(e),
                    "status": "listening"
                })
            
            # Mark conversation as no longer active
            self.conversation_active[client_id] = False

    async def handle_system_prompt(self, prompt: str, client_id: str):
        """Handle system prompt updates"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]["websocket"]
            chat = self.active_connections[client_id]["chat"]
            
            try:
                # Update the chat with the new system prompt
                chat.send_message(prompt)
                await websocket.send_json({
                    "text": "System prompt updated successfully.",
                    "status": "listening"
                })
            except Exception as e:
                await websocket.send_json({
                    "error": f"Failed to update system prompt: {str(e)}",
                    "status": "listening"
                })

def text_to_speech(text: str) -> str:
    """Convert text response to speech"""
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.1  # Slightly faster than normal for more natural conversation
    )

    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Ensure static directory exists
    os.makedirs("static", exist_ok=True)
    
    file_name = f"static/speech_{uuid.uuid4()}.mp3"
    with open(file_name, "wb") as out:
        out.write(response.audio_content)

    return f"/{file_name}"  # Return the audio file path

def speech_to_text(audio_content: bytes) -> str:
    """Convert speech to text using Google Cloud Speech-to-Text"""
    # Configure recognition
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
        model="default"
    )
    
    # Create audio object
    audio = speech.RecognitionAudio(content=audio_content)
    
    # Perform synchronous speech recognition
    response = stt_client.recognize(config=config, audio=audio)
    
    # Process the response
    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript
    
    return transcript

manager = ConnectionManager()

@app.get("/")
async def get():
    return FileResponse("static/index.html")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            
            # Check if it's a system prompt update
            if data.startswith("[SYSTEM PROMPT]"):
                prompt = data.replace("[SYSTEM PROMPT]", "").strip()
                await manager.handle_system_prompt(prompt, client_id)
            
            # Check if it's audio data (base64 encoded)
            elif data.startswith("data:audio/"):
                # Extract the base64 encoded audio data
                _, audio_base64 = data.split(',', 1)
                audio_bytes = base64.b64decode(audio_base64)
                
                # Only process if not already in a conversation
                if not manager.conversation_active.get(client_id, False):
                    asyncio.create_task(manager.process_audio(audio_bytes, client_id))
            
            # Regular text input
            else:
                # Only process if not already in a conversation
                if not manager.conversation_active.get(client_id, False):
                    asyncio.create_task(manager.send_message(data, client_id))
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)

# Endpoint for uploading audio files
@app.post("/upload-audio/{client_id}")
async def upload_audio(client_id: str, file: UploadFile = File(...)):
    if client_id in manager.active_connections and not manager.conversation_active.get(client_id, False):
        try:
            audio_content = await file.read()
            asyncio.create_task(manager.process_audio(audio_content, client_id))
            return {"status": "processing"}
        except Exception as e:
            return {"error": str(e)}
    return {"error": "Client not connected or busy"}
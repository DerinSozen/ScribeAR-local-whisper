import asyncio
import websockets
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from queue import Queue
import numpy as np
from scipy.signal import butter, lfilter
import speech_recognition as sr
from datetime import datetime, timedelta

# Set device and precision
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load and compile Whisper model
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)

# Enable static cache and compile the forward pass
model.generation_config.cache_implementation = "static"
model.generation_config.max_new_tokens = 256
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

processor = AutoProcessor.from_pretrained(model_id)

# Setup pipeline for ASR
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Bandpass filter functions
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

lowcut = 300.0
highcut = 3400.0
fs = 16000  # Sample rate

async def handler(websocket, path):
    data_queue = Queue()
    recorder = sr.Recognizer()
    
    # Use microphone as audio source
    with sr.Microphone(sample_rate=16000) as source:
        recorder.adjust_for_ambient_noise(source, 0.5)
        print("Done Calibrating.")
        
        # Listen in background and put data into queue
        recorder.listen_in_background(source, lambda _, audio: data_queue.put(audio.get_raw_data()), phrase_time_limit=2)

        print("Model loaded.")
        
        while True:
            try:
                if not data_queue.empty():
                    # Combine audio data from queue
                    audio_data = b''.join(data_queue.queue)
                    data_queue.queue.clear()
                    
                    # Convert to numpy array and apply bandpass filter
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    filtered_audio_np = bandpass_filter(audio_np, lowcut, highcut, fs)
                    
                    # Transcribe using Whisper model pipeline
                    result = pipe(filtered_audio_np)
                    text = result['text'].strip()
                    
                    await websocket.send(text)
                    
                    print(f"Transcribed Text: {text}")
            except KeyboardInterrupt:
                break

start_server = websockets.serve(handler, "localhost", 8000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
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
model_id = "openai/whisper-small"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)

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
    
    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    # use the default microphone as the audio source
    source = sr.Microphone(sample_rate=16000)
    # recorder.adjust_for_ambient_noise(source,  1)
    with sr.Microphone(sample_rate=16000) as source:                
        recorder.adjust_for_ambient_noise(source, 0.5)
        print("Done Calibrating.")
    
    # recorder.adjust_for_ambient_noise(source)
    # recorder.energy_threshold = 150
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # load model
    record_timeout = 2
    phrase_timeout = 3

    transcription = ['']
    

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)
    
    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.")
    
    while True:
        try:
            now = datetime.now()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now
                
                # Combine audio data from queue
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                # Convert in-ram buffer to something the model can use directly without needing a temp file.
                # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                filtered_audio_np = bandpass_filter(audio_np, lowcut, highcut, fs)
                filtered_audio_np = filtered_audio_np.astype(np.float32)

                # Time when transcription started
                start_time = datetime.now()
                # Read the transcription.
                result = pipe(filtered_audio_np)
                text = result['text'].strip()
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text
                #Calculate the latency
                latency = datetime.now() - start_time
                # text = "Ftesting websocket"
                await websocket.send('F'+transcription[-1]) 

                #DEBUGGING INFO
                print("-----------------------DEBUGGING------------------------")
                print(len(transcription))
                print("Transcribed Text:" + text)
                print(f"Latency: {latency.total_seconds()} seconds")
                print(recorder.energy_threshold)
                print("-----------------------DEBUGGING------------------------")
                #DEBUGGING INFO

        except KeyboardInterrupt:
            break

start_server = websockets.serve(handler, "localhost", 8000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
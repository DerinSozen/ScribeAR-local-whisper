import asyncio
import websockets
import speech_recognition as sr
import threading
import numpy as np
import whisper

import json

from datetime import datetime, timedelta
from queue import Queue

# r = sr.Recognizer()

async def handler(websocket, path):
    
    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    # use the default microphone as the audio source
    source = sr.Microphone(sample_rate=20000)
    # recorder.adjust_for_ambient_noise(source,  1)
    with sr.Microphone(sample_rate=20000) as source:                
        recorder.adjust_for_ambient_noise(source, 0.5)
    
    # recorder.adjust_for_ambient_noise(source)
    # recorder.energy_threshold = 150
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # load model
    audio_model = whisper.load_model("base.en") 
    
    record_timeout = 2
    phrase_timeout = 2.5

    transcription = ['']
    
    # with source:
    #     recorder.adjust_for_ambient_noise(source)

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

                # Time when transcription started
                start_time = datetime.now()
                # Read the transcription.
                result = audio_model.transcribe(audio_np, fp16=False, temperature=0.0)
                text = result['text'].strip()
                #Calculate the latency
                latency = datetime.now() - start_time
                # text = "Ftesting websocket"
                await websocket.send('F'+text) 

                #DEBUGGING INFO
                print("-----------------------DEBUGGING------------------------")
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

import numpy as np
import speech_recognition as sr
import whisper
import re
import asyncio
import websockets
from ctypes import *
from datetime import datetime


# From alsa-lib Git 3fd4ab9be0db7c7430ebd258f2717a976381715d
# $ grep -rn snd_lib_error_handler_t
# include/error.h:59:typedef void (*snd_lib_error_handler_t)(const char *file, int line, const char *function, int err, const char *fmt, ...) /* __attribute__ ((format (printf, 5, 6))) */;
# Define our error handler type
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    print("Alsa Error message")
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

async def handler(websocket):
    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = asyncio.Queue()
    # Use SpeechRecognizer to detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False

    # load model
    audio_model = whisper.load_model("base.en")

    record_timeout = 2
    phrase_timeout = 3

    transcription = ['']

    async def record_callback(audio: sr.AudioData) -> None:
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        await data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    async def listen_in_background():
        while True:
            with sr.microphone(sample_rate=16000) as source:
                audio = recorder.listen(source, phrase_time_limit=record_timeout)
                await record_callback(audio)

    listen_task = asyncio.create_task(listen_in_background())
    print("Starting recording...")
    while True:
        try:
            now = datetime.utcnow()
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
                audio_data = b''.join([await data_queue.get() for _ in range(data_queue.qsize())])

                # Convert in-ram buffer to something the model can use directly without needing a temp file.
                # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Read the transcription.
                print("Transcribing...")
                result = audio_model.transcribe(audio_np, fp16=False, temperature=0.0)
                text = result['text'].strip()

                await websocket.send(text)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

start_server = websockets.serve(handler, "localhost", 1234)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
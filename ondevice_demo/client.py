import sounddevice as sd
import numpy as np
import requests


class Constants:
    sample_rate = 16_000
    window_size = int(sample_rate * 0.5)
    stride_size = int(sample_rate * 0.05)
    server = "http://localhost:31589/asr"

# Globals
audio_buffer = None
token_buffer = []
hidden_state = None


def audio_callback(indata, frames, time, status):
    global audio_buffer, token_buffer

    # Prepare input audio
    new_audio = indata.flatten()
    if audio_buffer is None:
        audio_buffer = new_audio.copy()
    else:
        audio_buffer = np.concatenate((audio_buffer, new_audio))[-Constants.window_size:]

    if len(audio_buffer) == Constants.window_size:
        token = inference(audio_buffer)
        token_buffer.append(token)
        token_buffer = token_buffer[-128:]

        response = requests.post(Constants.server, json={"tokens": token_buffer})
        print(token_buffer)
        print(response.json()["text"])


def load_model():
    return "This is a model."

def inference(audio):
    print(model)
    return 0

def send(token):
    print("send")


if __name__ == "__main__":
    model = load_model()

    try:
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000, blocksize=Constants.stride_size):
            print("Listening... Press Ctrl+C to stop.")
            while True:  # Keep the stream open indefinitely
                sd.sleep(1000)  # Sleep for 1 second at a time to keep the stream alive
    except KeyboardInterrupt:
        print("\nStream stopped by user.")
    except Exception as e:
        print(f"Error: {str(e)}")

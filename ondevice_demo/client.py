import sounddevice as sd
import numpy as np
import requests

import onnxruntime as ort


class Constants:
    sample_rate = 16_000
    #window_size = int(sample_rate * 0.5)
    window_size = 3000
    stride_size = int(sample_rate * 0.05)
    server = "http://localhost:31589/asr"

# Globals
audio_buffer = None
token_buffer = []
hidden_state = None
h_in = np.zeros((1, 1, 1024)).astype(np.float32)
c_in = np.zeros((1, 1, 1024)).astype(np.float32)


def audio_callback(indata, frames, time, status):
    global audio_buffer, token_buffer, h_in, c_in

    # Prepare input audio
    new_audio = indata.flatten()
    if audio_buffer is None:
        audio_buffer = new_audio.copy()
    else:
        audio_buffer = np.concatenate((audio_buffer, new_audio))[-Constants.window_size:]
    print(audio_buffer)

    if len(audio_buffer) == Constants.window_size:
        token, h_in, c_in = inference(audio_buffer, h_in, c_in)
        token_buffer.append(token)
        token_buffer = token_buffer[-128:]
        print(token)
        #response = requests.post(Constants.server, json={"tokens": token_buffer})
        #print(token_buffer)
        #print(response.json()["text"])


def load_model():
    ckpt = "convrnn_model.onnx"
    model = ort.InferenceSession(ckpt)
    return model

def inference(audio, h, c):
    assert len(audio) == 3000
    output = model.run(None, {
        'speech': audio.astype(np.float32),
        'h_in': h,
        'c_in': c
    })
    unit = np.argmax(output[0], axis=-1)
    h_next = output[1]
    c_next = output[2]
    return unit, h_next, c_next

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

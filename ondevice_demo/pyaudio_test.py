import pyaudio
import numpy as np
import wave

# Parameters for the audio stream
FORMAT = pyaudio.paInt16  # Audio format (16-bit)
CHANNELS = 1  # Mono audio
RATE = 16000  # Sample rate (44.1 kHz)
CHUNK = 1024  # Chunk size (number of frames per buffer)
OUTPUT_FILENAME = "output_streamed.wav"  # Output file name

# Initialize PyAudio object
p = pyaudio.PyAudio()

# Open the stream with a callback
def audio_callback(indata, frames, time, status):
    """
    This function will be called every time a chunk of audio data is available.
    """
    # Write the incoming data to a file (or process it in real-time)
    audio = np.frombuffer(indata, dtype=np.int16)
    print(np.abs(audio).mean())
    return (None, pyaudio.paContinue)  # Return no data and continue the stream


if __name__ == "__main__":
    # Open the audio stream with the callback
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=audio_callback)

    # Start the stream
    print("Start stream...")
    stream.start_stream()

    try:
        while stream.is_active():
            # Stream will continue until manually stopped
            pass
    except KeyboardInterrupt:
        print("End stream.")
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

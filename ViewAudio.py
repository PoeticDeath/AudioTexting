import queue
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import sounddevice as sd

# 1. Configuration
SAMPLE_RATE = 12000  # 12 kHz sample rate
FFT_SIZE = 256      # Size of the FFT window (higher = more frequency detail)

# Create a thread-safe queue to pass audio data
audio_queue = queue.Queue()

# 2. Generate a frequency-dependent "Equalizer" curve to emphasize 1kHz and 2kHz
# This calculates the exact frequency mapping for each point on our graph
frequencies = np.fft.rfftfreq(FFT_SIZE, d=1.0/SAMPLE_RATE)

# Create a baseline curve (1.0 means no change)
emphasis_mask = np.ones_like(frequencies)

def add_frequency_boost(mask, center_hz, width_hz, boost_factor):
    """Applies a smooth Gaussian boost curve around a target frequency."""
    return mask + boost_factor * np.exp(-((frequencies - center_hz) / width_hz) ** 2)

# Boost 1,000 Hz and 2,000 Hz by a factor of 4x (adjust width/boost as needed)
#emphasis_mask = add_frequency_boost(emphasis_mask, center_hz=1000, width_hz=150, boost_factor=4.0)
#emphasis_mask = add_frequency_boost(emphasis_mask, center_hz=2000, width_hz=150, boost_factor=4.0)

# 3. Audio Callback Function
def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata[:, 0])

# 4. Graph Update Function (FFT Spectrum Analyzer)
audio_buffer = np.zeros(FFT_SIZE)

def update_plot(frame):
    global audio_buffer
    
    # Pull data from queue
    while True:
        try:
            data = audio_queue.get_nowait()
        except queue.Empty:
            break
        
        # Keep our buffer filled with the latest chunk of audio
        shift = len(data)
        audio_buffer = np.roll(audio_buffer, -shift)
        audio_buffer[-shift:] = data

    # Apply a Hanning window to prevent digital edge-artifacts
    windowed_audio = audio_buffer * np.hanning(FFT_SIZE)
    
    # Compute the Real FFT (converts time audio into frequency amplitudes)
    fft_magnitude = np.abs(np.fft.rfft(windowed_audio))
    
    # Apply our custom emphasis mask to amplify 1kHz and 2kHz
    emphasized_magnitude = fft_magnitude #* emphasis_mask

    # Update the graph line
    line.set_ydata(emphasized_magnitude)
    return line,

# 5. Initialize Setup the Plot
fig, ax = plt.subplots()
line, = ax.plot(frequencies, np.zeros_like(frequencies), color='#00ffcc', lw=2)

# Style the graph
ax.set_facecolor('#111111')
fig.patch.set_facecolor('#111111')
ax.set_xlim(0, 8000)       # Cut graph at 4 kHz since FRS radios drop completely at 3 kHz
ax.set_ylim(0, 100)         # Max amplitude height (adjust based on mic sensitivity)
ax.set_title("Live FRS Frequency Spectrum (1kHz & 2kHz Emphasized)", color='white')
ax.set_xlabel("Frequency (Hz)", color='white')
ax.set_ylabel("Amplitude", color='white')
ax.tick_params(colors='white')
ax.grid(True, color='#333333', linestyle='--')

# Highlight the target zones visually with background shading
ax.axvspan(850, 1150, color='#ff0055', alpha=0.15, label='1 kHz Zone')
ax.axvspan(1850, 2150, color='#ff0055', alpha=0.15, label='2 kHz Zone')
ax.legend(loc='upper right', facecolor='#222222', labelcolor='white')

# 6. Start the Live Stream
try:
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=256  # Smaller blocks for highly responsive FFT updates
    )
    
    with stream:
        ani = FuncAnimation(fig, update_plot, interval=20, blit=True, cache_frame_data=False)
        plt.show()

except Exception as e:
    print(f"Error starting audio stream: {e}")


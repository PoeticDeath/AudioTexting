import numpy as np
import sounddevice as sd

# Configurations
SAMPLE_RATE = 8000
BAUD_RATE = 100
FREQ_MARK = 1200    # Bit 1
FREQ_SPACE = 2200   # Bit 0

PREAMBLE_BITS_COUNT = 16  # 16 alternating bits (10101010...) to lock receiver clock
START_FLAG = b"$$"         # Unique marker indicating data begins next

def bytes_to_bits(text):
    """Encodes text into bits with Hamming(7,4) error correction."""
    raw_bits = []
    for char in text:
        raw_bits.extend([int(b) for b in f"{char:08b}"])
    
    encoded_bits = []
    for i in range(0, len(raw_bits), 4):
        chunk = raw_bits[i:i+4]
        if len(chunk) < 4:
            chunk += [0] * (4 - len(chunk))
            
        d1, d2, d3, d4 = chunk
        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p3 = d2 ^ d3 ^ d4
        encoded_bits.extend([p1, p2, d1, p3, d2, d3, d4])
        
    return encoded_bits

def generate_afsk_signal_with_flag(text):
    # 1. Generate Synchronizing Preamble (101010...)
    preamble_bits = [1 if i % 2 == 0 else 0 for i in range(PREAMBLE_BITS_COUNT)]
    
    if isinstance(text, str):
        text = text.encode('utf-8')
    elif isinstance(text, bytes):
        pass
    else:
        raise TypeError
    
    # 2. Add Payload Length
    text = len(text).to_bytes(8, "big") + text
        
    payload_bits = bytes_to_bits(START_FLAG + text)
    
    # 3. Add Flush Tail to push the final bits entirely into the audio buffer
    tail_bits = [1] * ((SAMPLE_RATE + 1999) // 2000)
    
    # Combine into one full bit array
    all_bits = preamble_bits + payload_bits + tail_bits
    
    # 4. Synthesize Continuous Audio
    samples_per_bit = int(SAMPLE_RATE / BAUD_RATE)
    total_samples = len(all_bits) * samples_per_bit
    
    frequencies = np.zeros(total_samples)
    for i, bit in enumerate(all_bits):
        start = i * samples_per_bit
        end = start + samples_per_bit
        frequencies[start:end] = FREQ_MARK if bit == 1 else FREQ_SPACE
        
    # Standard phase-continuous integration
    phase = 2 * np.pi * np.cumsum(frequencies) / SAMPLE_RATE
    signal = np.sin(phase)
    
    audio_8bit = ((signal + 1) * 127.5).astype(np.uint8)
    return audio_8bit

def send(message="Hello, World!\nTesting."):
    # --- RUN THE TRANSMITTER ---
    print(f"Transmitting message with clock-sync preamble and tail flush...")
    audio_data = generate_afsk_signal_with_flag(message)
    playback_buffer = (audio_data.astype(np.float32) - 127.5) / 127.5
    sd.play(playback_buffer, SAMPLE_RATE)
    sd.wait()
    print("Transmission finished.")

if __name__ == "__main__":
    send()

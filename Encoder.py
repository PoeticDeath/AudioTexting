import numpy as np
import sounddevice as sd
import time

SAMPLE_RATE = 12000
BAUD_RATE = 1500    
FREQ_MARK = 1000    # Bit 1
FREQ_SPACE = 2000   # Bit 0

PREAMBLE_BITS_COUNT = 750
START_FLAG = b"$$"   

PAYLOADS_SENT = 0

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

def generate_afsk_frame(frame_bytes):
    """Synthesizes a single frame (header + chunk) into an AFSK audio array."""
    # 1. Short clock-sync preamble
    preamble_bits = [1 for _ in range(PREAMBLE_BITS_COUNT)]
    
    # 2. Add size prefix required by your decoder's inner reader, then the payload
    inner_payload = len(frame_bytes).to_bytes(8, "big") + frame_bytes
    payload_bits = bytes_to_bits(START_FLAG + inner_payload)
    
    # 3. Flush tail
    tail_bits = [1, 1, 1, 1] * ((SAMPLE_RATE + 99) // 100)
    all_bits = preamble_bits + payload_bits + tail_bits
    
    # 4. Synthesize Continuous Phase Audio
    samples_per_bit = int(SAMPLE_RATE / BAUD_RATE)
    total_samples = len(all_bits) * samples_per_bit
    
    frequencies = np.zeros(total_samples)
    for i, bit in enumerate(all_bits):
        start = i * samples_per_bit
        end = start + samples_per_bit
        frequencies[start:end] = FREQ_MARK if bit == 1 else FREQ_SPACE
        
    phase = 2 * np.pi * np.cumsum(frequencies) / SAMPLE_RATE
    signal = np.sin(phase)
    
    audio_8bit = ((signal + 1) * 127.5).astype(np.uint8)
    return audio_8bit

def send(message, chunk_size=64):
    if isinstance(message, str):
        message = message.encode('utf-8')
    
    global PAYLOADS_SENT
    packet_id = PAYLOADS_SENT
    PAYLOADS_SENT += 1
    if PAYLOADS_SENT == 65536:
        PAYLOADS_SENT = 0

    # Slice the big payload into fragments
    chunks = [message[i:i + chunk_size] for i in range(0, len(message), chunk_size)]
    total_frames = len(chunks)
    
    print(f"[*] Splitting {len(message)} bytes into {total_frames} fragments (chunk size: {chunk_size} bytes)...")
    
    for seq_num, chunk in enumerate(chunks):
        # 8-byte Frame Header metadata
        header = (
            packet_id.to_bytes(2, "big") +
            seq_num.to_bytes(2, "big") +
            total_frames.to_bytes(2, "big") +
            len(chunk).to_bytes(2, "big")
        )
        
        frame_bytes = header + chunk
        print(f"[TX] Transmitting frame {seq_num + 1}/{total_frames}...", end="\r")
        
        audio_data = generate_afsk_frame(frame_bytes)
        playback_buffer = (audio_data.astype(np.float32) - 127.5) / 127.5
        
        sd.play(playback_buffer, SAMPLE_RATE)
        sd.wait()
        
        # Guard interval to allow the receiver's hardware buffer to breathe
        time.sleep(0.08)
        
    print(f"\n[+] Transmission of Packet {packet_id} complete.")

if __name__ == "__main__":
    # Simulate a large 1600-byte payload
    test_large_payload = "A" * 1600 
    send(test_large_payload)


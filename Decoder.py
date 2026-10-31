import numpy as np
import sounddevice as sd
import queue
import threading

SAMPLE_RATE = 12000
BAUD_RATE = 3000
FREQ_MARK = 1000
FREQ_SPACE = 3000

Messages = []

SAMPLES_PER_BIT = int(SAMPLE_RATE / BAUD_RATE)  # 80 samples
CHUNK_SIZE = SAMPLES_PER_BIT * 15  

audio_queue = queue.Queue()

def bits_to_bytes(bits):
    """Decodes Hamming(7,4) bits, correcting single-bit errors."""
    decoded_bits = []
    for i in range(0, len(bits), 7):
        block = bits[i:i+7]
        if len(block) < 7: break
        p1, p2, d1, p3, d2, d3, d4 = block
        s1 = p1 ^ d1 ^ d2 ^ d4
        s2 = p2 ^ d1 ^ d3 ^ d4
        s3 = p3 ^ d2 ^ d3 ^ d4
        syndrome = s1 + (s2 << 1) + (s3 << 2)
        if syndrome == 3: d1 ^= 1
        elif syndrome == 5: d2 ^= 1
        elif syndrome == 6: d3 ^= 1
        elif syndrome == 7: d4 ^= 1
        decoded_bits.extend([d1, d2, d3, d4])
        
    chars = []
    for i in range(0, len(decoded_bits), 8):
        byte_bits = decoded_bits[i:i+8]
        if len(byte_bits) < 8: break
        byte_str = "".join(str(b) for b in byte_bits)
        chars.append(int(byte_str, 2))
    try:
        return bytes(chars)
    except Exception:
        return ""

def bytes_to_bits(text):
    raw_bits = []
    for char in text:
        raw_bits.extend([int(b) for b in f"{char:08b}"])
    encoded_bits = []
    for i in range(0, len(raw_bits), 4):
        chunk = raw_bits[i:i+4]
        if len(chunk) < 4: chunk += [0] * (4 - len(chunk))
        d1, d2, d3, d4 = chunk
        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p3 = d2 ^ d3 ^ d4
        encoded_bits.extend([p1, p2, d1, p3, d2, d3, d4])
    return encoded_bits

FLAG = b"$$"
START_FLAG_BITS = bytes_to_bits(FLAG)
FLAG_LEN = len(START_FLAG_BITS)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.flatten())

def processing_worker():
    """Continuously demodulates audio via I-Q cross-correlation and tracks packets

    using a high-resolution sliding bit synchronizer to hit 100% accuracy.
    """
    print("Demodulator worker thread started (Phase-Locked Timing Mode)...")
    
    audio_buffer = np.array([], dtype='float32')
    
    # 4x oversampling parameters
    OVERSAMPLE_FACTOR = 4 
    SAMPLES_PER_STEP = SAMPLES_PER_BIT // OVERSAMPLE_FACTOR 
    
    # High-density oversampled raw decisions buffer
    oversampled_stream = [] 
    
    # State Machine Variables
    STATE_LOOKING_FOR_FLAG = 0
    STATE_READING_PAYLOAD = 1
    current_state = STATE_LOOKING_FOR_FLAG

    # Precompute Digital Bandpass Correlation Waves
    t = np.arange(SAMPLES_PER_BIT) / SAMPLE_RATE
    mark_cos = np.cos(2 * np.pi * FREQ_MARK * t)
    mark_sin = np.sin(2 * np.pi * FREQ_MARK * t)
    space_cos = np.cos(2 * np.pi * FREQ_SPACE * t)
    space_sin = np.sin(2 * np.pi * FREQ_SPACE * t)

    while True:
        try:
            chunk = audio_queue.get()
            audio_buffer = np.append(audio_buffer, chunk)
            
            # Step 1: Precision Demodulation (Fills the high-density buffer)
            while len(audio_buffer) >= SAMPLES_PER_BIT:
                bit_window = audio_buffer[:SAMPLES_PER_BIT]
                audio_buffer = audio_buffer[SAMPLES_PER_STEP:] 
                
                if np.max(np.abs(bit_window)) < 0.015:
                    oversampled_stream.append(1) # Squelch defaults to Mark (1)
                    continue 
                
                mark_mag = np.sqrt(np.sum(bit_window * mark_cos)**2 + np.sum(bit_window * mark_sin)**2)
                space_mag = np.sqrt(np.sum(bit_window * space_cos)**2 + np.sum(bit_window * space_sin)**2)
                
                oversampled_stream.append(1 if mark_mag > space_mag else 0)
            
            # --- Step 2: High-Resolution Packet Extraction State Machine ---
            
            if current_state == STATE_LOOKING_FOR_FLAG:
                # Calculate flag length in terms of oversampled micro-steps
                # FLAG_LEN (28 bits) * 4 oversamples = 112 steps
                oversampled_flag_len = FLAG_LEN * OVERSAMPLE_FACTOR
                
                if len(oversampled_stream) >= oversampled_flag_len:
                    # Slide sample-by-sample to find the phase phase phase offset
                    for i in range(len(oversampled_stream) - oversampled_flag_len + 1):
                        
                        # Downsample a candidate stream based on this specific offset 'i'
                        test_bits = [0] * FLAG_LEN
                        for b in range(FLAG_LEN):
                            # Grab 4 oversamples starting at our current offset
                            idx = i + (b * OVERSAMPLE_FACTOR)
                            bit_slice = oversampled_stream[idx : idx + OVERSAMPLE_FACTOR]
                            # Majority vote
                            if sum(bit_slice) > (OVERSAMPLE_FACTOR / 2):
                                test_bits[b] = 1
                        
                        # Test if this specific phase alignment unlocks the flag
                        if bits_to_bytes(test_bits) == FLAG:
                            print(f"\n[SYNC] Phase Locked! Phase shift offset: {i % 4}")
                            current_state = STATE_READING_PAYLOAD
                            
                            # Trim everything before this perfect starting point
                            oversampled_stream = oversampled_stream[i:]
                            break
                    
                    # Prevent memory leak when hunting for a flag
                    if current_state == STATE_LOOKING_FOR_FLAG and len(oversampled_stream) > 4000:
                        oversampled_stream = oversampled_stream[-oversampled_flag_len:]

            if current_state == STATE_READING_PAYLOAD:
                # Build the synchronized bitstream using the established phase offset
                synchronized_bits = []
                
                # Check how many whole bits are fully available in the oversampled buffer
                available_bits = len(oversampled_stream) // OVERSAMPLE_FACTOR
                
                for b in range(available_bits):
                    idx = b * OVERSAMPLE_FACTOR
                    bit_slice = oversampled_stream[idx : idx + OVERSAMPLE_FACTOR]
                    synchronized_bits.append(1 if sum(bit_slice) > (OVERSAMPLE_FACTOR / 2) else 0)
                
                # Skip the flag bits themselves to look at the text payload
                payload_bits = synchronized_bits[FLAG_LEN:]
                
                if len(payload_bits) >= 14: # Minimum 1 character
                    decoded = bits_to_bytes(payload_bits)
                    print(f"\r[DECODING]: {decoded}", end="", flush=True)
                    
                    length = int.from_bytes(decoded[:8], "big")
                    if (len(decoded[8:]) >= length) and (len(decoded) >= 8):
                        print(f"\n[RECEIVED COMPLETE]: {decoded[8:][:length]}")
                        Messages.append(decoded[8:][:length])
                        
                        # Calculate exactly how many raw text characters were sent
                        chars_transmitted = len(decoded)
                        
                        # total bits = $$ flag bits (28) + payload bits (chars * 14)
                        total_bits_consumed = FLAG_LEN + (chars_transmitted * 14)
                        oversampled_consumed = total_bits_consumed * OVERSAMPLE_FACTOR
                        
                        # Chop off the processed message, perfectly preserving the remaining 
                        # oversamples for the next back-to-back preamble!
                        oversampled_stream = oversampled_stream[oversampled_consumed:]
                        
                        current_state = STATE_LOOKING_FOR_FLAG
                                
        except Exception as e:
            print(f"\nWorker Error: {e}")
        finally:
            audio_queue.task_done()

def recv():
    worker_thread = threading.Thread(target=processing_worker, daemon=True)
    worker_thread.start()
    
    print("Initializing real-time stream. Press Ctrl+C to stop.")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, 
                            channels=1, 
                            callback=audio_callback, 
                            blocksize=CHUNK_SIZE, 
                            dtype='float32'):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopping Decoder...")

if __name__ == "__main__":
    recv()

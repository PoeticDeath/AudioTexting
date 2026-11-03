import numpy as np
import sounddevice as sd
import queue
import threading

SAMPLE_RATE = 12000
BAUD_RATE = 300
FREQ_MARK = 1000
FREQ_SPACE = 2000

Messages = []

SAMPLES_PER_BIT = int(SAMPLE_RATE / BAUD_RATE)  
CHUNK_SIZE = SAMPLES_PER_BIT * 15  

audio_queue = queue.Queue()

# Thread-safe dictionary to gather packets: { packet_id: { seq_num: payload_bytes } }
REASSEMBLY_POOL = {}

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
        return b""

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

def process_decoded_frame(frame_bytes):
    """Parses fragment metadata headers and reconstructs original file streams."""
    if len(frame_bytes) < 8:
        print("\n[ERR] Frame too small to parse headers.")
        return

    # Unpack 8-byte Frame Header
    packet_id = int.from_bytes(frame_bytes[0:2], "big")
    seq_num = int.from_bytes(frame_bytes[2:4], "big")
    total_frames = int.from_bytes(frame_bytes[4:6], "big")
    payload_len = int.from_bytes(frame_bytes[6:8], "big")
    
    payload = frame_bytes[8 : 8 + payload_len]
    
    if packet_id not in REASSEMBLY_POOL:
        REASSEMBLY_POOL[packet_id] = {}
        
    REASSEMBLY_POOL[packet_id][seq_num] = payload
    print(f"[RECV] Received fragment {seq_num + 1}/{total_frames} for Packet ID: {packet_id}")
    
    # Verify if entire file array has been captured
    if len(REASSEMBLY_POOL[packet_id]) == total_frames:
        print(f"[SUCCESS] Reassembling Packet {packet_id}...")
        full_payload = b""
        for i in range(total_frames):
            if i in REASSEMBLY_POOL[packet_id]:
                full_payload += REASSEMBLY_POOL[packet_id][i]
            else:
                print(f"[ERR] Missing fragment index {i}! Reassembly aborted.")
                return
                
        print("=" * 40)
        print(f"REASSEMBLED PAYLOAD ({len(full_payload)} Bytes):")
        print(full_payload)
        Messages.append(full_payload)
        print("=" * 40)
        
        # Clean memory pool
        del REASSEMBLY_POOL[packet_id]

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.flatten())

def processing_worker():
    print("Demodulator worker thread started (Fragmented Frame Reassembly Mode)...")
    audio_buffer = np.array([], dtype='float32')
    OVERSAMPLE_FACTOR = 4 
    SAMPLES_PER_STEP = SAMPLES_PER_BIT // OVERSAMPLE_FACTOR 
    oversampled_stream = []
    
    STATE_LOOKING_FOR_FLAG = 0
    STATE_READING_PAYLOAD = 1
    current_state = STATE_LOOKING_FOR_FLAG

    t = np.arange(SAMPLES_PER_BIT) / SAMPLE_RATE
    mark_cos = np.cos(2 * np.pi * FREQ_MARK * t)
    mark_sin = np.sin(2 * np.pi * FREQ_MARK * t)
    space_cos = np.cos(2 * np.pi * FREQ_SPACE * t)
    space_sin = np.sin(2 * np.pi * FREQ_SPACE * t)

    while True:
        try:
            chunk = audio_queue.get()
            audio_buffer = np.append(audio_buffer, chunk)
            
            while len(audio_buffer) >= SAMPLES_PER_BIT:
                bit_window = audio_buffer[:SAMPLES_PER_BIT]
                audio_buffer = audio_buffer[SAMPLES_PER_STEP:] 
                
                if np.max(np.abs(bit_window)) < 0.02:
                    continue 
                
                mark_mag = np.sqrt(np.sum(bit_window * mark_cos)**2 + np.sum(bit_window * mark_sin)**2)
                space_mag = np.sqrt(np.sum(bit_window * space_cos)**2 + np.sum(bit_window * space_sin)**2)
                oversampled_stream.append(1 if mark_mag > space_mag else 0)
            
            if current_state == STATE_LOOKING_FOR_FLAG:
                oversampled_flag_len = FLAG_LEN * OVERSAMPLE_FACTOR
                if len(oversampled_stream) >= oversampled_flag_len:
                    try:
                        for i in range(oversampled_stream.index(0), len(oversampled_stream) - oversampled_flag_len + 1):
                            test_bits = [0] * FLAG_LEN
                            for b in range(FLAG_LEN):
                                idx = i + (b * OVERSAMPLE_FACTOR)
                                bit_slice = oversampled_stream[idx : idx + OVERSAMPLE_FACTOR]
                                if sum(bit_slice) > (OVERSAMPLE_FACTOR / 2):
                                    test_bits[b] = 1
                        
                            if bits_to_bytes(test_bits) == FLAG:
                                current_state = STATE_READING_PAYLOAD
                                oversampled_stream = oversampled_stream[i:]
                                break
                    except ValueError:
                        pass
                    
                    if current_state == STATE_LOOKING_FOR_FLAG and len(oversampled_stream) >= oversampled_flag_len:
                        oversampled_stream = oversampled_stream[-oversampled_flag_len:]

            if current_state == STATE_READING_PAYLOAD:
                available_bits = len(oversampled_stream) // OVERSAMPLE_FACTOR
                synchronized_bits = [0] * available_bits
                for b in range(available_bits):
                    idx = b * OVERSAMPLE_FACTOR
                    bit_slice = oversampled_stream[idx : idx + OVERSAMPLE_FACTOR]
                    if sum(bit_slice) > (OVERSAMPLE_FACTOR / 2):
                        synchronized_bits[b] = 1
                
                payload_bits = synchronized_bits[FLAG_LEN:]
                
                if len(payload_bits) >= 14: 
                    decoded = bits_to_bytes(payload_bits)
                    if len(decoded) >= 8:
                        length = int.from_bytes(decoded[:8], "big")
                        # Once our fragment data matches the inner expected window length
                        if len(decoded[8:]) >= length:
                            frame_data = decoded[8:][:length]
                            
                            # Route raw payload directly out to the fragment manager
                            process_decoded_frame(frame_data)
                            
                            # Chop exactly the consumed bits out of our streaming buffer
                            chars_transmitted = len(decoded)
                            total_bits_consumed = FLAG_LEN + (chars_transmitted * 14)
                            oversampled_consumed = total_bits_consumed * OVERSAMPLE_FACTOR
                            oversampled_stream = oversampled_stream[oversampled_consumed:]
                            
                            current_state = STATE_LOOKING_FOR_FLAG
                                
        except Exception as e:
            print(f"\nWorker Error: {e}")
        finally:
            audio_queue.task_done()

def recv():
    worker_thread = threading.Thread(target=processing_worker, daemon=True)
    worker_thread.start()
    print("Initializing real-time fragmented decoder. Press Ctrl+C to stop.")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SIZE, dtype='float32'):
            while True: sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopping Decoder...")

if __name__ == "__main__":
    recv()


class ConvolutionalCodec:
    def __init__(self):
        # Generator polynomials for Rate 1/2, K=3
        # G1 = 111 (binary) -> [1, 1, 1]
        # G2 = 101 (binary) -> [1, 0, 1]
        self.g1 = [1, 1, 1]
        self.g2 = [1, 0, 1]
        self.K = 3
        self.num_states = 4
        
        # Precompute state transitions for the Viterbi Decoder
        self.transitions = {}
        for state in range(self.num_states):
            self.transitions[state] = {}
            for input_bit in [0, 1]:
                # Shift register contents: [input_bit, state_bit1, state_bit2]
                reg = [input_bit, (state >> 1) & 1, state & 1]
                
                # Calculate output parity bits using XOR
                out1 = sum(r * g for r, g in zip(reg, self.g1)) % 2
                out2 = sum(r * g for r, g in zip(reg, self.g2)) % 2
                
                # Next state drops oldest bit and shifts new input bit in
                next_state = ((state << 1) | input_bit) & 0x3
                
                self.transitions[state][input_bit] = (next_state, [out1, out2])

    def encode(self, message):
        """Encodes an array of input bits into a stream of parity bits."""
        state = 0  # Initial state
        encoded_stream = []
        for bit in message:
            next_state, outputs = self.transitions[state][bit]
            encoded_stream.extend(outputs)
            state = next_state
        return encoded_stream

    def viterbi_decode(self, received_stream):
        """Decodes a noisy stream of bits back into the original message."""
        num_steps = len(received_stream) // 2
        
        # Track minimum Hamming distance and input bit history for each state
        path_metrics = {0: 0, 1: float('inf'), 2: float('inf'), 3: float('inf')}
        history = {0: [], 1: [], 2: [], 3: []}
        
        for step in range(num_steps):
            rec_bits = received_stream[step*2 : step*2 + 2]
            
            next_path_metrics = {s: float('inf') for s in range(self.num_states)}
            next_history = {s: [] for s in range(self.num_states)}
            
            for curr_state in range(self.num_states):
                if path_metrics[curr_state] == float('inf'):
                    continue
                    
                for input_bit in [0, 1]:
                    next_state, expected_outputs = self.transitions[curr_state][input_bit]
                    
                    # Calculate Hamming distance between received and expected bits
                    distance = sum(b1 != b2 for b1, b2 in zip(rec_bits, expected_outputs))
                    new_metric = path_metrics[curr_state] + distance
                    
                    if new_metric < next_path_metrics[next_state]:
                        next_path_metrics[next_state] = new_metric
                        next_history[next_state] = history[curr_state] + [input_bit]
            
            path_metrics = next_path_metrics
            history = next_history
            
        best_final_state = min(path_metrics, key=path_metrics.get)
        return history[best_final_state]


# --- Demonstration / Testing ---
if __name__ == "__main__":
    codec = ConvolutionalCodec()
    
    # 1. Original test message bits
    original_message = [1, 0, 1, 1, 0]
    print(f"Original Bits: {original_message}")
    
    # 2. Encode
    encoded_bits = codec.encode(original_message)
    print(f"Encoded Bits:  {encoded_bits}")
    
    # 3. Transmit over a clean channel and decode
    decoded_message = codec.viterbi_decode(encoded_bits)
    print(f"Decoded Bits:  {decoded_message}")
    
    if original_message == decoded_message:
        print("\nSuccess! Code runs cleanly without errors.")


import sys
import os
import time
import numpy as np
import socket
import argparse

from config import InferenceConfig
from kvcache import KVCache
from engine import ACLModel, initResource
from session import recv_msg, send_msg, log_debug
import socket
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, help="Path to OM model file")
    parser.add_argument('--port', type=int, required=True, help="Port to listen on (from prev node)")
    parser.add_argument('--next_ip', type=str, default='127.0.0.1', help="IP of next node")
    parser.add_argument('--next_port', type=int, required=True, help="Port of next node")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--hf_dir', type=str, required=True, help="Path to HF config for init")
    parser.add_argument('--n_layer', type=int, default=6, help="Number of layers in this block")
    parser.add_argument('--node_name', type=str, default="Node", help="Name for logging")
    parser.add_argument('--kv_size', type=int, default=1024, help="KV Cache max size (must match OM model)")
    parser.add_argument('--kv_method', type=str, default="sliding-window", help="basic | sliding-window | streamllm | H2O")
    args = parser.parse_args()

    node_name = args.node_name
    print(f"[{node_name}] Initializing...")

    # 1. Setup Network
    # Server (Listen for Previous Node)
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Retry bind if port is busy (optional, but good for restart)
    try:
        server_sock.bind(('0.0.0.0', args.port))
    except OSError as e:
        print(f"[{node_name}] Bind failed: {e}")
        return

    server_sock.listen(1)
    print(f"[{node_name}] Listening on port {args.port}...")

    # Accept connection (Blocking)
    conn_prev, addr_prev = server_sock.accept()
    print(f"[{node_name}] Connected by Previous Node: {addr_prev}")

    # Client (Connect to Next Node)
    # We wait for server connection first, or concurrent?
    # Usually safer to connect after accept effectively forms a chain.
    # But if everyone waits for accept first, we need reverse launch order or non-blocking.
    # Distributed logic: Node 4 listens. Node 3 listens. Node 2 listens. Node 1 connects to 2, 2 to 3, 3 to 4.
    # So we should Listen first, THEN connect next. 
    # Because Node N-1 connects to Node N. So Node N must Listen first.
    
    print(f"[{node_name}] Connecting to Next Node at {args.next_ip}:{args.next_port}...")
    client_sock_next = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            client_sock_next.connect((args.next_ip, args.next_port))
            print(f"[{node_name}] Connected to Next Node!")
            break
        except ConnectionRefusedError:
            time.sleep(1)
            print(f"[{node_name}] Retrying connection to Next Node...")

    # 2. Init Resources
    context = initResource(args.device)
    
    # 3. Init Config & KVCache
    user_config = InferenceConfig(
        hf_model_dir=args.hf_dir,
        device=args.device,
        max_cache_size=args.kv_size,
        model=args.model,
        # IMPORTANT: Only allocate KV cache for local layers
        # But InferenceConfig loads n_layer from json in post_init, overwriting constructor args if we aren't careful?
        # InferenceConfig.__post_init__ reads headers from json.
        # We need to manually overwrite n_layer AFTER init if possible, or modify config logic.
        # Let's inspect InferenceConfig again.
    )
    # Re-verify config... config overrides n_layer from file.
    # We must overwrite it manually.
    user_config.n_layer = args.n_layer
    user_config.max_input_len = 1 # Assuming token generation phase mostly
    # But wait, prefill phase (first run) might have seq_len > 1 if we supported it. 
    # Current codebase seems to handle max_input_len=1 for OM generally, but session.py run loop handles slices.
    # The user instruction `config.max_input_len` is used in Session.
    
    kv_cache = KVCache.create(user_config)
    print(f"[{node_name}] KVCache initialized for {args.n_layer} layers.")

    # 4. Load Model
    print(f"[{node_name}] Loading model: {args.model}")
    model = ACLModel(args.model, context=context, mode=user_config.acl_mode)
    
    # 5. Inference Loop
    print(f"[{node_name}] Ready for inference loop...")
    step = 0
    while True:
        # Receive [hidden_states, input_ids, mask, pos_ids, kv_0]
        msg = recv_msg(conn_prev)
        if msg is None:
            print(f"[{node_name}] Connection closed by prev node.")
            break
        
        # Unpack
        # msg: [hidden, input_ids, mask, pos_ids, kv0_past]
        hidden_states = msg[0]
        input_ids = msg[1]     # shape (1, seq_len)
        mask = msg[2]
        pos_ids = msg[3]
        kv0_past = msg[4]
        
        log_debug(f"debug_{node_name.replace(' ', '')}.log", f"Step {step}: Received Msg", hidden_states)
        
        seq_len = input_ids.shape[1]
        
        # Prepare Local KV
        local_cache_ret = kv_cache.getInputs(user_config.max_length)
        local_kvs_list = local_cache_ret[0] 
        
        # Prepare ACL Inputs
        acl_inputs = [hidden_states, input_ids, mask, pos_ids, kv0_past]
        acl_inputs.extend(local_kvs_list)
        
        log_debug(f"debug_{node_name.replace(' ', '')}.log", f"Step {step}: OM Input Hidden", hidden_states)
        log_debug(f"debug_{node_name.replace(' ', '')}.log", f"Step {step}: OM Input Cache", local_kvs_list[0] if local_kvs_list else [])

        # Inference
        res = model.inference(acl_inputs)
        # Result: [hidden_out, new_kv_0, new_kv_1, ...]
        hidden_out = res[0]
        new_kvs = res[1:]
        
        log_debug(f"debug_{node_name.replace(' ', '')}.log", f"Step {step}: Output Hidden", hidden_out)
        
        # Update Local KV Cache
        kv_cache.update(seq_len, new_kvs, None)
        
        # Send to Next Node
        new_msg = [hidden_out, input_ids, mask, pos_ids, kv0_past]
        send_msg(client_sock_next, new_msg)
        
        step += 1

    conn_prev.close()
    client_sock_next.close()

if __name__ == "__main__":
    main()

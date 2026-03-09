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
import argparse
import socket
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Path to M3 (Final) OM model file")
    parser.add_argument('--port', type=int, required=True, help="Port to listen on (from prev node)")
    parser.add_argument('--head_ip', type=str, default='127.0.0.1', help="IP of Head Node (Node 1)")
    parser.add_argument('--head_port', type=int, default=8004, help="Port of Head Node (Node 1)")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--hf_dir', type=str, required=True, help="Path to HF config for init")
    parser.add_argument('--n_layer', type=int, default=5, help="Number of layers in this block (M3 usually 17-21=5 layers)")
    parser.add_argument('--node_name', type=str, default="Node 4", help="Name for logging")
    parser.add_argument('--kv_size', type=int, default=1024, help="KV Cache max size (must match OM model)")
    parser.add_argument('--kv_method', type=str, default="sliding-window", help="basic | sliding-window | streamllm | H2O")
    
    args = parser.parse_args()

    node_name = args.node_name
    print(f"[{node_name}] Initializing...")

    # 1. Setup Network
    # Server (Listen for Node 3)
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_sock.bind(('0.0.0.0', args.port))
    except OSError as e:
        print(f"[{node_name}] Bind failed: {e}")
        return
    server_sock.listen(1)
    
    print(f"[{node_name}] Listening on port {args.port}...")
    conn_prev, addr_prev = server_sock.accept()
    print(f"[{node_name}] Connected by Previous Node: {addr_prev}")

    # Client (Connect to Head Node / Node 1)
    print(f"[{node_name}] Connecting to Head Node at {args.head_ip}:{args.head_port}...")
    client_sock_head = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            client_sock_head.connect((args.head_ip, args.head_port))
            print(f"[{node_name}] Connected to Head Node!")
            break
        except ConnectionRefusedError:
            time.sleep(1)
            print(f"[{node_name}] Retrying connection to Head Node...")

    # 2. Init Resources
    context = initResource(args.device)
    
    # 3. Init Config & KVCache
    user_config = InferenceConfig(
        hf_model_dir=args.hf_dir,
        device=args.device,
        model=args.model,
        max_cache_size=args.kv_size
    )
    user_config.n_layer = args.n_layer
    user_config.max_input_len = 1
    
    kv_cache = KVCache.create(user_config)
    print(f"[{node_name}] KVCache initialized for {args.n_layer} layers.")

    # 4. Load Model
    print(f"[{node_name}] Loading model: {args.model}")
    model = ACLModel(args.model, context=context, mode=user_config.acl_mode)
    
    # 5. Inference Loop
    print(f"[{node_name}] Ready for inference loop...")
    step = 0
    while True:
        # Receive [hidden, input_ids, mask, pos_ids, kv0]
        msg = recv_msg(conn_prev)
        if msg is None:
            print(f"[{node_name}] Connection closed by prev node.")
            break
        
        # Unpack
        hidden_states = msg[0]
        input_ids = msg[1]
        mask = msg[2]
        pos_ids = msg[3]
        kv0_past = msg[4]
        
        
            
        log_debug(f"debug_{node_name.replace(' ', '')}.log", f"Step {step}: Received Msg", hidden_states)

        seq_len = input_ids.shape[1]
        
        # Local KV
        local_cache_ret = kv_cache.getInputs(user_config.max_length)
        local_kvs_list = local_cache_ret[0]
        
        # Prepare ACL Inputs
        # Signature: [hidden, input_ids, mask, pos_ids, kv0_rope, local_kvs...]
        acl_inputs = [hidden_states, input_ids, mask, pos_ids, kv0_past]
        acl_inputs.extend(local_kvs_list)
        
        log_debug(f"debug_{node_name.replace(' ', '')}.log", f"Step {step}: OM Input Hidden", hidden_states)
        log_debug(f"debug_{node_name.replace(' ', '')}.log", f"Step {step}: OM Input Cache", local_kvs_list[0] if local_kvs_list else [])

        # Inference
        res = model.inference(acl_inputs)
        
        logits = res[0]
        new_kvs = res[1:]
        
        log_debug(f"debug_{node_name.replace(' ', '')}.log", f"Step {step}: Output Logits", logits)

        # Update Local KV
        kv_cache.update(seq_len, new_kvs, None)
        
        # Send logits to Head Node
        send_msg(client_sock_head, logits)
        
        step += 1

    conn_prev.close()
    client_sock_head.close()

if __name__ == "__main__":
    main()

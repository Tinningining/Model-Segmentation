import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from config import InferenceConfig
from inference import LlamaInterface

def main(cli:bool,engine:LlamaInterface):
    while True:
        line = input()
        print(engine.predict(line))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv_size", type=int, default=256)
    parser.add_argument(
        "--engine", type=str, default="acl",
        help="inference backend, onnx or acl"
    )
    parser.add_argument(
        "--sampling", type=str, default="top_k",
        help="sampling method, greedy, top_k or top_p"
    )
    parser.add_argument(
        "--sampling_value",type=float,default=10,
        help="if sampling method is seted to greedy, this argument will be ignored; if top_k, it means value of p; if top_p, it means value of p"
    )
    parser.add_argument(
        "--temperature",type=float,default=0.7,
        help="sampling temperature if sampling method is seted to greedy, this argument will be ignored."
    )
    parser.add_argument(
        "--hf-dir", type=str, default="/root/model/tiny-llama-1.1B", 
        help="path to huggingface model dir"
    )
    parser.add_argument(
        "--model", type=str, default="/root/model/tiny-llama-seq-1-key-256-int8.om", 
        help="path to onnx or om model"
    )
    parser.add_argument(
        "--split-model-dir", type=str, default="", 
        help="directory containing split onnx models (M0..M3)"
    )
    # Network args
    parser.add_argument("--next-ip", type=str, default="127.0.0.1", help="IP of next node")
    parser.add_argument("--next-port", type=int, default=8001, help="Port of next node")
    parser.add_argument("--listen-port", type=int, default=8004, help="Port to listen for return")
    parser.add_argument("--kv_method", type=str, default="sliding-window", help="basic | sliding-window | streamllm | H2O")

    args = parser.parse_args()
    cfg = InferenceConfig(
        hf_model_dir=args.hf_dir,
        model=args.model,
        split_model_dir=args.split_model_dir,
        max_cache_size=args.kv_size,
        kvcache_method=args.kv_method,
        sampling_method=args.sampling,
        sampling_value=args.sampling_value,
        temperature=args.temperature,
        session_type=args.engine,
        next_ip=args.next_ip,
        next_port=args.next_port,
        listen_port=args.listen_port
    )
    engine = LlamaInterface(cfg)
    main(True,engine)
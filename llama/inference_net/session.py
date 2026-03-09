from config import InferenceConfig
from kvcache import KVCache
import numpy as np
import time
import socket
import struct
import pickle
import os

def log_debug(filename, msg, tensor=None):
    try:
        with open(filename, 'a') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
            if tensor is not None:
                if isinstance(tensor, list):
                    f.write(f"  Type: List, Len: {len(tensor)}\n")
                    if len(tensor) > 0:
                        # Print first few elements if they vary
                        for idx, item in enumerate(tensor[:3]):
                             if isinstance(item, np.ndarray):
                                f.write(f"  [{idx}] NDArray Shape: {item.shape}, Dtype: {item.dtype}, Mean: {item.mean():.4f}, Sum: {item.sum():.4f}\n")
                                f.write(f"  [{idx}] Data: {item.flatten()[:10]}\n")
                             else:
                                f.write(f"  [{idx}] Val: {item}\n")
                elif isinstance(tensor, np.ndarray):
                    f.write(f"  Shape: {tensor.shape}, Dtype: {tensor.dtype}, Min: {tensor.min()}, Max: {tensor.max()}, Mean: {tensor.mean():.4f}, Sum: {tensor.sum():.4f}\n")
                    f.write(f"  Data (flat[:10]): {tensor.flatten()[:10]}\n")
                else:
                    f.write(f"  Type: {type(tensor)}, Val: {tensor}\n")
    except Exception as e:
        print(f"Log Error: {e}")

def send_msg(sock, msg):
    # msg is a list of numpy arrays or other serializable objects
    # Use pickle for simplicity, or manual packing for performance
    # For MVP, pickle is fine. For high perf, use raw buffers.
    data = pickle.dumps(msg)
    # Send length first
    sock.sendall(struct.pack('>I', len(data)))
    sock.sendall(data)

def recv_msg(sock):
    # Read length
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read data
    data = recvall(sock, msglen)
    return pickle.loads(data)

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

class Session:
	def __init__(self,config:InferenceConfig) -> None:
		self.kvCache = KVCache.create(config)
		self.max_len = config.max_input_len
		self.step_count = 0

	def run(self,input_ids:np.ndarray):
		pass
	
	@staticmethod
	def fromConfig(config:InferenceConfig) -> 'Session':
		if config.session_type == "onnx":
			return OnnxSession(config)
		elif config.session_type=='acl':
			return AclSession(config)
		elif config.session_type == 'net':
			return NetSession(config)
		else:
			return None
	
	def reset(self):
		self.kvCache.reset()
		self.step_count = 0

	def rollback(self,seq_len):
		self.kvCache.rollback(seq_len)

	def evict(self,space_need):
		self.kvCache.evict(space_need)

class OnnxSession(Session):
	def __init__(self,config:InferenceConfig)->None:
		super().__init__(config)
		import onnxruntime
		import os
		options = onnxruntime.SessionOptions()
		
		# 判断是否使用切分模型 (4 parts: m0=embed, m1, m2, m3=head)
		self.is_split = False
		if config.split_model_dir and os.path.isdir(config.split_model_dir):
			self.is_split = True
			self.models = {}
			model_files = {
				"m0": "llama_m0_embed_layers_0_4.onnx",
				"m1": "llama_m1_layers_5_10.onnx",
				"m2": "llama_m2_layers_11_16.onnx",
				"m3": "llama_m3_layers_17_21_lmhead.onnx"
			}

			providers = [
				"DmlExecutionProvider",
				"CUDAExecutionProvider",
				"CPUExecutionProvider",
			]
			
			print(f"Loading split models from {config.split_model_dir}...")
			for key, filename in model_files.items():
				path = os.path.join(config.split_model_dir, filename)
				if not os.path.exists(path):
					# Fallback or error? Let's check if user named them differently.
					# For now raise error.
					raise FileNotFoundError(f"Split model file not found: {path} (Expected 4-part split)")
				print(f"Loading {key}: {filename}")
				self.models[key] = onnxruntime.InferenceSession(
					path,
					sess_options=options,
					providers=providers,
				)
			
			# 定义每段模型的 layer indices
			# M0: 0-4 (5 layers) - handled separately
			# M1: 5-10 (6 layers)
			# M2: 11-16 (6 layers)
			# M3: 17-21 (5 layers)
			self.layer_ranges = {
				"m1": range(5, 11),
				"m2": range(11, 17),
				"m3": range(17, 22)
			}
			
		else:
			self.llm_session = onnxruntime.InferenceSession(
				config.model,
				sess_options=options,
				providers=[
					"DmlExecutionProvider",
					"CUDAExecutionProvider",
					"CPUExecutionProvider",
				],
			)

	def run(self,input_ids:np.ndarray):
		seq_len=input_ids.shape[-1]
		l,r,result = 0,self.max_len,None
		while l < seq_len:
			r = min(seq_len,r)
			cache,mask,pos_ids = self.kvCache.getInputs(r-l)
			
			if self.is_split:
				# ===== Split Inference =====
				
				# M0: Embed + Layers 0-4
				m0_sess = self.models["m0"]
				m0_inp_names = [x.name for x in m0_sess.get_inputs()]
				m0_inputs = {}
				
				if "input_ids" in m0_inp_names: m0_inputs["input_ids"] = input_ids[:, l:r]
				if "position_ids" in m0_inp_names: m0_inputs["position_ids"] = pos_ids
				
				# M0 (layers 0-4) 必然需要自己的 KV cache (0-4)
				current_layer_range = range(0, 5) # M0 layers
				for i in current_layer_range:
					name = f"past_key_values_{i}"
					if name in m0_inp_names:
						m0_inputs[name] = cache[i]

				# 显式检查 attention_mask，如果图里需要就给
				# 注意：如果 mask 机制和 KV 一样，那这里可能已经没问题了
				# 但保险起见，显式给
				if "attention_mask" in m0_inp_names: m0_inputs["attention_mask"] = mask
				
				m0_out = m0_sess.run(None, m0_inputs)
				hidden_states = m0_out[0]
				
				# 收集 M0 的输出 KV (present_0..4)
				m0_kvs = m0_out[1:] 
				
				all_new_kv = []
				all_new_kv.extend(m0_kvs)
				
				# Helper to run layer block
				def run_block(model_key, current_hidden):
					sess = self.models[model_key]
					inp_names = [x.name for x in sess.get_inputs()]
					inputs = {}
					
					# 1. Hidden State Input (通常是第一个，但也可能乱序，最好用名字匹配)
					# 在 export 时我们用的是特定的输出名作为输入名，例如 "/model/layers.4/Add_1_output_0"
					# 但在这里我们需要动态匹配，或者假设第一个 Input 就是 hidden_states
					# 比较稳妥的是遍历 input names，排除掉已知名称(mask, pos, kvs)
					known_names = {"attention_mask", "position_ids", "input_ids", "past_key_values_0"}
					hidden_name = None
					for name in inp_names:
						if name not in known_names and not name.startswith("past_key_values_"):
							hidden_name = name
							break
					
					if hidden_name:
						inputs[hidden_name] = current_hidden
					else:
						# Fallback: take the first one
						inputs[inp_names[0]] = current_hidden

					# 2. Add common inputs
					if "attention_mask" in inp_names: inputs["attention_mask"] = mask
					if "position_ids" in inp_names: inputs["position_ids"] = pos_ids
					if "input_ids" in inp_names: inputs["input_ids"] = input_ids[:, l:r]
					
					# 3. Add past_key_values_0 dependency (if needed)
					if "past_key_values_0" in inp_names:
						# 即使当前 Block 不包含第0层，如果因为 RoPE 依赖导致需要第0层，也要给
						inputs["past_key_values_0"] = cache[0]

					# 4. Add KV caches for this block
					indices = self.layer_ranges[model_key]
					for i in indices:
						name = f"past_key_values_{i}"
						if name in inp_names:
							inputs[name] = cache[i]
					
					outs = sess.run(None, inputs)
					return outs[0], outs[1:] # hidden, [kvs...]

				# M1
				hidden_states, m1_kvs = run_block("m1", hidden_states)
				all_new_kv.extend(m1_kvs)
				
				# M2
				hidden_states, m2_kvs = run_block("m2", hidden_states)
				all_new_kv.extend(m2_kvs)
				
				# M3: Head (includes layers 17-21)
				m3_sess = self.models["m3"]
				m3_inp_names = [x.name for x in m3_sess.get_inputs()]
				m3_inputs = {}
				
				# 1. Hidden State
				known_names = {"attention_mask", "position_ids", "input_ids", "past_key_values_0"}
				hidden_name = None
				for name in m3_inp_names:
					if name not in known_names and not name.startswith("past_key_values_"):
						hidden_name = name
						break
				if hidden_name:
					m3_inputs[hidden_name] = hidden_states
				else:
					# Fallback
					m3_inputs[m3_inp_names[0]] = hidden_states
					
				# 2. Common Inputs
				if "attention_mask" in m3_inp_names: m3_inputs["attention_mask"] = mask
				if "position_ids" in m3_inp_names: m3_inputs["position_ids"] = pos_ids
				if "input_ids" in m3_inp_names: m3_inputs["input_ids"] = input_ids[:, l:r]
				
				# 3. RoPE dependency
				if "past_key_values_0" in m3_inp_names:
					m3_inputs["past_key_values_0"] = cache[0]
					
				# 4. KV Caches for 17-21
				for i in self.layer_ranges["m3"]:
					name = f"past_key_values_{i}"
					if name in m3_inp_names:
						m3_inputs[name] = cache[i]
				
				m3_out = m3_sess.run(None, m3_inputs)
				logits = m3_out[0]
				
				# Collect M3 KV outputs
				if len(m3_out) > 1:
					all_new_kv.extend(m3_out[1:])

				# Reconstruct result
				# result = [logits, *all_new_kv, attn_scores=None]
				present_key_values_list = all_new_kv
				attn_scores = None
				
				self.kvCache.update(r-l, present_key_values_list, attn_scores)
				
			else:
				# ===== Single Model =====
				inputs = {
					"input_ids": input_ids[:,l:r],
					"attention_mask": mask,
					"position_ids": pos_ids,
				}
				for i, layer_cache in enumerate(cache):
					inputs[f"past_key_values_{i}"] = layer_cache

				result = self.llm_session.run(None, inputs)
				
				logits = result[0]
				present_key_values_list = result[1:-1] 
				attn_scores = result[-1]

				self.kvCache.update(r-l, present_key_values_list, attn_scores)
				
			l , r = l+self.max_len , r + self.max_len
		
		# Return logits for the last step (compat with outside loop)
		# Note: Original code returns 'result', which was a list.
		# Outside expects engine.predict -> session.run -> logits?
		# No, inference.py accesses `logits = self.session.run(input_ids)[0]`.
		# So we should return [logits, ...]
		if self.is_split:
			return [logits] # Minimal return for inference.py
		else:
			return result

class AclSession(Session):
	context = None
	def __init__(self,config:InferenceConfig)->None:
		super().__init__(config)
		from engine import ACLModel,initResource
		self.context = initResource(config.device)
		
		import os
		self.is_split = False
		if config.split_model_dir and os.path.isdir(config.split_model_dir):
			self.is_split = True
			self.models = {}
			model_files = {
				"m0": "llama_m0_embed_layers_0_4",
				"m1": "llama_m1_layers_5_10",
				"m2": "llama_m2_layers_11_16",
				"m3": "llama_m3_layers_17_21_lmhead"
			}
			print(f"Loading split OM models from {config.split_model_dir}...")
			for key, filename in model_files.items():
				# Append .om if usually needed or check file
				path = os.path.join(config.split_model_dir, filename)
				if not path.endswith(".om"):
					# Auto-append .om if not present, assuming standard ATC output
					if os.path.exists(path + ".om"):
						path += ".om"
				
				if not os.path.exists(path):
					raise FileNotFoundError(f"Split OM model file not found: {path} (Expected 4-part split: {filename})")
				print(f"Loading {key}: {path}")
				self.models[key] = ACLModel(path, context=self.context, mode=config.acl_mode)
			
			self.layer_ranges = {
				"m1": range(5, 11),
				"m2": range(11, 17),
				"m3": range(17, 22)
			}
			# Ensure buffer is 1x1 for input_ids as per ATC config
			self.input_ids = np.zeros((1, 1), dtype=np.int64)
		else:
			self.model = ACLModel(config.model,context=self.context,mode=config.acl_mode)
			self.input_ids = np.zeros((1,self.max_len),dtype=np.int64)
			if config.acl_mode == 'rc':
				# Update logic for getting buffer pointers if needed
				# self.input_ids,_,_,self.kvCache.kvCache = self.model.getInputs()
				pass 

	def run(self,input_ids:np.ndarray):
		seq_len=input_ids.shape[-1]
		l,r,result = 0,self.max_len,None
		while l < seq_len:
			r = min(seq_len,r)

			if self.is_split:
				# Copy current token(s) to input buffer.
				curr_len = r - l
				# if curr_len > self.input_ids.shape[1]: ...
				self.input_ids[:, :curr_len] = input_ids[:, l:r]
				
				cache,mask,pos_ids = self.kvCache.getInputs(self.max_len)
				
				log_debug("debug_acl.log", f"Step {self.step_count}: Start M0", self.input_ids[:, :curr_len])

				# M0: Embed + Layers 0-4
				m0_inputs = [self.input_ids, mask, pos_ids]
				for i in range(5):
					m0_inputs.append(cache[i])
				
				log_debug("debug_acl.log", f"Step {self.step_count}: M0 Input Cache[0]", cache[0])
				log_debug("debug_acl.log", f"Step {self.step_count}: M0 Input Ids", self.input_ids[:, :curr_len])
					
				m0_res = self.models["m0"].inference(m0_inputs)
				hidden_states = m0_res[0]

				log_debug("debug_acl.log", f"Step {self.step_count}: M0 Output Hidden", hidden_states)
				
				# M0 Outputs: [hidden, kv0_new, ... kv4_new]
				m0_kvs = m0_res[1:]
				all_new_kv = []
				all_new_kv.extend(m0_kvs)
				
				def run_acl_block(model_key, current_hidden):
					acl_inputs = [current_hidden, self.input_ids, mask, pos_ids]
					acl_inputs.append(cache[0]) # RoPE
					indices = self.layer_ranges[model_key]
					for i in indices:
						acl_inputs.append(cache[i])
					
					log_debug("debug_acl.log", f"Step {self.step_count}: {model_key} Input Hidden", current_hidden)
					log_debug("debug_acl.log", f"Step {self.step_count}: {model_key} Input Cache[{indices[0]}]", cache[indices[0]])

					res = self.models[model_key].inference(acl_inputs)
					return res[0], res[1:]

				# M1
				hidden_states, m1_kvs = run_acl_block("m1", hidden_states)
				log_debug("debug_acl.log", f"Step {self.step_count}: M1 Output Hidden", hidden_states)
				all_new_kv.extend(m1_kvs)
				
				# M2
				hidden_states, m2_kvs = run_acl_block("m2", hidden_states)
				log_debug("debug_acl.log", f"Step {self.step_count}: M2 Output Hidden", hidden_states)
				all_new_kv.extend(m2_kvs)
				
				# M3
				hidden_states, m3_kvs = run_acl_block("m3", hidden_states)
				logits = hidden_states
				log_debug("debug_acl.log", f"Step {self.step_count}: M3 Output Logits", logits)
				all_new_kv.extend(m3_kvs)
				
				self.kvCache.update(r-l, all_new_kv, None)
				result = [logits]
			else:

				self.input_ids[:,:r-l] = input_ids[:,l:r]
				cache,mask,pos_ids = self.kvCache.getInputs(self.max_len)
				
				# For single big model OM:
				# inputs = [input_ids, mask, pos_ids, kv_0, kv_1... kv_n]
				# Flatten cache list
				acl_inputs = [self.input_ids, mask, pos_ids]
				acl_inputs.extend(cache)
				
				result:List[np.ndarray] = self.model.inference(acl_inputs)
				# result: [logits, new_kv_0, new_kv_1..., attn_scores]
				logits = result[0]
				new_kvs = result[1:-1]
				attn = result[-1]
				
				self.kvCache.update(r-l, new_kvs, attn)
			
			l , r = l+self.max_len , r + self.max_len
			self.step_count += 1
		return result

class NetSession(Session):
    def __init__(self, config:InferenceConfig) -> None:
        # Node 1 needs only 5 layers (0-4) for KV Cache
        config.n_layer = 5
        super().__init__(config)
        
        from engine import ACLModel, initResource
        self.context = initResource(config.device)
        self.model_path = os.path.join(config.split_model_dir, "llama_m0_embed_layers_0_4.om")
        if not os.path.exists(self.model_path):
             self.model_path = os.path.join(config.split_model_dir, "llama_m0_embed_layers_0_4")

        print(f"[NetSession] Loading M0 model from {self.model_path}")
        self.model = ACLModel(self.model_path, context=self.context, mode=config.acl_mode)
        
        # Network Setup
        self.next_ip = config.next_ip 
        self.next_port = config.next_port
        self.listen_port = config.listen_port
        
        # 1. Start Server for return path (from Node 4)
        print(f"[NetSession] Binding server on port {self.listen_port}...")
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(('0.0.0.0', self.listen_port))
        self.server_sock.listen(1)
        
        # 2. Connect to Next Node (Node 2)
        print(f"[NetSession] Connecting to Node 2 at {self.next_ip}:{self.next_port}...")
        self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.client_sock.connect((self.next_ip, self.next_port))
                print(f"[NetSession] Connected to Node 2")
                break
            except ConnectionRefusedError:
                print(f"[NetSession] Waiting for Node 2...")
                time.sleep(1)
        
        # 3. Accept connection from Node 4
        print(f"[NetSession] Waiting for connection from Node 4...")
        self.prev_conn, addr = self.server_sock.accept()
        print(f"[NetSession] Accepted connection from Node 4: {addr}")

        self.input_ids = np.zeros((1, self.max_len), dtype=np.int64)

    def run(self, input_ids: np.ndarray):
        seq_len = input_ids.shape[-1]
        l, r = 0, self.max_len
        result = None
        
        while l < seq_len:
            r = min(seq_len, r)
            current_batch_len = r - l
            
            # Prepare Input Buffer
            self.input_ids[:, :current_batch_len] = input_ids[:, l:r]
            
            # Get KV Cache inputs (layers 0-4)
            cache, mask, pos_ids = self.kvCache.getInputs(self.max_len)
            
            log_debug("debug_node1.log", f"Step {self.step_count}: Start NetSession Node1", self.input_ids[:, :current_batch_len])

            # Prepare M0 Inputs: [input_ids, mask, pos_ids, kv0, kv1, kv2, kv3, kv4]
            acl_inputs = [self.input_ids[:, :current_batch_len], mask, pos_ids]
            acl_inputs.extend(cache)
            
            log_debug("debug_node1.log", f"Step {self.step_count}: M0 Input Cache[0]", cache[0])
            
            # Run M0
            m0_res = self.model.inference(acl_inputs)
            # Outputs: [hidden_states, new_kv0, new_kv1, new_kv2, new_kv3, new_kv4]
            hidden_states = m0_res[0]
            new_kvs = m0_res[1:]
            
            log_debug("debug_node1.log", f"Step {self.step_count}: M0 Output Hidden", hidden_states)

            # Update Local KV Cache
            self.kvCache.update(current_batch_len, new_kvs, None)
            
            # Send to Node 2
            # Needs: [hidden_states, input_ids, mask, pos_ids, past_key_values_0]
            msg = [hidden_states, self.input_ids[:, :current_batch_len], mask, pos_ids, cache[0]]
            
            log_debug("debug_node1.log", f"Step {self.step_count}: Sending Msg to Node 2", [hidden_states, self.input_ids[:, :current_batch_len]])

            send_msg(self.client_sock, msg)
            
            # Wait for Node 4 result
            logits = recv_msg(self.prev_conn)
            
            log_debug("debug_node1.log", f"Step {self.step_count}: Received Logits from Node 4", logits)

            if isinstance(logits, list):
                result = logits
            else:
                result = [logits]
            
            l += self.max_len
            r += self.max_len
            self.step_count += 1
            
        return result

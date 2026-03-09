import numpy as np
import os
import time
import psutil
import sys
from typing import Any, Generator, List,Tuple,Dict
from threading import Lock
from session import Session, log_debug
from config import InferenceConfig
import numpy as np
import time
import os
import psutil

class LlamaInterface:
    def __init__(self,config:InferenceConfig) -> None:
        # 记录启动开始时间
        init_start_time = time.time()
        
        self.max_length = config.max_length
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(config.tokenizer)
        self.sampling_method=config.sampling_method
        self.sampling_value = config.sampling_value
        self.temperature=config.temperature
        self.session=Session.fromConfig(config)
        self.prompt=config.prompt
        self.state:dict[str,Any] = {"code":200,"isEnd":False,"message":""}        
        self.first=True
        self.stop_mp = {"[|Human|]":6,"[|AI|]":5,"<|assistant|>":6,"<|user|>":5,"<|system|>":5}
        self.stop_words = ["<|user|>","<|assistant|>","<|system|>","[|AI|]","[|Human|]"]
        self.model_type = config.model_type
        self.last_output=""
        self.last_speed_info = ""  # 存储速度信息
        self.lock = Lock()
        
        # Log filename
        if config.session_type == "acl":
            self.log_filename = "debug_acl.log"
        elif config.session_type == "net":
            # For master node (inference.py runs on master)
            self.log_filename = "debug_node1.log"
        else:
            self.log_filename = "inference.log"
            
        self.reset()
        
        # 计算启动时间
        init_end_time = time.time()
        self.init_time = init_end_time - init_start_time
        
        # 获取内存占用（RSS - Resident Set Size，实际物理内存占用）
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        self.memory_mb = memory_info.rss / 1024 / 1024  # 转换为 MB
        
        print(f"init success [启动时间: {self.init_time:.2f}s, 内存占用: {self.memory_mb:.2f} MB]")

    def generate_cache(self,prompt:str):
        if len(prompt) == 0 :
            return
        input_ids = np.asarray(self.encode(prompt,add_bos_token=self.first),dtype=np.int64).reshape(1,-1)
        self.first=False
        logits = self.session.run(input_ids)[0]
        return self.sample_logits(logits[0][-1:],self.sampling_method,self.sampling_value,self.temperature),logits

    def sample_logits(
        self,
        logits: np.ndarray,
        sampling_method: str = "greedy",
        sampling_value: float = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        if temperature == 0 or sampling_method == "greedy":
            next_token = np.argmax(logits, axis=-1).astype(np.int64)

        elif sampling_method == "top_k" or sampling_method == "top_p":
            assert sampling_value is not None
            logits = logits.astype(np.float32)
            logits /= temperature
            probs = np.exp(logits) / np.sum(np.exp(logits))
            sorted_probs = np.sort(probs)[:, ::-1]
            sorted_indices = np.argsort(probs)[:, ::-1]

            if sampling_method == "top_k":
                index_of_interest = int(sampling_value)
            elif sampling_method == "top_p":
                p = sampling_value
                cumulative_probs = np.cumsum(sorted_probs, axis=-1)
                for index_of_interest, cumulative_prob in enumerate(
                    cumulative_probs[0]
                ):
                    if cumulative_prob > p:
                        break

            probs_of_interest = sorted_probs[:, : index_of_interest + 1]
            indices_of_interest = sorted_indices[:, : index_of_interest + 1]
            probs_of_interest /= np.sum(probs_of_interest)
            next_token = np.array(
                [np.random.choice(indices_of_interest[0], p=probs_of_interest[0])]
            )
        else:
            raise Exception(f"Unknown sampling method {sampling_method}")

        return next_token

    
    def format_last_output(self):
        if len(self.last_output) == 0:
            return 
        text_format = self.apply_chat_template([{"role":"assistant","content":self.last_output}])
        self.generate_cache(text_format[len(self.last_output):])
        self.last_output = ""
    
    def predict(self, text):
        with self.lock:
            self.state['isEnd'],self.state['message'] = False,""        
        if text == "":
            return
        self.format_last_output()
        text = self.apply_chat_template([{"role":"user","content":text}])
        input_ids = self.encode(text,add_bos_token=self.first)
        input_ids = np.asarray(input_ids,dtype=np.int64).reshape(1,-1)
        
        log_debug(self.log_filename, f"Predict Start. Text: {text}", input_ids)
        
        self.first,ids_list = False,[]
        
        # 记录开始时间
        start_time = time.time()
        
        for i in range(self.max_length):
            # 移除 step 参数
            logits = self.session.run(input_ids)[0]
            
            input_ids = self.sample_logits(logits[0][-1:], self.sampling_method, self.sampling_value, self.temperature)
            input_ids = input_ids.reshape(1, -1)
            
            # log_debug("inference.log", f"Step {i} Result", input_ids)

            if input_ids[0] == self.tokenizer.eos_token_id:
                self.session.rollback(1) 
                break
            ids_list.append(input_ids[0].item())
            text_out = self.tokenizer.decode(ids_list)
            stop_word = is_stop_word_or_prefix(text_out,self.stop_words)
            if stop_word != "":
                ids_list = ids_list[:-self.stop_mp[stop_word]]
                self.session.rollback(self.stop_mp[stop_word]) 
                break
            if i%3 == 0:
                with self.lock:
                    self.state['message']=text_out
        
        # 计算生成时间和速度（基于 token 数量，不是字符数）
        end_time = time.time()
        elapsed_time = end_time - start_time
        num_tokens = len(ids_list)  # token 数量
        tokens_per_second = num_tokens / elapsed_time if elapsed_time > 0 else 0
        
        self.last_output = self.tokenizer.decode(ids_list)
        # 保存速度信息（单独存储，不影响原始输出）
        self.last_speed_info = f"[生成速度: {tokens_per_second:.2f} tokens/s, 总tokens: {num_tokens}, 耗时: {elapsed_time:.2f}s]"
        
        with self.lock:
            self.state['message'],self.state['isEnd'] = self.last_output,True
        
        return self.last_output

    def reset(self):
        self.first = True
        self.last_output = ""
        self.session.reset()
        self.generate_cache(self.apply_chat_template(self.prompt))
        
    def getState(self):
        with self.lock:
            return self.state.copy()

    def apply_chat_template(self,messages:List[Dict[str,str]]) -> str:
        text = ""
        if self.model_type == "llama-2-7b":
            for message in messages:
                if message["role"] == "user":
                    text += f'[|Human|]\n{message["content"]}\n[|AI|]'
                elif message["role"] == "system":
                    text += f'[|System|]\n{message["content"]}\n'
                else:
                    text += f'{message["content"]}\n'
        elif self.model_type == "tiny-llama":
            for message in messages:
                if message["role"] == "user":
                    text += f'<|user|>\n{message["content"]}</s>\n<|assistant|>'
                elif message["role"] == "system":
                    text += f'<|system|>\n{message["content"]}</s>\n'
                else:
                    text += f'{message["content"]}</s>\n'
        return text
    
    def encode(self,text,add_bos_token=False):
        self.tokenizer.add_bos_token = add_bos_token
        return self.tokenizer.encode(text)

def is_stop_word_or_prefix(s: str, stop_words: list) -> int:
    for stop_word in stop_words:
        if s.endswith(stop_word):
            return stop_word
    return ""
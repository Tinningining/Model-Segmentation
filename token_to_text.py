from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    r"D:\qwen_split\qwen3_1.7b",
    trust_remote_code=True
)

ids = [
3837,100630,116509,104949,100166,
5373,104034,100178,5373,113272,
100178,5373,116541,49567,1773,
40666,100,104949,105166,99257,
105318,100630,104949
]

text = tokenizer.decode(ids)

print(text)
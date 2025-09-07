# gpt2.prompt

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import TextStreamer
import os,sys,re,random,json,glob 
import multiprocessing
import time




#######################################################################################################################
# MODELS
#######################################################################################################################

with open(sys.argv[1],"r") as jf:
    system = json.load(jf)


# 1️⃣ Specify your base model and the path to your saved LoRA adapter
ADAPTER_PATH = system["OUTPUT_DIR"]
use_4bit = system["LOAD_4BIT"] == "True"
use_8bit = system["LOAD_8BIT"] == "True"
max_seq = 1024

model_name = ADAPTER_PATH

use_peft = False

if(use_peft):  
    model = AutoModelForCausalLM.from_pretrained(
        "llama.2.7b",
        load_in_4bit = use_4bit,
        load_in_8bit = use_8bit,
        device_map="auto",
        dtype="bfloat16")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    # (Important) Make sure the intended adapter is active
    print("Available adapters:", list(model.peft_config.keys()))
    model.set_adapter(list(model.peft_config.keys())[0])  # e.g. "default"

else:
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",dtype="bfloat16")
    model.eval()    

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


# Custom stopping criteria to stop when the <|endoftext|> token is generated
class StopOnEndOfText(StoppingCriteria):
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if the last token generated is the eos_token_id
        return input_ids[0, -1] == self.eos_token_id

# Create an instance of the stopping criteria with the model's EOS token
eos_token_id = tokenizer.eos_token
EOT = eos_token_id
stopping_criteria = StoppingCriteriaList([StopOnEndOfText(eos_token_id)])

textstreamer = TextStreamer(tokenizer, skip_prompt = True)
temperature = 0.9
top_p = 0.9
top_k = 20
count = 1

# 4️⃣ Define generation function
def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")    
    out = model.generate(
        inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        streamer = textstreamer,
        do_sample = True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,        
        pad_token_id=tokenizer.eos_token_id,
        max_length=max_seq,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,                   # mild anti-repeat
        remove_invalid_values=True,
        stopping_criteria=stopping_criteria
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


while 1:
    print("### SYSTEM PROMPT ###")
    print("Press CTRL+D to send.")
    p = sys.stdin.read().strip()
    
    temp = []
    for line in p.split("\n"):
        if("~count:" in line):
            count = int(p.split("~count:")[1].strip())
            continue 
        elif("~topk:" in line):
            top_k = int(p.split("~topk:")[1].strip())
            continue 
        elif("~topp:" in line):
            top_p = float(p.split("~topp:")[1].strip())
            continue 
        elif("~temp:" in line):
            temperature = float(p.split("~temp:")[1].strip())
            continue 
        else:
            temp.append(line)
    p = '\n'.join(temp)
    p = p.lower()
    
    output = generate(f"{p}")
```

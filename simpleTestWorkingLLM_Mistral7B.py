# Testing Mistral-7B-v0.1 model with 4-bit quantization
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
from huggingface_hub import login

def check_system_capabilities():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def prepare_model():
    # Login to Hugging Face
    token = "hf_JwqNErcgWgpowmcGwuDNsxgNiPSSCRIGLKZ" #change the token to a valid one
    login(token)
    
    model_name = "mistralai/Mistral-7B-v0.1"
    print(f"Loading model: {model_name}")
    
    # Configure 4-bit quantization with CPU offload enabled
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offload
    )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=True
    )
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        device_map="auto",  # Transformers handle device mapping
        quantization_config=quantization_config,
        trust_remote_code=True,
        offload_folder="offload"  # Folder for offloaded weights
    )
    return model, tokenizer

def test_model(model, tokenizer, text):
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.3,
            max_length=2048,
            min_length=100,
            top_p=0.85,
            top_k=30,
            repetition_penalty=1.5,
            no_repeat_ngram_size=4,
            length_penalty=1.0,
            num_beams=4,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    check_system_capabilities()
    model, tokenizer = prepare_model()
    
    test_text = "What are the data types in python?"
    result = test_model(model, tokenizer, test_text)
    
    print("\nResponse:")
    print("-" * 50)
    print(result)

if __name__ == "__main__":
    main()

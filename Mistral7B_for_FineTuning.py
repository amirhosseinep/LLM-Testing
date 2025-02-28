# Because of lack of GPU, the training is not working

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def check_system_capabilities():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def format_instruction(example):
    """Format the instruction-input-output example into a prompt"""
    if example["input"]:
        prompt = f"### Instruction: {example['instruction']}\n### Input: {example['input']}\n### Response: {example['output']}"
    else:
        prompt = f"### Instruction: {example['instruction']}\n### Response: {example['output']}"
    return prompt

def prepare_training_data(tokenizer):
    """
    Prepare and tokenize the training data
    """
    training_data = [
        {
            "instruction": "Explain Python data types",
            "input": "",
            "output": "Python has several built-in data types including: integers (int) for whole numbers, floating-point (float) for decimal numbers, strings (str) for text, lists for ordered sequences, dictionaries (dict) for key-value pairs, tuples for immutable sequences, and booleans (bool) for True/False values."
        },
        {
            "instruction": "What is a Python list?",
            "input": "",
            "output": "A Python list is an ordered, mutable sequence that can store multiple items of different types. Lists are created using square brackets [] and can contain numbers, strings, or other objects. They support operations like indexing, slicing, and various built-in methods for manipulation."
        }
    ]
    
    # Convert to Dataset
    dataset = Dataset.from_list(training_data)
    
    # Format the prompts
    def preprocess_function(examples):
        prompts = [format_instruction({"instruction": instr, "input": inp, "output": out}) 
                  for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"])]
        
        # Tokenize
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].clone()
        }
    
    # Process the dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def prepare_model():
    token = "hf_JwqNErcgWgpowmcGwuDNsxgNiPSSCRIGLKZ" #change the token to a valid one
    login(token)
    
    model_name = "mistralai/Mistral-7B-v0.1"
    print(f"Loading model: {model_name}")
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    return model, tokenizer

def train_model(model, tokenizer, dataset):
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")

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
    # Check system capabilities
    check_system_capabilities()
    
    # Load model and tokenizer
    model, tokenizer = prepare_model()
    
    # Prepare dataset
    dataset = prepare_training_data(tokenizer)
    
    # Train the model
    train_model(model, tokenizer, dataset)
    
    # Test the trained model
    test_text = "What are the data types in python?"
    result = test_model(model, tokenizer, test_text)
    print("\nTest Result:")
    print("-" * 50)
    print(result)

if __name__ == "__main__":
    main()

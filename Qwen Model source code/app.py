import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available
from peft import PeftModel
import torch
import os

# --- Configuration ---
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_MODEL_PATH = "/data/qwen-orwell-finetuned"
DEFAULT_SYSTEM_PROMPT = "You are an expert in creating educational assessments based on George Orwell's literary works, trained to generate high-quality questions."

# --- Load Tokenizer ---
print(f"Attempting to load tokenizer from adapter path: {ADAPTER_MODEL_PATH}")
if os.path.exists(os.path.join(ADAPTER_MODEL_PATH, "tokenizer_config.json")):
    try:
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL_PATH, trust_remote_code=True, local_files_only=True)
        print("Tokenizer loaded successfully from adapter path.")
    except Exception as e:
        print(f"Could not load tokenizer from adapter path {ADAPTER_MODEL_PATH}. Error: {e}")
        print(f"Falling back to loading tokenizer from base model: {BASE_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
else:
    print(f"Tokenizer config not found at {ADAPTER_MODEL_PATH}. Loading tokenizer from base model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.eos_token}")

# --- Quantization Configuration ---
bnb_config_inference = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- Determine Attention Implementation for Inference ---
attn_implementation_to_use_inference = "sdpa"
if is_flash_attn_2_available():
    print("Flash Attention 2 is available, will attempt to use it for inference.")
    attn_implementation_to_use_inference = "flash_attention_2"
else:
    print("Flash Attention 2 is not available for inference, using 'sdpa'.")

# --- Load Base Model and Apply Adapters ---
print(f"Loading base model {BASE_MODEL_NAME} for inference with 4-bit quantization and {attn_implementation_to_use_inference}...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config_inference,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation=attn_implementation_to_use_inference
)
print(f"Base model loaded using attention implementation: {base_model.config._attn_implementation}")

adapter_config_file_path = os.path.join(ADAPTER_MODEL_PATH, "adapter_config.json")
if os.path.exists(ADAPTER_MODEL_PATH) and os.path.exists(adapter_config_file_path):
    print(f"Loading LoRA adapters from {ADAPTER_MODEL_PATH}...")
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_PATH)
        model = model.merge_and_unload()
        print("LoRA adapters loaded and merged successfully.")
    except Exception as e:
        print(f"Failed to load or merge LoRA adapters from {ADAPTER_MODEL_PATH}. Error: {e}")
        print("Using the base model without fine-tuned adapters as a fallback.")
        model = base_model
else:
    if not os.path.exists(ADAPTER_MODEL_PATH): print(f"Adapter directory {ADAPTER_MODEL_PATH} not found.")
    else: print(f"Adapter config {adapter_config_file_path} not found in {ADAPTER_MODEL_PATH}.")
    print("Using base model without fine-tuning.")
    model = base_model

model.eval()
print("Model is ready for inference.")

# --- Gradio Chat Interface ---
def generate_response(prompt, history, system_prompt_input):
    current_system_prompt = system_prompt_input if system_prompt_input and system_prompt_input.strip() else DEFAULT_SYSTEM_PROMPT
    messages = [{"role": "system", "content": current_system_prompt}]
    for user_msg, assistant_msg in history: messages.append({"role": "user", "content": user_msg}); messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": prompt})
    try:
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(input_ids=tokenized_chat, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
        response_text = tokenizer.decode(outputs[0][tokenized_chat.shape[-1]:], skip_special_tokens=True)
        history.append((prompt, response_text)); return "", history, current_system_prompt
    except Exception as e: print(f"Error: {e}"); import traceback; traceback.print_exc(); history.append((prompt, f"Error: {str(e)}")); return "", history, current_system_prompt

def clear_history_fn(): return "", [], DEFAULT_SYSTEM_PROMPT

def modify_system_prompt_fn(new_system_prompt_input):
    updated_system_prompt = new_system_prompt_input if new_system_prompt_input and new_system_prompt_input.strip() else DEFAULT_SYSTEM_PROMPT
    return updated_system_prompt, [] # Output 1: new value for system_input, Output 2: new value for chatbot (empty list clears it)

print("Starting Gradio demo...")
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Qwen Fine-Tuned: Orwell Literature Assessment Generator")
    with gr.Row():
        with gr.Column(scale=3): chatbot = gr.Chatbot(label="Conversation", height=600); user_input = gr.Textbox(label="Your Prompt", lines=3)
        with gr.Column(scale=1): system_input = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, lines=5, label="System Prompt"); modify_system_btn = gr.Button("🔄 Update System & Clear History"); clear_history_btn = gr.Button("🧹 Clear Chat History")
    with gr.Row(): send_btn = gr.Button("🚀 Send", variant="primary")
    
    send_btn.click(generate_response, [user_input, chatbot, system_input], [user_input, chatbot, system_input])
    user_input.submit(generate_response, [user_input, chatbot, system_input], [user_input, chatbot, system_input])
    clear_history_btn.click(clear_history_fn, [], [user_input, chatbot, system_input])
    modify_system_btn.click(modify_system_prompt_fn, [system_input], [system_input, chatbot]) # Corrected outputs

# Launch with share=True
demo.launch(share=True, debug=True)

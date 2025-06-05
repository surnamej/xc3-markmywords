# train.py
print("<<<<< RUNNING train.py VERSION 2025-05-24_GRPO_CompletionListFix >>>>>")
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from transformers.utils import is_flash_attn_2_available
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
import re
import os
import json
import spacy
from spacy.matcher import PhraseMatcher

try:
    from chromadb import HttpClient
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    print("Warning: chromadb library not found. RAG functionality will be disabled.")
    CHROMA_AVAILABLE = False; HttpClient = None; Settings = None

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_FILE = "train_dataset.json"
OUTPUT_DIR = "/data/model_output/qwen-orwell-finetuned-grpo"

CHROMA_HOST = "34.126.200.250"; CHROMA_PORT = 8000; COLLECTION_NAME = "orwell_books"
NLP_MODEL_LOADED = False
try:
    nlp = spacy.load("en_core_web_sm")
    NLP_MODEL_LOADED = True; print("spaCy 'en_core_web_sm' loaded.")
except OSError:
    print("CRITICAL WARNING: spaCy 'en_core_web_sm' not found. RAG limited."); nlp = None

chroma_client = None; collection = None
if CHROMA_AVAILABLE:
    try:
        chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=Settings(anonymized_telemetry=False))
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        print(f"Connected to ChromaDB and got/created collection '{COLLECTION_NAME}'.")
    except Exception as e_client:
        print(f"Warning: Could not connect/setup ChromaDB: {e_client}. RAG affected.")
        collection = None

# --- RAG Helper Functions (Keep your full versions) ---
BOOK_TITLES = ["Nineteen Eighty-Four", "Animal Farm"] # Keep your full list
phrase_matcher = None
if NLP_MODEL_LOADED:
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(title) for title in BOOK_TITLES]
    phrase_matcher.add("BOOK_TITLE", patterns)

def extract_metadata(prompt_text): # Placeholder, use your actual robust function
    if not NLP_MODEL_LOADED or not phrase_matcher: return ({}, None)
    doc = nlp(prompt_text); name = None; metadata = {}
    matches = phrase_matcher(doc)
    for _, start, end in matches: name = doc[start:end].text.lower(); break
    if name: metadata = {"title": name}
    return metadata, name

def retrieve_context(prompt_text, top_k=1): # Placeholder, use your actual robust function
    if not collection or not NLP_MODEL_LOADED: print("DEBUG RAG: Skipping context retrieval."); return []
    filters, nameBook = extract_metadata(prompt_text)
    query_text = f"Info about {nameBook}" if nameBook else prompt_text
    try:
        results = collection.query(query_texts=[query_text], where=filters if filters else None, n_results=top_k)
        if results and results.get('documents') and results['documents'][0]: return [{"content": d} for d in results['documents'][0]]
    except Exception as e: print(f"Warning: RAG query error: {e}")
    return []


# --- Tokenizer, Model, LoRA ---
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token; print(f"Set pad_token to eos_token.")
print("Setting up BitsAndBytesConfig...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
print(f"Using attention implementation: {attn_implementation}")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, attn_implementation=attn_implementation)
print(f"Base model loaded. Actual attention: {model.config._attn_implementation}")
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- Data Preprocessing for GRPO ---
def format_example_for_grpo(example):
    original_chat_prompt_list = example.get("prompt", []) # This is List[Dict] from your data
    
    actual_completion_string = example.get("result", "Error: No result field in data.")
    if not isinstance(actual_completion_string, str):
        actual_completion_string = "Error: Result field was not a string."

    # 'prompt' for GRPOTrainer should be the List[Dict] chat format
    grpo_prompt_messages = original_chat_prompt_list

    # *** 'completion' MUST ALSO BE List[Dict] for trl.apply_chat_template ***
    grpo_completion_messages = [{"role": "assistant", "content": actual_completion_string}]

    # Ensure grpo_prompt_messages is valid before returning
    if not (isinstance(grpo_prompt_messages, list) and len(grpo_prompt_messages) > 0 and
            all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in grpo_prompt_messages)):
        print(f"Warning: Malformed 'prompt' (chat format) in format_example_for_grpo: {grpo_prompt_messages}")
        return None # Skip this example if prompt is malformed

    return {
        "prompt": grpo_prompt_messages,
        "completion": grpo_completion_messages, # Now List[Dict]
        **example
    }

# --- Reward Functions ---
def binary_reward(value_list):
    if isinstance(value_list, list) and value_list: return 1.0 if str(value_list[0]).lower() == "yes" else -1.0
    return -1.0
def normalize_rating(value_list, max_value=5):
    if isinstance(value_list, list) and value_list:
        try: return float(value_list[0]) / max_value
        except (ValueError, TypeError): return 0.0
    return 0.0
def compute_reward_from_example(example):
    intent = binary_reward(example.get("intent_relevance.responses", ["no"]))
    citation = binary_reward(example.get("citation_support.responses", ["no"]))
    hallucination = binary_reward(example.get("hallucination_check.responses", ["no"]))
    clarity = normalize_rating(example.get("clarity_rating.responses", [0]))
    relevance = normalize_rating(example.get("relevance_rating.responses", [0]))
    overall = normalize_rating(example.get("overall_quality.responses", [0]))
    reward = (0.2*intent + 0.2*citation + 0.2*hallucination + 0.1*clarity + 0.15*relevance + 0.15*overall)
    return reward

def add_reward_for_grpo(example):
    if example is None: return None
    example["reward"] = compute_reward_from_example(example)
    return example

def grpo_reward_function(prompts: list, completions: list, **kwargs) -> list[float]:
    # prompts and completions are now expected to be List[List[Dict]]
    # kwargs should contain the 'reward' column from the original dataset for the batch.
    if "reward" not in kwargs:
        print("Warning: 'reward' key not found in kwargs for grpo_reward_function. Returning default rewards.")
        return [0.0] * len(completions)
    batch_rewards = kwargs["reward"]
    if not isinstance(batch_rewards, list):
        if len(completions) == 1: return [float(batch_rewards)]
        else: return [float(batch_rewards)] * len(completions)
    return [float(r) for r in batch_rewards]


# --- Load and Process Dataset ---
print(f"Loading dataset from {DATA_FILE}...")
if not os.path.exists(DATA_FILE): raise FileNotFoundError(f"Data file {DATA_FILE} not found.")
raw_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
print(f"DEBUG: Number of examples loaded: {len(raw_dataset)}")
if len(raw_dataset) > 0: print(f"DEBUG: First raw example: {raw_dataset[0]}")

dataset_filter_lambda = lambda ex: (ex.get('prompt') is not None and isinstance(ex.get('prompt'), list) and len(ex.get('prompt')) > 0 and ex.get('result') is not None and isinstance(ex.get('result'), str) and all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in ex['prompt']))
filtered_dataset = raw_dataset.filter(dataset_filter_lambda)
print(f"Number of examples after basic filter: {len(filtered_dataset)}")
if len(filtered_dataset) == 0: raise ValueError("No valid data after filtering. Check 'prompt' (list of chat dicts) & 'result' (str).")

processed_dataset = filtered_dataset.map(format_example_for_grpo, desc="Formatting for GRPO")
processed_dataset = processed_dataset.filter(lambda x: x is not None)
processed_dataset = processed_dataset.map(add_reward_for_grpo, desc="Adding rewards")
processed_dataset = processed_dataset.filter(lambda x: x is not None)

columns_to_keep = ["prompt", "completion", "reward"]
for col in columns_to_keep:
    if col not in processed_dataset.column_names: raise ValueError(f"CRITICAL: Column '{col}' missing after processing.")

# Ensure no null/None prompts or completions before passing to trainer
def validate_trainer_input(example):
    if not example["prompt"] or not example["completion"]: return False
    if not (isinstance(example["prompt"], list) and all(isinstance(msg, dict) for msg in example["prompt"])): return False
    if not (isinstance(example["completion"], list) and all(isinstance(msg, dict) for msg in example["completion"])): return False # Check completion is also List[Dict]
    return True
train_dataset_for_grpo = processed_dataset.filter(validate_trainer_input)
train_dataset_for_grpo = train_dataset_for_grpo.select_columns(columns_to_keep)


print(f"Final dataset size for GRPO training: {len(train_dataset_for_grpo)}")
if len(train_dataset_for_grpo) == 0: raise ValueError("Dataset empty after all processing for GRPOTrainer.")
if len(train_dataset_for_grpo) > 0:
    print(f"Example entry for GRPOTrainer (prompt is List[Dict], completion is List[Dict]):")
    print(f"  Prompt: {train_dataset_for_grpo[0]['prompt']}")
    print(f"  Completion: {train_dataset_for_grpo[0]['completion']}") # This will now be a list of dicts
    print(f"  Reward: {train_dataset_for_grpo[0]['reward']}")


if len(train_dataset_for_grpo) > 20: # Or some small number
    print(f"DEBUG: Selecting a subset of 20 examples for quick test.")
    train_dataset_for_grpo = train_dataset_for_grpo.select(range(20))
    
# --- GRPO Training Configuration ---
print("Setting up GRPOConfig...")
grpo_training_args = GRPOConfig(
    output_dir=OUTPUT_DIR, learning_rate=5e-6, optim="paged_adamw_8bit",
    logging_steps=10, per_device_train_batch_size=1, gradient_accumulation_steps=8,
    num_train_epochs=3, bf16=True,
    max_prompt_length=1536, max_completion_length=512, # These lengths apply to tokenized versions of the chat lists
    remove_unused_columns=False,
    report_to="none", save_strategy="steps",
    save_steps=50, save_total_limit=2,
)

# --- Initialize GRPO Trainer ---
print("Initializing GRPOTrainer...")
trainer = GRPOTrainer(
    model=model,
    args=grpo_training_args,
    train_dataset=train_dataset_for_grpo,
    reward_funcs=grpo_reward_function,
    # processing_class=tokenizer # Keep this if GRPOTrainer still needs it for other internal steps
                                # Or remove if it causes issues now that prompt/completion are both chat lists.
                                # The trl.apply_chat_template might not need it if data is already chat lists.
                                # Let's try with it first, as GRPOTrainer might use it for its own generations.
    processing_class=tokenizer
)

# --- Start GRPO Training ---
print("Starting GRPO training...")
try:
    trainer.train()
    print("GRPO training completed successfully.")
    # ... (saving logic) ...
except Exception as e:
    print(f"An error occurred during GRPO training: {e}"); import traceback; traceback.print_exc(); raise
print("train.py script finished.")


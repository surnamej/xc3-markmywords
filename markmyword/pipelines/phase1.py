import os
import re
import json
import sys
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from etl.load_and_preprocess import enrich_book_metadata, split_chapters, prepare_cleaned_json
from utils.constants import RAW_BOOK_DIR, CLEANED_OUTPUT_DIR, ENRICHED_OUTPUT_DIR
RAW_BOOK_DIR = RAW_BOOK_DIR
CLEANED_OUTPUT_DIR = CLEANED_OUTPUT_DIR
ENRICHED_OUTPUT_DIR = ENRICHED_OUTPUT_DIR
def phase1():
    os.makedirs(CLEANED_OUTPUT_DIR, exist_ok=True)

    # Step 1: Collect cleaned base filenames (e.g., "A Clergyman's Daughter-George Orwell")
    cleaned_books = set()
    for cleaned_file in os.listdir(CLEANED_OUTPUT_DIR):
        if cleaned_file.endswith("_clean.json"):
            cleaned_base = cleaned_file.replace("_clean.json", "")
            cleaned_books.add(cleaned_base.strip())

    # Step 2: Process only new .txt files not already cleaned
    for filename in os.listdir(RAW_BOOK_DIR):
        if filename.endswith(".txt"):
            try:
                base_name = filename[:-4].strip()  # Remove .txt
                if base_name in cleaned_books:
                    print(f"✅ Skipping already cleaned: {filename}")
                    continue

                # Process the new book
                filepath = os.path.join(RAW_BOOK_DIR, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    raw_text = f.read()

                title, author = base_name.rsplit("-", 1)
                chapters = split_chapters(raw_text)
                enrich_metadata = enrich_book_metadata(title.strip(), author.strip())
                structured = prepare_cleaned_json(title.strip(), author.strip(), chapters, enrich_metadata)

                output_path = os.path.join(CLEANED_OUTPUT_DIR, f"{title.strip()}-{author.strip()}_clean.json")
                with open(output_path, "w", encoding="utf-8") as out_f:
                    json.dump(structured, out_f, ensure_ascii=False, indent=4)

                print(f"✅ Processed {filename} and saved to {output_path}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

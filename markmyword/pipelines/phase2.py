import os
import re
import json
import sys
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from etl.load_and_preprocess import enrich_chapter, clean_llm_output, truncate_text,safe_enrich
from utils.constants import RAW_BOOK_DIR, CLEANED_OUTPUT_DIR, ENRICHED_OUTPUT_DIR
RAW_BOOK_DIR = RAW_BOOK_DIR
CLEANED_OUTPUT_DIR = CLEANED_OUTPUT_DIR
ENRICHED_OUTPUT_DIR = ENRICHED_OUTPUT_DIR
def phase2():
    os.makedirs(ENRICHED_OUTPUT_DIR, exist_ok=True)

    enriched_books = {
        fname.replace("_enriched.json", "").replace("_", " ").strip()
        for fname in os.listdir(ENRICHED_OUTPUT_DIR)
        if fname.endswith("_enriched.json")
    }

    for fname in os.listdir(CLEANED_OUTPUT_DIR):
        if not fname.endswith("_clean.json"):
            continue

        base_name = fname.replace("_clean.json", "").strip()
        if base_name in enriched_books:
            print(f"✅ Skipping already enriched: {fname}")
            continue

        try:
            with open(os.path.join(CLEANED_OUTPUT_DIR, fname), "r", encoding="utf-8") as f:
                book = json.load(f)

            enriched_chapters = []
            for chapter in book["chapters"]:
                if "subchapters" in chapter:
                    enriched_subs = []
                    for idx, sub in enumerate(chapter["subchapters"]):
                        sub_for_enrich = {
                            "chapter_number": chapter.get("chapter_number", idx),
                            "subchapter": sub.get("subchapter", f"{idx}"),
                            "title": chapter.get("title", f"Chapter {chapter.get('chapter_number', '?')}"),
                            "text": sub["text"]
                        }
                        enriched = safe_enrich(sub_for_enrich, book["book_metadata"])
                        enriched_subs.append({**sub, **enriched})
                    chapter["subchapters"] = enriched_subs
                    enriched_chapters.append(chapter)
                else:
                    enriched = safe_enrich(chapter, book["book_metadata"])
                    enriched_chapters.append({**chapter, **enriched})



            output_path = os.path.join(
                ENRICHED_OUTPUT_DIR,
                f"{book['book_metadata']['title']}-{book['book_metadata']['author']}_enriched.json"
            )

            with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump({
                    "book_metadata": book["book_metadata"],
                    "chapters": enriched_chapters
                }, out_f, ensure_ascii=False, indent=4)

            print(f"✅ Enriched: {fname} -> {output_path}")
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")

# def phase2():
#     os.makedirs(ENRICHED_OUTPUT_DIR, exist_ok=True)

#     # Step 1: Collect enriched book base names
#     enriched_books = set()
#     for enriched_file in os.listdir(ENRICHED_OUTPUT_DIR):
#         if enriched_file.endswith("_enriched.json"):
#             enriched_base = enriched_file.replace("_enriched.json", "").replace("_", " ")
#             enriched_books.add(enriched_base.strip())

#     for filename in os.listdir(CLEANED_OUTPUT_DIR):
#         if filename.endswith("_clean.json"):
#             try:
#                 base_name = filename.replace("_clean.json", "").strip()

#                 if base_name in enriched_books:
#                     print(f"✅ Skipping already enriched: {filename}")
#                     continue

#                 with open(os.path.join(CLEANED_OUTPUT_DIR, filename), "r", encoding="utf-8") as f:
#                     cleaned_book = json.load(f)

#                 enriched_chapters = []
#                 for chapter in cleaned_book["chapters"]:
#                     enriched = safe_enrich(chapter, cleaned_book["book_metadata"])
#                     enriched_chapters.append({**chapter, **enriched})

#                 output_path = os.path.join(
#                     ENRICHED_OUTPUT_DIR,
#                     f"{cleaned_book['book_metadata']['title']}-{cleaned_book['book_metadata']['author']}_enriched.json"
#                 )

#                 with open(output_path, "w", encoding="utf-8") as out_f:
#                     json.dump({
#                         "book_metadata": cleaned_book["book_metadata"],
#                         "chapters": enriched_chapters
#                     }, out_f, ensure_ascii=False, indent=4)

#                 print(f"✅ Enriched: {filename} -> {output_path}")

#             except Exception as e:
#                 print(f"❌ Error processing {filename}: {e}")

#                 #     # Process the new file
#             #     with open(os.path.join(CLEANED_OUTPUT_DIR, filename), "r", encoding="utf-8") as f:
#             #         cleaned_book = json.load(f)

#             #     enriched_chapters = []
#             #     for chapter in cleaned_book['chapters']:
#             #         try:
#             #             enriched_raw = enrich_chapter(chapter, cleaned_book['book_metadata'], 4096)
#             #         except Exception:
#             #             print(f"⚠️ 4096 tokens failed for {chapter['title']}, retrying with 2048...")
#             #             try:
#             #                 enriched_raw = enrich_chapter(chapter, cleaned_book['book_metadata'], 2048)
#             #             except Exception:
#             #                 print(f"pass")
#             #                 enriched_raw = chapter['text']
#             #             except Exception:
#             #                 print(f"⚠️ 2048 tokens also failed for {chapter['title']}, retrying with 1024...")
#             #                 enriched_raw = enrich_chapter(chapter, cleaned_book['book_metadata'], 1024)
#             #         try:
#             #             enriched = json.loads(enriched_raw)
#             #             if isinstance(enriched, str):
#             #                 enriched = json.loads(enriched)
#             #         except json.JSONDecodeError:
#             #             enriched = {"llm_output": enriched_raw}
#             #         enriched_chapters.append({
#             #             **chapter,
#             #             **enriched
#             #         })

#             #     final_output = {
#             #         "book_metadata": cleaned_book['book_metadata'],
#             #         "chapters": enriched_chapters
#             #     }

#             #     # Format: A_Clergyman's_Daughter_George_Orwell_enriched.json
#             #     book_title = cleaned_book['book_metadata']['title']
#             #     book_author = cleaned_book['book_metadata']['author']
#             #     output_path = os.path.join(
#             #         ENRICHED_OUTPUT_DIR,
#             #         f"{book_title}-{book_author}_enriched.json"
#             #     )

#             #     with open(output_path, "w", encoding="utf-8") as out_f:
#             #         json.dump(final_output, out_f, ensure_ascii=False, indent=4)

#             #     print(f"✅ Enriched: {filename} -> {output_path}")

#             # except Exception as e:
#             #     print(f"❌ Error processing {filename}: {e}")



# PHASE 1: Text Cleaning and Structuring
import os
import re
import json
import sys
import tiktoken
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.constants import AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT_NAME, RAW_BOOK_DIR, CLEANED_OUTPUT_DIR
RAW_BOOK_DIR = RAW_BOOK_DIR
CLEANED_OUTPUT_DIR = CLEANED_OUTPUT_DIR
AZURE_ENDPOINT = AZURE_ENDPOINT
AZURE_API_KEY = AZURE_API_KEY
AZURE_DEPLOYMENT_NAME = AZURE_DEPLOYMENT_NAME
client = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_API_KEY)
) 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# RAW_TEXT_PATH = "data\raw_book"
# OUT_JSON_PATH = 'data\transformed_book'
def clean_chapter_text(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def split_chapters(text):
    """
    Splits book into chapters using:
    - TOC-based essay splitting
    - CHAPTER X + subchapters (e.g., 1\n\n or Chapter 1\n)
    - Fallbacks: UPPERCASE, PART, CHAPTER, Roman, Numeric
    """

    def clean_chapter_text(txt):
        txt = txt.replace('\r\n', '\n').replace('\r', '\n')
        txt = re.sub(r'-\n', '', txt)
        txt = re.sub(r'(?<!\n)\n(?!\n)', ' ', txt)
        txt = re.sub(r'\n{2,}', '\n\n', txt)
        txt = re.sub(r'[ \t]+', ' ', txt)
        return txt.strip()

    text = clean_chapter_text(text)

    # === STEP 1: TOC-style essays (for Fifty Orwell Essays)
    toc_match = re.search(r"Contents\s*\n(.*?)(?:\n{2,}|\Z)", text, re.DOTALL)
    if toc_match:
        toc_block = toc_match.group(1)
        lines = [l.strip() for l in toc_block.splitlines() if l.strip()]
        toc_titles = []
        buffer = ''
        for line in lines:
            if re.match(r'^\(?\d{4}\)?$', line):
                if toc_titles:
                    toc_titles[-1] += f" {line}"
                continue
            if buffer:
                buffer += f" {line}"
                toc_titles.append(buffer.strip())
                buffer = ''
            else:
                buffer = line
        if buffer:
            toc_titles.append(buffer.strip())
        toc_titles_upper = [re.sub(r'\s+', ' ', t.upper()) for t in toc_titles]
        print(f"✅ Extracted {len(toc_titles_upper)} TOC titles.")

    # STEP 2: Strict essay split by UPPERCASE + \n\n
    matches = list(re.finditer(r"\n{2,}([A-Z][A-Z0-9 ,.\'\"\-\—:()]+)\n", text))
    chapters = []

    if matches and matches[0].start() > 0:
        intro_text = text[:matches[0].start()].strip()
        chapters.append({
            "chapter_number": 0,
            "title": "Introduction",
            "text": clean_chapter_text(intro_text)
        })

    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapter_text = clean_chapter_text(text[start_idx:end_idx])
        chapters.append({
            "chapter_number": len(chapters),
            "title": title,
            "text": chapter_text
        })

    if len(chapters) >= 52:
        print(f"✅ Split {len(chapters)-1} essays + intro using strict essay pattern.")
        return chapters

    # === Fallback 1: CHAPTER X + subchapters (number or Chapter-style)
    chapter_pattern = r"\n{2,}(CHAPTER\s+\d+)\n+"
    chapter_matches = list(re.finditer(chapter_pattern, text, flags=re.IGNORECASE))
    if chapter_matches:
        chapters = []
        for i, match in enumerate(chapter_matches):
            chapter_title = match.group(1)
            start_idx = match.end()
            end_idx = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(text)
            chapter_text = text[start_idx:end_idx]

            sub_pattern = r"(?:^|\n)(?:Chapter\s+)?(\d{1,2})(?:\n|\n{2,})"
            sub_matches = list(re.finditer(sub_pattern, chapter_text, flags=re.IGNORECASE))

            subchapters = []
            if sub_matches:
                for j, sub_match in enumerate(sub_matches):
                    sub_title = sub_match.group(1)
                    sub_start = sub_match.end()
                    sub_end = sub_matches[j + 1].start() if j + 1 < len(sub_matches) else len(chapter_text)
                    sub_text = chapter_text[sub_start:sub_end].strip()
                    if j == 0 and sub_match.start() > 0:
                        pre_text = chapter_text[:sub_match.start()].strip()
                        if pre_text:
                            subchapters.append({"subchapter": "0", "text": clean_chapter_text(pre_text)})
                    subchapters.append({"subchapter": sub_title, "text": clean_chapter_text(sub_text)})
            else:
                subchapters.append({"subchapter": None, "text": clean_chapter_text(chapter_text)})

            chapters.append({
                "chapter_number": i + 1,
                "title": chapter_title,
                "subchapters": subchapters
            })

        print(f"✅ Split {len(chapters)} chapters with subchapters.")
        return chapters
    # === Fallback 2: UPPERCASE blocks (excluding slogans) with subchapters
    excluded_phrases = {"FREEDOM IS SLAVERY", "TWO AND TWO MAKE FIVE", "GOD IS POWER"}
    exclude_pattern = "|".join(re.escape(p) for p in excluded_phrases)
    uppercase_title_pattern = rf"(?:^|\n)(?!{exclude_pattern})([A-Z][A-Z\s'’:\-]{{3,}})(?:\n|\r|\r\n)+"

    matches = list(re.finditer(uppercase_title_pattern, text))
    if len(matches) >= 2:
        chapters = []
        for idx, match in enumerate(matches):
            chapter_title = match.group(1).strip()
            start_idx = match.end()
            end_idx = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            chapter_text = text[start_idx:end_idx]

            # Subchapter detection: matches "1", "2", "3", etc. as a heading line
            sub_pattern = r"(?:^|\n)[ \t]*(\d{1,2})[ \t]*(?=\n{2,})"
            sub_matches = list(re.finditer(sub_pattern, chapter_text))

            subchapters = []
            if sub_matches:
                for j, sub_match in enumerate(sub_matches):
                    sub_title = sub_match.group(1)
                    sub_start = sub_match.end()
                    sub_end = sub_matches[j + 1].start() if j + 1 < len(sub_matches) else len(chapter_text)
                    sub_text = chapter_text[sub_start:sub_end].strip()
                    if j == 0 and sub_match.start() > 0:
                        pre_text = chapter_text[:sub_match.start()].strip()
                        if pre_text:
                            subchapters.append({"subchapter": "0", "text": clean_chapter_text(pre_text)})
                    subchapters.append({"subchapter": sub_title, "text": clean_chapter_text(sub_text)})
            else:
                subchapters.append({"subchapter": None, "text": clean_chapter_text(chapter_text)})

            chapters.append({
                "chapter_number": idx + 1,
                "title": chapter_title,
                "subchapters": subchapters
            })
        print(f"✅ Fallback: {len(chapters)} chapters using UPPERCASE titles with subchapters.")
        return chapters

    # === Fallback 3: CHAPTER/PART patterns with CHAPTER subchapters
    part_pattern = re.compile(r"(?:^|\n)(PART\s+[IVXLC0-9]+)", flags=re.IGNORECASE)
    chapter_pattern = re.compile(r"(?:^|\n)(CHAPTER\s+\d+)", flags=re.IGNORECASE)

    part_matches = list(re.finditer(part_pattern, text))
    if part_matches:
        chapters = []
        for i, part_match in enumerate(part_matches):
            part_title = part_match.group(1).strip()
            part_start = part_match.end()
            part_end = part_matches[i + 1].start() if i + 1 < len(part_matches) else len(text)
            part_text = text[part_start:part_end]

            subchapters = []
            chapter_matches = list(re.finditer(chapter_pattern, part_text))
            if chapter_matches:
                for j, sub_match in enumerate(chapter_matches):
                    sub_title = sub_match.group(1).strip()
                    sub_start = sub_match.end()
                    sub_end = chapter_matches[j + 1].start() if j + 1 < len(chapter_matches) else len(part_text)
                    sub_text = part_text[sub_start:sub_end].strip()
                    subchapters.append({
                        "subchapter": sub_title,
                        "text": clean_chapter_text(sub_text)
                    })
            else:
                subchapters.append({
                    "subchapter": None,
                    "text": clean_chapter_text(part_text)
                })

            chapters.append({
                "chapter_number": i + 1,
                "title": part_title,
                "subchapters": subchapters
            })
        print(f"✅ Fallback: Structured {len(chapters)} parts with CHAPTER substructure.")
        return chapters

    # If fallback 3 failed but fallback 2 succeeded, return fallback 2
    if fallback2_result:
        print(f"✅ Fallback: {len(fallback2_result)} chapters using UPPERCASE titles.")
        return fallback2_result

    # === Final fallback
    print("⚠️ No split logic matched. Returning full text.")
    return [{
        "chapter_number": 1,
        "title": "Full Text",
        "text": clean_chapter_text(text)
    }]

def enrich_book_metadata(title, author):
    prompt = f"""
You are a literary analyst assistant. Based on the following book titled \"{title}\" by {author}, generate structured metadata in **valid raw JSON** format only — without any markdown, explanation, or code block.

Return only the JSON object with the following fields:
- genre: list of genres
- description: 1–2 sentence summary
- keywords: list of 5–10 important keywords
- age_range: suitable age group (e.g., "14+")
- grade_range: appropriate grade levels (e.g., "9-12")
- related_titles: list of similar famous book titles (include author names if possible)
- publication_year: year of first publication (if known)
"""
    response = client.complete(
        messages=[
            SystemMessage(content="You are a helpful literary assistant."),
            UserMessage(content=prompt)
        ],
        max_tokens=16384,
        temperature=0.7,
        top_p=1.0,
        model=AZURE_DEPLOYMENT_NAME
    )
    return response.choices[0].message.content
def prepare_cleaned_json(book_title, author, chapters,enrich_metadata):
    try:
        metadata_fields = json.loads(enrich_metadata)
    except:
        metadata_fields = {
            "llm_output": enrich_metadata
        }
    book_metadata = {
        "title": book_title,
        "author": author,
        **metadata_fields
    }
    return {
        "book_metadata": book_metadata,
        "chapters": chapters
    }
def clean_llm_output(raw_response):
    # Remove triple backticks and leading/trailing whitespace
    cleaned = raw_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()
def enrich_chapter(chapter, metadata,max_tokens):
    prompt = prompt = f"""
You are an education-focused literary analyst.

Here is Chapter {chapter['chapter_number']} of "{metadata['title']}" by {metadata['author']}:

{chapter['text']}

🛑 IMPORTANT INSTRUCTIONS:
- Return ONLY valid JSON.
- Do NOT use triple backticks (```).
- Do NOT include any explanation or comments.
- The output must be directly parsable by Python’s json.loads() function.

🧠 Expected JSON format:
{{
  "summary": "Short paragraph summary...",
  "literary_analysis": {{
    "themes": ["..."],
    "motifs": ["..."],
    "symbols": {{"...": "..."}},
    "setting": {{
      "place": "...",
      "time": "...",
      "atmosphere": "..."
    }},
    "protagonist": "...",
    "antagonist": "...",
    "tone": "...",
    "style": "...",
    "point_of_view": "...",
    "genre": "...",
    "allusions": ["..."],
    "foreshadowing": "..."
  }},
  "character_analysis": {{
    "Character Name": {{
      "description": "...",
      "arc": "...",
      "relationships": {{
        "Other Character": "Description of relationship"
      }}
    }}
  }},
  "educational_qa_pairs": [
    {{
      "question": "...",
      "answer": "..."
    }},
    {{
      "question": "...",
      "answer": "..."
    }}
  ]
}}
"""


    response = client.complete(
        messages=[
            SystemMessage(content="You are a helpful educational assistant."),
            UserMessage(content=prompt)
        ],
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=1.0,
        model=AZURE_DEPLOYMENT_NAME
    )
 
    return response.choices[0].message.content


encoding = tiktoken.encoding_for_model("gpt-4o")

def truncate_text(text, max_tokens):
    tokens = encoding.encode(text)
    truncated = encoding.decode(tokens[:max_tokens])
    return truncated

def safe_enrich(chapter, metadata, retries=[4096, 2048, 1024]):
    for max_tokens in retries:
        try:
            truncated_chapter = {
                **chapter,
                "text": truncate_text(chapter["text"], max_tokens)
            }

            enriched_raw = enrich_chapter(truncated_chapter, metadata, max_tokens)

            try:
                enriched = json.loads(enriched_raw)
                if isinstance(enriched, str):
                    enriched = json.loads(enriched)
                return enriched
            except json.JSONDecodeError:
                print(f"⚠️ JSON decode failed at {max_tokens} tokens for Chapter {chapter['chapter_number']}")
                # Optional: save output for debugging
                debug_path = f"debug_ch{chapter['chapter_number']}_{max_tokens}.txt"
                with open(debug_path, "w", encoding="utf-8") as debug_file:
                    debug_file.write(enriched_raw)

        except Exception as e:
            print(f"❌ Exception at {max_tokens} tokens for Chapter {chapter['chapter_number']}: {e}")

    # Final fallback
    return {"llm_output": chapter["text"]}
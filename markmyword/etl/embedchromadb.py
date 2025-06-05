import chromadb
from chromadb.config import Settings
import json
import os
import glob
import sys
import requests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.constants import AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT_NAME,AZURE_API_VERSION
from sentence_transformers import SentenceTransformer
# Conntect to ChromaDB Server

client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    settings=Settings(anonymized_telemetry=False)
)
# Load the embedding model (local, fast, accurate)
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # You can switch to another model if needed

# Define the function for embedding
def embed_text(texts):
    """Embed a list of texts using sentence-transformers."""
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()

# Create or get collection
try:
    client.delete_collection("book_collection")
    print("✅ Collection deleted")
except Exception as e:
    print(f"⚠️ Could not delete collection (maybe it didn't exist): {e}")
collection = client.get_or_create_collection(
    name="book_collection",
    embedding_function=None  # Put it HERE, not in HttpClient
)

# Prepare Data
documents = []
metadatas = []
ids = []

# Load all books from the directory
script_dir = os.path.dirname(os.path.abspath(__file__))
book_dir = os.path.normpath(os.path.join(script_dir, '..', 'data', 'final'))
book_files = glob.glob(os.path.join(book_dir, '*.json'))
for book_file in book_files:
    with open(book_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    book_title  = data['book_metadata']['title']
    author = data['book_metadata']['author']
    genres = ", ".join(data['book_metadata']['genre'])
    description = data['book_metadata']['description']
    publication_year = data['book_metadata']['publication_year']
    age_range = data['book_metadata']['age_range']
    grade_range = data['book_metadata']['grade_range']
    keywords = ", ".join(data['book_metadata']['keywords'])
    related_titles = ", ".join(data['book_metadata']['related_titles'])
    for chapter in data['chapters']:
        chapter_number = chapter.get('chapter_number', 'No chapter available')
        chapter_title = chapter.get('title', 'No title available')
        chapter_text = chapter.get('text', 'No text available')
        educational_qa_pairs = chapter.get('educational_qa_pairs', 'No educational QA pairs available')
        character_analysis  = chapter.get('character_analysis', 'No character analysis available')
        literary_analysis = chapter.get('literary_analysis', 'No literary analysis available')
        summary = chapter.get('summary', 'No summary available')
        # prepare full text for embedding
        full_text_for_embedding = f"{chapter_title} \n {chapter_text} \n Summary: {summary} \n Literary Analysis: {literary_analysis}"
        documents.append(full_text_for_embedding)
        ids.append(f"{book_title.replace(' ', '_')}- chapter-{chapter_number}")
        metadatas.append({
            "book_title": book_title,
            "author": author,
            "genres": genres,
            "description": description,
            "publication_year": publication_year,
            "age_range": age_range,
            "grade_range": grade_range,
            "keywords": keywords,
            "related_titles": related_titles,
            "chapter_number": chapter_number,
            "chapter_title": chapter_title,
            # Convert complex objects to string
            "educational_qa_pairs": json.dumps(educational_qa_pairs),
            "character_analysis": json.dumps(character_analysis),
            "literary_analysis": json.dumps(literary_analysis),
            "summary": summary
        })
embeddings = embed_text(documents)
print(f"embedding sample: {embeddings[0]}")
print(f"Total embeddings: {len(embeddings)}, Each shape: {len(embeddings[0])}")
collection.add(
    ids=ids,
    metadatas=metadatas,
    embeddings=embeddings,
    documents=documents
)
print(f"✅ Inserted Successfully")      

        




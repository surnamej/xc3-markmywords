# 📚 MarkMyWords – Document Embedding Pipeline with ChromaDB

This project builds a document embedding pipeline for educational content (e.g. George Orwell books) using [ChromaDB](https://www.trychroma.com/), [sentence-transformers](https://www.sbert.net/), and a custom data processing flow. It supports local or containerized deployment via Docker.
## 🧰 Requirements

- Python 3.11+
- Docker & Docker Compose (You should download the Docker desktop in both Windows, and MacOS)
- Git
- (Optional) Virtual environment tool (`venv`, `conda`)

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repo

```bash
git clone https://github.com/surnamej/COS80029-Swinburne-MarkMyWords.git
cd COS80029-Swinburne-MarkMyWords
```
### 2️⃣ Start ChromaDB with Docker
```bash
cd chroma_db
docker-compose up -d
```
At this part you can check the Chromadb with the data I have embed it into the Chromadb, go to this link: localhost:8501. You can do multiple things with the Chromadb by clicking UI since I have already build the function inside it. Or you can do furturstep below (by deleting the sub folder in chroma_db/my_db (you should keep the folder my_db or you can remove it if you want, it will automatically create for you when you start the docker)
### 3️⃣ Install requirements
```bash
pip install -r requirements.txt
pip install -r requirements2.txt
```
## 🧪 Data Preparation & Embedding
### 🔧 Step 1: Generate enriched JSON data
```bash
python3 dags/flow_dag.py
```
### 🧠 Step 2: Embed documents and insert into ChromaDB
```bash
python3 etl/embedchromadb.py
```
This will:

Read enriched chapters

Generate sentence embeddings

Store in a ChromaDB collection called book_collection
### ✅ Expected Output
```bash
✅ Inserted Successfully
```


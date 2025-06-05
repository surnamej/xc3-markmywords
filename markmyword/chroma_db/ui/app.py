import streamlit as st
import requests

st.title("🧠 ChromaDB Admin UI")

chroma_api = "http://vectordb:8000"  # Docker service name

# Your API Token (set it here)
api_token = "test-token"  # Must match CHROMA_SERVER_AUTHN_CREDENTIALS

# Common headers
headers = {
    "Authorization": f"Bearer {api_token}"
}

# Section: Heartbeat
st.subheader("💓 Health Check")
if st.button("Check Heartbeat"):
    r = requests.get(f"{chroma_api}/api/v1/heartbeat", headers=headers)
    st.json(r.json())

# Section: Collections
st.subheader("📚 Manage Collections")
if st.button("List Collections"):
    try:
        r = requests.get(f"{chroma_api}/api/v1/collections", headers=headers)  # << You forgot this line
        collections = r.json()
        if not isinstance(collections, list):
            raise ValueError("Expected a list of collections")
        for col in collections:
            st.markdown(f"- **{col['name']}** (ID: `{col['id']}`)")
    except Exception as e:
        st.error(f"Error loading collections: {e}")

# Create new collection
st.markdown("### ➕ Create New Collection")
with st.form("create_form"):
    col_name = st.text_input("Collection name")
    col_desc = st.text_input("Description (optional)")
    submitted = st.form_submit_button("Create Collection")
    if submitted and col_name:
        payload = {
            "name": col_name,
            "metadata": {"description": col_desc}
        }
        r = requests.post(f"{chroma_api}/api/v1/collections", headers=headers, json=payload)
        st.success(f"✅ Created collection: {col_name}") if r.status_code == 200 else st.error(r.text)

# Delete a collection
st.markdown("### ❌ Delete Collection")
del_col = st.text_input("Collection name to delete")
if st.button("Delete Collection"):
    r = requests.delete(f"{chroma_api}/api/v1/collections/{del_col}", headers=headers)
    st.success("✅ Deleted") if r.status_code == 200 else st.error(r.text)

# Search documents by metadata filter
st.markdown("### 🔍 Query by Metadata")
query_col = st.text_input("Collection name to query")
filter_key = st.text_input("Metadata field (e.g. book_title)")
filter_val = st.text_input("Value to match")

if st.button("Search"):
    try:
        payload = {
            "where": {filter_key: filter_val},
            "limit": 10,
            "include": ["embeddings", "documents", "metadatas"]
        }
        r = requests.post(
            f"{chroma_api}/api/v1/collections/{query_col}/get",
            headers=headers,
            json=payload
        )
        if r.status_code == 200:
            st.json(r.json())
        else:
            st.error(r.text)
    except Exception as e:
        st.error(f"Error: {e}")

st.caption("Built with ❤️ using ChromaDB and Streamlit")
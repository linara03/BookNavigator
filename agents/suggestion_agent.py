import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
books = pd.read_csv(r"C:\Users\Acer\Desktop\Book_Recommendation\BookNavigator\data\books_cleaned.csv")

# -----------------------------
# 2️⃣ Initialize embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# 3️⃣ Prepare Chroma DB
# -----------------------------
documents = [
    Document(page_content=row['tagged_description'], metadata={"isbn13": row['isbn13']})
    for _, row in books.iterrows()
]

db_books = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory="db_books"  # Database will be automatically persisted in Chroma >= 0.4.x
)


# -----------------------------
# 4️⃣ Define semantic recommendation function
# -----------------------------
def retrieve_semantic_recommendations(query: str, top_k: int = 10) -> pd.DataFrame:
    """
    Retrieve top-k books similar to the query using Chroma embeddings.
    """
    # Perform similarity search
    recs = db_books.similarity_search(query, k=50)

    # Extract ISBNs safely from document metadata
    isbn_list = [doc.metadata.get("isbn13") for doc in recs if "isbn13" in doc.metadata]

    # Return top-k recommended books
    return books[books["isbn13"].isin(isbn_list)].head(top_k)


# -----------------------------
# 5️⃣ Main execution
# -----------------------------
if _name_ == "_main_":
    query_text = "A book to teach children about animals"
    recommendations = retrieve_semantic_recommendations(query_text)

    print("📚 Top Book Recommendations:")
    print(recommendations[["isbn13", "title"]])
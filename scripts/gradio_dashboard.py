import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# --- Load dataset (use the new combined CSV) ---
books_path = r"C:/Users/MSI/Desktop/BookNavigator/books_with_emotions_and_ratings.csv"
books = pd.read_csv(books_path)

# --- Handle missing thumbnails with fallback ---
local_cover = os.path.join(os.path.dirname(__file__), "Nocover.jpg")
if os.path.exists(local_cover):
    default_cover = local_cover
else:
    default_cover = "https://via.placeholder.com/150x220.png?text=No+Cover"

books["large_thumbnail"] = books["thumbnail"].fillna(default_cover)

# --- Initialize sentence transformer model ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Generate embeddings for all book descriptions ---
print("Generating embeddings for all book descriptions...")
book_embeddings = model.encode(
    books["description"].fillna("").tolist(),
    show_progress_bar=True
)

# --- Semantic search function ---
def retrieve_semantic_recommendations(query, category="All", tone="All", rating="All", top_k=12):
    query_emb = model.encode([query])
    similarities = cosine_similarity(query_emb, book_embeddings)[0]
    books["similarity"] = similarities

    # Apply filters
    filtered_books = books.copy()
    if category != "All":
        filtered_books = filtered_books[filtered_books["simple_categories"] == category]
    if rating != "All":
        filtered_books = filtered_books[filtered_books["rating_category"] == rating]
    if tone != "All":
        sort_key = {
            "Happy": "happiness",
            "Surprising": "surprise",
            "Angry": "anger",
            "Suspenseful": "fear",
            "Sad": "sadness"
        }[tone]
        filtered_books = filtered_books.sort_values(by=sort_key, ascending=False)

    # Sort by similarity
    filtered_books = filtered_books.sort_values(by="similarity", ascending=False)
    return filtered_books.head(top_k)

# --- Recommendation wrapper ---
def recommend_books(query, category, tone, rating):
    recs = retrieve_semantic_recommendations(query, category, tone, rating)
    results = []
    for _, row in recs.iterrows():
        desc_trunc = " ".join(str(row["description"]).split()[:40]) + "..."
        authors = str(row["authors"])
        caption = (
            f"### 📖 {row['title']}\n"
            f"👤 *{authors}*\n"
            f"⭐ {row['average_rating']} ({row['rating_category']})\n\n"
            f"{desc_trunc}"
        )
        results.append((row["large_thumbnail"], caption))
    return results

# --- Dropdown options ---
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
ratings = ["All"] + sorted(books["rating_category"].dropna().unique())

# --- UI ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="gray")) as dashboard:
    # Title section
    with gr.Row():
        gr.Markdown(
            """
            <div style="text-align:center; padding: 10px;">
                <h1 style="color:#0D3B66;">📚 Book Navigator</h1>
                <p style="font-size:18px; color:#3E4C59;">
                    Discover your next favorite book with <b>AI-powered personalized recommendations</b>.  
                    Enter a short description, apply filters, and explore our curated gallery!
                </p>
            </div>
            """,
            elem_id="title"
        )

    # Input and filters
    with gr.Row():
        with gr.Column(scale=2):
            user_query = gr.Textbox(
                label="Enter a book description:",
                placeholder="e.g., A story about friendship and adventure",
                lines=2
            )
            submit_button = gr.Button("🔍 Find Recommendations", variant="primary")
        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(choices=categories, label="📂 Select Category", value="All")
            tone_dropdown = gr.Dropdown(choices=tones, label="🎭 Emotional Tone", value="All")
            rating_dropdown = gr.Dropdown(choices=ratings, label="⭐ Rating Category", value="All")

    # Output gallery
    with gr.Row():
        output = gr.Gallery(
            label="✨ Recommended Books",
            columns=4,
            rows=3,
            object_fit="contain",
            height="auto"
        )

    # Footer
    gr.Markdown(
        """
        <div style="text-align:center; color:#6B7280; padding:15px; font-size:14px;">
            🚀 Powered by Sentence Transformers + Gradio | Developed by Team <b>SYNERGY </b>
        </div>
        """
    )

    # Connect button
    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown, rating_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    dashboard.launch()

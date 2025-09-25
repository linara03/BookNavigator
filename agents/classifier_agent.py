import os
import pandas as pd
from transformers import pipeline

# -------------------------
# 1️⃣ Set file path
# -------------------------
# Update this path to where your CSV file actually exists
file_path = r"D:\BookNavigator\data\books_cleaned.csv"

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"CSV file not found at: {file_path}")

# -------------------------
# 2️⃣ Load dataset
# -------------------------
books = pd.read_csv(file_path)

# -------------------------
# 3️⃣ Initialize classifier
# -------------------------
# Using zero-shot classification
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1  # CPU; change to 0 for GPU if available
)

# -------------------------
# 4️⃣ Define categories
# -------------------------
categories = ["Fiction", "Nonfiction", "Children's Fiction", "Children's Nonfiction"]


# -------------------------
# 5️⃣ Functions
# -------------------------
def classify_book(description: str) -> str:
    """
    Classify a book description into a category.
    Returns 'Unknown' if description is empty.
    """
    if not description or pd.isna(description):
        return "Unknown"
    result = classifier(description, candidate_labels=categories)
    return result["labels"][0]  # top predicted label


def classify_all_books(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'predicted_category' column to the dataframe
    using classify_book function.
    """
    # Ensure 'tagged_description' column exists
    if 'tagged_description' not in df.columns:
        raise KeyError("'tagged_description' column not found in dataframe.")

    df["predicted_category"] = df["tagged_description"].apply(classify_book)
    return df


# -------------------------
# 6️⃣ Run script
# -------------------------
if _name_ == "_main_":
    books_classified = classify_all_books(books)
    print(books_classified[["isbn13", "title", "predicted_category"]].head())

    # Optional: save results
    output_path = r"D:\BookNavigator\data\books_classified.csv"
    books_classified.to_csv(output_path, index=False)
    print(f"✅ Classified CSV saved at: {output_path}")
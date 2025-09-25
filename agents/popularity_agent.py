import pandas as pd

# Load books dataset
books = pd.read_csv(r"D:\BookNavigator\data\books_cleaned.csv")

def get_top_books(df: pd.DataFrame, metric: str = "ratings_count", top_k: int = 10) -> pd.DataFrame:
    """
    Return top-k popular books based on a metric column.
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in dataframe.")
    return df.sort_values(by=metric, ascending=False).head(top_k)

if _name_ == "_main_":
    top_books = get_top_books(books, metric="ratings_count", top_k=10)
    print(top_books[["isbn13", "title", "ratings_count"]])
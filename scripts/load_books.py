import pandas as pd
import os

# Define path to your dataset
path = "C:/Users/Acer\Desktop\Book_Recommendation\BookNavigator\data"

# Load CSV
books = pd.read_csv(f"{path}/books_cleaned.csv")

# Check first rows
print(books.head())
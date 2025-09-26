 # Book Navigator - Intelligent Book Recommendation System

Welcome to *Book Navigator, an intelligent book recommendation system that helps readers discover their next favorite book! Using **AI agents, semantic search, and natural language processing*, this system provides personalized recommendations based on what you want to read, your preferred genre, emotional tone, and book ratings.

---

## Project Overview

Book Navigator is designed with *agentic AI principles, meaning it uses multiple intelligent agents that **communicate and collaborate* to deliver recommendations. Unlike a simple search engine, this system analyzes multiple dimensions of books to provide *personalized and meaningful suggestions*.

### Main Agents

1. *Book Recommendation Agent*  
   - Takes user input (book description or keywords)  
   - Finds candidate books using *semantic similarity search* with *vector embeddings* (Sentence Transformers & Chroma)  

2. *Genre Classifier Agent*  
   - Tags candidate books into categories (Fiction, Nonfiction, Children’s Fiction, etc.)  
   - Uses *zero-shot classification* with *HuggingFace Transformers*  

3. *Popularity Analyzer Agent*  
   - Scores books based on ratings, number of reviews, and popularity metrics  
   - Ensures highly-rated and trusted books appear higher in recommendations  

4. *Suggestion Agent*  
   - Combines outputs from all agents (semantic similarity, genre, rating, emotional tone)  
   - Produces a final *ranked list of recommended books*  

> This collaborative workflow makes Book Navigator a true *agentic AI system* rather than a simple search tool.

---

## Key Features

- *Semantic search* powered by *Sentence Transformers* for understanding book descriptions  
- *Vector database* using *Chroma* for fast and accurate similarity queries  
- *Genre prediction* with *zero-shot classification*  
- *Emotion analysis* using pre-trained *DistilRoBERTa emotion model*  
- Filter recommendations by *genre, rating, and emotional tone*  
- Shows *book thumbnails, titles, authors, brief descriptions, and ratings*  
- Handles queries like: “A story about friendship and adventure”  
- Provides *top-K recommendations* based on combined agent outputs  

---

## Technologies Used

| Feature | Technology |
|---------|------------|
| Data Cleaning & Preprocessing | Python, Pandas, NumPy |
| Semantic Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database | Chroma Vector Store |
| Genre Classification | HuggingFace Transformers (facebook/bart-large-mnli) |
| Emotion Analysis | HuggingFace Transformers (j-hartmann/emotion-english-distilroberta-base) |
| Agent Interaction | Python DataFrames, internal pipeline |
| User Interface | Gradio Blocks |
| Similarity Search | Cosine Similarity on embeddings |

---

## How It Works

1. *User Input:* The user enters a book description or keywords.  
2. *Semantic Search (Book Recommendation Agent):* Converts book descriptions and user query into embeddings and computes similarity.  
3. *Genre Classification (Genre Classifier Agent):* Assigns a category to each candidate book using zero-shot classification.  
4. *Popularity Scoring (Popularity Analyzer Agent):* Scores books based on rating and review metrics.  
5. *Recommendation Integration (Suggestion Agent):* Combines similarity, genre, rating, and emotional tone to produce a *ranked list* of recommended books.  
6. *Output:* Displays a gallery of books with images, titles, authors, ratings, and brief descriptions.

> This *agentic workflow* ensures recommendations are *personalized, relevant, and context-aware*, not just keyword-based.

---

## Steps to Use

1. Enter a *book description* in the input box.  
2. Optionally select filters:  
   - *Category*: Fiction, Nonfiction, Children’s Fiction, etc.  
   - *Tone*: Happy, Sad, Angry, Surprising, Suspenseful  
   - *Rating*: Excellent, Very Good, Good, etc.  
3. Click *Find Recommendations*.  
4. View a gallery of recommended books with *images and brief summaries*.  

---

## Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/nipdofficial/BookNavigator.git
cd BookNavigator

# Game of Thrones — Word2Vec

A Word2Vec model trained on all five books of the *A Song of Ice and Fire* series by George R.R. Martin. The project converts raw text into word vector embeddings and visualizes them in an interactive 3D scatter plot.

## How It Works

### 1. Loading the Corpus

All five books are read from the `GOT_Book/` directory:

- *A Game of Thrones*
- *A Clash of Kings*
- *A Storm of Swords*
- *A Feast for Crows*
- *A Dance with Dragons*

Each file is read and its text is concatenated into one large corpus string.

### 2. Sentence Tokenization

Using NLTK's `sent_tokenize`, the corpus is split into individual sentences. Each sentence is then lowercased and split into a list of words, producing **158,874 sentences** ready for training.

### 3. Training Word2Vec

The word lists are fed into **Gensim's Word2Vec** model with the following parameters:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `vector_size` | 100 | Each word is represented as a 100-dimensional vector |
| `window` | 10 | Context window of 10 words on each side |
| `min_count` | 2 | Ignores words that appear fewer than 2 times |
| `workers` | 10 | Parallel training threads |

Word2Vec learns vector representations by predicting neighbouring words in a sliding window across every sentence. Words that appear in similar contexts end up with similar vectors — so *"Stark"* and *"Lannister"* land close together because they share the context of being noble houses.

### 4. Exploring the Model

Once trained, the model supports queries like:

```python
model.wv.most_similar("daenerys")   # words closest to Daenerys
model.wv.most_similar("winterfell") # words closest to Winterfell
model.wv.most_similar(positive=["arya"], negative=["sansa"])  # vector arithmetic
model.wv.doesnt_match(["cersei", "jaime", "tyrion", "stark"]) # odd one out
```

### 5. Dimensionality Reduction & Visualization

The 100-dimensional word vectors are reduced to **3 dimensions** using **PCA** (Principal Component Analysis) from scikit-learn. The reduced vectors are then plotted as an interactive 3D scatter plot using **Plotly**, with each point labelled by its word.

## Output

### 3D Scatter Plot — Desktop View
![3D Word2Vec scatter plot viewed in desktop browser, hovering over "robert"](Screenshot%202026-04-18%20080749.png)

### 3D Scatter Plot — Mobile View
![3D Word2Vec scatter plot viewed on mobile, hovering over "tyrion"](WhatsApp%20Image%202026-02-09%20at%201.59.38%20PM.jpeg)

The plot is interactive — hover over any point to see the word and its PCA coordinates. Character names, locations, and thematically related words cluster together.

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Download NLTK tokenizer data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Run the script
python game_of_thrones_word2vec.py
```

The script generates `word2vec_plot.html` which opens automatically in your default browser.

## Project Structure

```
├── GOT_Book/                        # Source text files (all 5 books)
├── game_of_thrones_word2vec.py      # Main script
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

## Dependencies

- **gensim** — Word2Vec training
- **nltk** — Sentence tokenization
- **scikit-learn** — PCA dimensionality reduction
- **plotly** — Interactive 3D visualization
- **numpy** / **pandas** — Data handling

## Author

**Amit Rastogi**

import numpy as np
import pandas as pd
import gensim
import os
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from sklearn.decomposition import PCA
import plotly.express as px

# --- 1. Load and tokenize the GOT books ---
story = []
data_dir = 'GOT_Book'
for filename in os.listdir(data_dir):
    filepath = os.path.join(data_dir, filename)
    with open(filepath, encoding='utf-8', errors='ignore') as f:
        corpus = f.read()
    raw_sent = sent_tokenize(corpus)
    for sent in raw_sent:
        story.append(simple_preprocess(sent))

print(f"Total sentences: {len(story)}")

# --- 2. Build and train Word2Vec model ---
model = gensim.models.Word2Vec(
    window=10,
    min_count=2
)
model.build_vocab(story)
model.train(story, total_examples=model.corpus_count, epochs=model.epochs)

# --- 3. Explore the model ---
print("\nMost similar to 'daenerys':")
print(model.wv.most_similar('daenerys'))

print("\nDoesn't match in ['jon','rikon','robb','arya','sansa','bran']:")
print(model.wv.doesnt_match(['jon', 'rikon', 'robb', 'arya', 'sansa', 'bran']))

print("\nDoesn't match in ['cersei', 'jaime', 'bronn', 'tyrion']:")
print(model.wv.doesnt_match(['cersei', 'jaime', 'bronn', 'tyrion']))

print("\nVector for 'king':")
print(model.wv['king'])

print(f"\nSimilarity arya-sansa:  {model.wv.similarity('arya', 'sansa')}")
print(f"Similarity cersei-sansa: {model.wv.similarity('cersei', 'sansa')}")
print(f"Similarity tywin-sansa:  {model.wv.similarity('tywin', 'sansa')}")

# --- 4. PCA visualization ---
y = model.wv.index_to_key
pca = PCA(n_components=3)
X = pca.fit_transform(model.wv.get_normed_vectors())
print(f"\nPCA shape: {X.shape}")

fig = px.scatter_3d(X[200:300], x=0, y=1, z=2, color=y[200:300], text=y[200:300])
fig.update_traces(textposition='top center', textfont_size=8)
fig.write_html("word2vec_plot.html", auto_open=True)
print("Plot saved to word2vec_plot.html")

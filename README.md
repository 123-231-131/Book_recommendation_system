# Book Recommendation System

This repository contains a set of data files and a Jupyter notebook for building a hybrid book recommendation system. It combines a popularity-based model with a collaborative filtering approach that uses cosine similarity between user rating vectors. The project is based on the [Book-Crossing dataset](http://www2.informatik.uni-freiburg.de/%7Ecziegler/BX/).

## Repository Contents

- `Books.csv`, `Users.csv`, `Ratings.csv` – Raw data exported from the Book-Crossing dataset.
- `Untitled.ipynb` – Jupyter notebook that walks through exploratory analysis and builds the recommendation pipelines.
- `popular.pkl`, `pt.pkl`, `books.pkl`, `similarity_scores.pkl` – Pickled artifacts produced by the notebook:
  - `popular.pkl`: Top books ranked by popularity (volume and quality of ratings).
  - `pt.pkl`: Pivot table of book titles versus user identifiers with ratings filled in.
  - `books.pkl`: Serialized book metadata used to enrich recommendations.
  - `similarity_scores.pkl`: Precomputed cosine similarity matrix between books.

## Getting Started

### Prerequisites

Create a Python 3.9+ environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib notebook
```

### Running the Notebook

1. Activate your virtual environment (see above).
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `Untitled.ipynb` and run the cells sequentially to load the data, explore the dataset, and generate recommendation artifacts.

The notebook performs the following steps:
- Loads the raw CSV files into Pandas DataFrames and cleans null or duplicate rows.
- Constructs a popularity-based recommendation list by combining the number of ratings with their mean value.
- Builds a user–item matrix filtered to active users and frequently rated books, then computes cosine similarity to recommend similar titles.
- Saves intermediate results and models as pickle files for reuse outside the notebook.

### Using the Pickled Artifacts

You can load the generated pickle files in your own Python scripts to serve recommendations without rerunning the entire notebook:

```python
import pickle
import numpy as np
import pandas as pd

popular_df = pickle.load(open("popular.pkl", "rb"))
pt = pickle.load(open("pt.pkl", "rb"))
books = pickle.load(open("books.pkl", "rb"))
similarity_scores = pickle.load(open("similarity_scores.pkl", "rb"))

# Example: recommend books similar to a given title
book_name = "1984"
index = np.where(pt.index == book_name)[0][0]
similar_items = sorted(
    list(enumerate(similarity_scores[index])),
    key=lambda x: x[1],
    reverse=True
)[1:11]
recommendations = []
for i in similar_items:
    temp = books[books['Book-Title'] == pt.index[i[0]]].drop_duplicates('Book-Title')
    recommendations.append(
        {
            "title": temp['Book-Title'].values[0],
            "author": temp['Book-Author'].values[0],
            "cover": temp['Image-URL-M'].values[0],
        }
    )
print(recommendations)
```

## Data Source and Licensing

The dataset originates from the public Book-Crossing data dump. Please review the dataset's original licensing terms before distributing or deploying derived models.

## Next Steps

Potential improvements include adding implicit feedback handling, incorporating content-based features (e.g., genres or descriptions), and wrapping the recommendation logic into an API or web application for interactive use.

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os

# Data source
"https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks"

# Load the dataset
df = pd.read_csv('books.csv', on_bad_lines='skip')

# Data Preparation
df = df[(df['language_code'] == 'eng') | (df['language_code'] == 'en-US')] # filter english only
df = df.drop(columns=['bookID','isbn','isbn13','language_code']) # remove some unnecessary columns
df = df.astype(str) # convert all columns to string

# Setup the embedding and stored vector location
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content = row['title'] + " " + row['authors'] + " " + row['average_rating'] + " " + row['ratings_count'] + " " + row['publication_date'],
            metadata = {'publisher': row['publisher'], 'text_reviews_count': row['text_reviews_count']},
            id = str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(collection_name='books_rating',
                      persist_directory=db_location,
                        embedding_function=embeddings
                        )

# Splitting the list to avoid error while adding documents to the vector database
def split_list(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]
        
try:
    split_docs_chunked = split_list(documents, 5000)
except:
    pass

if add_documents:
    for split_docs_chunk in split_docs_chunked:
        vector_store.add_documents(documents=split_docs_chunk,ids=ids)

# Setup retriever to retrieve top 10 vector similarity result
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

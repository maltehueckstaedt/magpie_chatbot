from langchain_community.document_loaders import DuckDBLoader
from langchain_community.vectorstores import Chroma
import duckdb
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# Create loader
loader = DuckDBLoader(
    database="magpie.db",
    query="SELECT * FROM mview_daten_beschr LIMIT 10"
)

data = loader.load()

# Initialize the SentenceTransformer model using LangChain's wrapper
embedding_model = SentenceTransformerEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')

# Create the Chroma vector database
persist_directory = "./"  # Save in the same directory as the script

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model  # Pass the entire embedding model
)

# Load documents into the vector store
docs = [doc.page_content for doc in data]  # Use the entire page_content as document
vectordb.add_texts(docs)

# Initialize the Hugging Face QA pipeline
qa_model = pipeline('question-answering', model='deepset/roberta-base-squad2')


def perform_qa(query, vectordb, qa_model):
    # Retrieve the most relevant document from the vector store
    results = vectordb.similarity_search(query, k=1)
    
    # Access the text attribute of the first result document
    context = results[0].page_content if results else ''
    
    # Perform QA using the Hugging Face pipeline
    answer = qa_model(question=query, context=context)
    return answer

# Example query
query = "Wie viel FuE_personal war 2011 im Wirtschaftszweig Herstellung von DV-Geräten, elektronischen und optischen Erzeignissen beschäftigt?"
result = perform_qa(query, vectordb, qa_model)
print(result)
print(docs)
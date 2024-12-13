
from langchain_community.document_loaders import PyPDFLoader

# from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain_community.vectorstores import Chroma
# file_path=r"C:\Users\VAIO\Desktop\DSC\RNN\ChatBot\book\co_salient.pdf"
# loader=PyPDFLoader(file_path)
# pages=loader.load_and_split()
# print (len(pages))

# embedding_functions=SentenceTransformerEmbeddings(
#     model_name="all-miniLM-L6-v2"
# )

# vectordb=Chroma.from_documents(
#     documents=pages,
#     embedding=embedding_functions,
#     persist_directory="..\J1J2",
#     collection_name="Good Paper"
# )
# vectordb.persist()
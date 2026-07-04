import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters             import RecursiveCharacterTextSplitter
from langchain_community.embeddings       import OllamaEmbeddings
from langchain_chroma                     import Chroma

CHROMA_PATH     = './chroma_data'
FILE_PATH       = 'content.txt'
COLLECTION_NAME = 'new_knowledge_1'

def init_vector_db():
    if not os.path.exists(FILE_PATH):
        print(f"Error: {FILE_PATH} not found! Put some text in it first.")
        return
    
    print("---Langchain Local Ingestion Pipeline Started ---")

    '''
    LOAD THE TEXT INSIDE THE 'context.txt':
    '''
    loader = TextLoader(FILE_PATH, encoding="utf-8")
    '''
    LOADS THE CONTENT INTO A STANDARDIZED OBJECT INSEAD OF A DUMB STRING:
    '''
    raw_documents = loader.load()

    '''
    USE A SMART RECURSIVE SPLITTER INSTEAD OF RAW 'next' SPLITTING:
    CHUNK SIZE SHOULD BE 1000 AND THE OVERLAPPING SHOULD BE 200.
    (oVERLAPPING: LAST 200 CHARS OF C1 IS THE FIRST 200 CHARS OF C2)
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function=len
    )

    '''
    NOW, USE THAT TEXT_SPLITTER TO SPLIT THE 'RAW_DOCUMENTS':
    '''
    chunks = text_splitter.split_documents(raw_documents)
    '''
    PRINTS HOW MANY SMART ASS CHUNKS WERE CREATED:
    '''
    print(f"[LangChain]: Split text into {len(chunks)} intelligent chunks.")

    '''
    NOW WE USE THE EMBEDDING MODEL TO CREATE VECTOR EMBEDDINGS:
    '''
    embeddings = OllamaEmbeddings(model='nomic-embed-text')

    '''
    IF THE FOLDER AT CHROMA_PATH WITH OLD SHIT EXISTS, DELETE THE VECTOR DB:
    CREATE A NEW ONE:
    '''
    if os.path.exists(CHROMA_PATH):
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )

        db.delete_collection()
        print("[Chroma]: Dropped old collection indices.")

    print("[Chroma]: Generating coordinates and saving to disk...")

    '''
    THIS IS THE CREATING PART:
    '''
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME
    )

    print(f"Success! Local Vector DB initialized using Nomic embeddings.")


def query_vector_db(user_query:str, num_results:int = 1) -> str:
    """
    
    """
    embeddings = OllamaEmbeddings(model='nomic-embed-text')

    db = Chroma(
        persist_directory= CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    results = db.similarity_search(user_query, k=num_results)

    if results:
        return "\n\n".join([doc.page_content for doc in results])
    
    return "No matching context found."

if __name__ == "main":
    init_vector_db()
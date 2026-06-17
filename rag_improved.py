import os
from dotenv import load_dotenv

load_dotenv()

# ── 1. Load Document ──────────────────────────────────────
def load_document(file):
    name, extension = os.path.splitext(file)
    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Unsupported format.')
        return None
    return loader.load()

# ── 2. Chunk ──────────────────────────────────────────────
def chunk_data(data, chunk_size=256):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.15)
    )
    return splitter.split_documents(data)

# ── 3. Embed and Store ────────────────────────────────────
def create_embeddings_chroma(chunks, persist_directory='./chroma_db'):
    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model='models/gemini-embedding-001',
        google_api_key=os.environ["GEMINI_API_KEY"]
    )
    return Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)

# ── 4. Retrieve with Similarity Threshold ─────────────────
def retrieve_with_threshold(vector_store, query, k=5, threshold=0.45):
    results = vector_store.similarity_search_with_score(query, k=k)
    if not results:
        return None, 0.0
    # ChromaDB returns cosine distance; convert to similarity
    filtered = [(doc, 1 - score) for doc, score in results if (1 - score) >= threshold]
    if not filtered:
        max_sim = max(1 - score for _, score in results)
        return None, round(max_sim, 4)
    docs = [doc for doc, _ in filtered]
    max_sim = max(sim for _, sim in filtered)
    return docs, round(max_sim, 4)

# ── 5. Grounded System Prompt ─────────────────────────────
SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on the provided document context.

Rules:
1. Answer ONLY from the context below. Do not use external knowledge.
2. If the context does not contain enough information, respond exactly with: "I don't have that information in the source documents."
3. If context is partially relevant, answer only what you can support and acknowledge what is missing.

Context:
{context}"""

# ── 6. Ask with Memory ────────────────────────────────────
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []

def ask_with_memory(vector_store, question):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnableLambda
    from langchain_core.output_parsers import StrOutputParser

    docs, max_sim = retrieve_with_threshold(vector_store, question)

    print(f"[similarity: {max_sim}]")

    if docs is None:
        response = "I don't have that information in the source documents."
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response))
        return response

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0.0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT.format(context=context)),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))
    return response


# ── 7. Main ───────────────────────────────────────────────
if __name__ == "__main__":
    pdf_path = input("Enter path to your PDF file: ").strip()

    print("Loading document...")
    data = load_document(pdf_path)
    print(f"Pages loaded: {len(data)}")

    print("Chunking...")
    chunks = chunk_data(data)
    print(f"Chunks created: {len(chunks)}")

    print("Creating embeddings (this may take a minute)...")
    vector_store = create_embeddings_chroma(chunks)
    print("Embeddings stored in ChromaDB.")

    print("\nAsk anything about the document. Type 'exit' to quit.\n")
    while True:
        query = input("Your question: ").strip()
        if query.lower() in ['exit', 'quit', 'bye']:
            print("Bye!")
            break
        answer = ask_with_memory(vector_store, query)
        print(f"\nAnswer: {answer}")
        print("-" * 50)
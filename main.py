
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Initialize the SAME embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load the existing vector database
vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

# 3. Create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 15,
        "lambda_mult": 0.5
    }
)

# 4. Initialize Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

# 5. Create prompt template
template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
    Use ONLY the provided context to answer the question.
    If the answer is not present in the context, say:
    "I could not find the answer in the document."
    """),
    ("human", """Context:
    {context}

    Question: {question}""")
])

print("RAG system created!")
print("Ask questions about your ASP.NET book.")
print("Press 0 to exit.")

# 6. Chat loop
while True:
    query = input("\nYou: ")

    if query == "0":
        break

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(query)

    # Combine retrieved chunks into context
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Format prompt
    final_prompt = template.invoke({
        "context": context,
        "question": query
    })

    # Generate answer
    response = llm.invoke(final_prompt)

    answer = response.content

    if isinstance(answer, list):
        answer = "\n".join(
            item["text"]
            for item in answer
            if isinstance(item, dict) and item.get("type") == "text"
        )

    print("\nAI:")
    print(answer)

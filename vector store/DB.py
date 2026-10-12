from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

docs = [
     Document(
        page_content="ASP.NET is a framework for building web applications.",
        metadata={"source": "ASP.NET book", "chapter": "Introduction"}
    ),
    Document(
        page_content="The page life cycle defines the sequence of events in a Web Forms page.",
        metadata={"source": "ASP.NET book", "chapter": "Page Life Cycle"}
    ),
    Document(
    page_content="ASP.NET Web Forms uses server controls to create interactive web pages and handle user input.",
    metadata={"source": "ASP.NET book", "chapter": "Web Forms"}
)
]

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)


result = vectorstore.similarity_search("What do ASP.NET uses for server controls",k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriver = vectorstore.as_retriever()

docs = retriver.invoke("Explain webforms in ASP.NET?")

for d in docs:
    print(d.page_content)
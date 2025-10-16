import requests
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import os

# Récupération des données depuis ton backend Spring Boot
products = requests.get("http://localhost:8080/api/products").json()
print(products)

# Conversion des données en documents LangChain
documents = []

for p in products:
    text = f"Produit : {p['name']}, prix : {p['price']}, description : {p['description']}"
    documents.append(Document(page_content=text, metadata={"type": "product", "id": p["id"]}))

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Conversion des documents en vecteurs
embeddings = [embedding_model.embed_query(doc.page_content) for doc in documents]
print("✅ Embeddings créés :", len(embeddings))

vectorstore = FAISS.from_documents(documents, embedding_model)
print("✅ Base vectorielle FAISS créée avec succès.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


query = "Quels produits coûtent 999 dirhams ?"
result = rag_chain.invoke({"query": query})

print("\n💬 Réponse générée :")
print(result["result"])

print("\n📚 Documents utilisés :")
for doc in result["source_documents"]:
    print("-", doc.page_content)
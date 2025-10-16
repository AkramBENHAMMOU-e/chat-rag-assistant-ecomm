import requests
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Configuration depuis les variables d'environnement
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# Vérifier que la clé API est présente
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY n'est pas définie dans les variables d'environnement")

print("🚀 Démarrage de l'assistant RAG...")

# Récupération des données depuis le backend Spring Boot
try:
    products = requests.get(f"{BACKEND_URL}/api/products").json()
    print(f"✅ {len(products)} produits récupérés depuis le backend")
except Exception as e:
    print(f"❌ Erreur lors de la récupération des produits: {e}")
    exit(1)

# Conversion des données en documents LangChain
documents = []
for p in products:
    text = f"Produit : {p['name']}, prix : {p['price']}, description : {p['description']}"
    documents.append(Document(page_content=text, metadata={"type": "product", "id": p["id"]}))

print(f"📄 {len(documents)} documents créés")

# Initialisation du modèle d'embedding
print("🔄 Initialisation du modèle d'embedding...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Conversion des documents en vecteurs
print("🔄 Création de la base vectorielle...")
vectorstore = FAISS.from_documents(documents, embedding_model)
print("✅ Base vectorielle FAISS créée avec succès")

# Initialisation du LLM
print("🔄 Initialisation du modèle de langage...")
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GOOGLE_API_KEY
)

# Configuration du retriever et de la chaîne RAG
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

print("✅ Assistant RAG prêt !")

# Exemple d'utilisation
def ask_question(question: str):
    """Pose une question à l'assistant RAG"""
    print(f"\n❓ Question: {question}")
    result = rag_chain.invoke({"query": question})
    
    print("\n💬 Réponse générée :")
    print(result["result"])
    
    print("\n📚 Documents utilisés :")
    for doc in result["source_documents"]:
        print("-", doc.page_content)
    
    return result

# Test avec quelques questions
if __name__ == "__main__":
    questions = [
        "Quels produits coûtent 999 dirhams ?",
        "Quels sont les produits les moins chers ?",
        "Trouve-moi des produits de café premium"
    ]
    
    for question in questions:
        ask_question(question)
        print("\n" + "="*50 + "\n")
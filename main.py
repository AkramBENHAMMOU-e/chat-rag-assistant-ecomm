import requests
import os
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Configuration depuis les variables d'environnement
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash") # gemini-1.5-flash est un bon choix

# Configuration PostgreSQL avec pgvector
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "ecommerceInt")
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin123")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "products_reviews")

# Construire la connection string pour PostgreSQL
CONNECTION_STRING = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Vérifier que la clé API est présente
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY n'est pas définie dans les variables d'environnement")

print("🚀 Démarrage de l'assistant RAG...")

# 1. Récupération des données depuis le backend Spring Boot
try:
    products_response = requests.get(f"{BACKEND_URL}/api/products/with-stats")
    products_response.raise_for_status()
    products = products_response.json()
    print(f"✅ {len(products)} produits récupérés depuis le backend")
    
    # Récupération des reviews pour enrichir les données
    try:
        reviews_response = requests.get(f"{BACKEND_URL}/api/reviews")
        reviews_response.raise_for_status()
        reviews = reviews_response.json()
        print(f"✅ {len(reviews)} reviews récupérées depuis le backend")
    except Exception as reviews_error:
        print(f"⚠️ Aucune review trouvée: {reviews_error}")
        reviews = []
        
except Exception as e:
    print(f"❌ Erreur lors de la récupération des données: {e}")
    print(f"Vérifiez que le backend Spring Boot est démarré sur {BACKEND_URL}")
    exit(1)

# 2. Conversion des données en documents LangChain
documents = []
product_count = 0
review_count = 0

# Créer des documents pour chaque produit
for p in products:
    # Formatage amélioré du contenu pour le LLM (avec sauts de ligne)
    text_parts = [
        f"Produit : {p.get('name', 'N/A')}",
        f"Prix : {p.get('price', 'N/A')} dirhams",
        f"Description : {p.get('description', 'N/A')}"
    ]
    if p.get('brand'): text_parts.append(f"Marque : {p['brand']}")
    if p.get('origin'): text_parts.append(f"Origine : {p['origin']}")
    if p.get('roastLevel'): text_parts.append(f"Torréfaction : {p['roastLevel']}")
    if p.get('tastingNotes'): text_parts.append(f"Notes de dégustation : {p['tastingNotes']}")
    if p.get('quantity'): text_parts.append(f"Stock : {p['quantity']} unités")
    if p.get('averageRating'): text_parts.append(f"Note moyenne : {p['averageRating']}/5")
    
    text = "\n".join(text_parts)
    
    # Métadonnées améliorées pour le SelfQueryRetriever
    metadata = {
        "type": "product",
        "id": p.get("id"),
        "name": p.get("name"),
    }
    # Ajout des champs filtrables (importants !)
    try: metadata["price"] = float(p.get("price"))
    except (ValueError, TypeError): pass
    try: metadata["averageRating"] = float(p.get("averageRating"))
    except (ValueError, TypeError): pass
    if p.get("category"): metadata["category"] = p.get("category")
    if p.get("brand"): metadata["brand"] = p.get("brand")
    if p.get("origin"): metadata["origin"] = p.get("origin")
    
    documents.append(Document(page_content=text, metadata=metadata))
    product_count += 1

# Ajouter les reviews comme documents séparés
for review in reviews:
    if review.get('comment') and review.get('isVisible', True):
        product_name = "produit"
        for p in products:
            if p.get('id') == review.get('productId'):
                product_name = p.get('name', 'produit')
                break
        
        review_text = f"Avis sur {product_name} : Note {review.get('rating', 'N/A')}/5 - {review['comment']}"
        review_text += f" - Client : {review.get('customerName', 'Anonyme')}"
        if review.get('isVerified'):
            review_text += " (Achat vérifié)"
            
        # Métadonnées pour les reviews
        review_metadata = {
            "type": "review", 
            "productId": review.get('productId'),
            "productName": product_name,
            "customerName": review.get('customerName', 'Anonyme'),
        }
        try: review_metadata["rating"] = int(review.get("rating"))
        except (ValueError, TypeError): pass
        
        documents.append(Document(
            page_content=review_text, 
            metadata=review_metadata
        ))
        review_count += 1

print(f"📄 {len(documents)} documents créés ({product_count} produits + {review_count} reviews)")

# 3. Initialisation du modèle d'embedding
print("🔄 Initialisation du modèle d'embedding...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# 4. Conversion des documents en vecteurs avec pgvector
print("🔄 Création de la base vectorielle PostgreSQL avec pgvector...")
print(f"📊 Connexion à PostgreSQL: {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")

try:
    vectorstore = PGVector.from_documents(
        documents=documents,
        embedding=embedding_model,
        connection_string=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
        pre_delete_collection=True  # Supprimer la collection existante si elle existe
    )
    print("✅ Base vectorielle PostgreSQL (pgvector) créée avec succès")
except Exception as e:
    print(f"❌ Erreur lors de la création de la base vectorielle: {e}")
    print("\n💡 Vérifiez que:")
    print("   1. PostgreSQL est installé et démarré")
    print("   2. L'extension pgvector est installée: CREATE EXTENSION vector;")
    print("   3. Les identifiants PostgreSQL dans .env sont corrects")
    exit(1)

# 5. Initialisation du LLM
print("🔄 Initialisation du modèle de langage...")
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GOOGLE_API_KEY
)

# 6. Configuration du SelfQueryRetriever
print("🔄 Configuration du SelfQueryRetriever...")

# Définir les champs de métadonnées que le LLM peut utiliser pour filtrer
metadata_field_info = [
    AttributeInfo(
        name="type",
        description="Le type de document, 'product' pour un produit ou 'review' pour un avis",
        type="string",
    ),
    AttributeInfo(
        name="name",
        description="Le nom du produit",
        type="string",
    ),
    AttributeInfo(
        name="price",
        description="Le prix du produit en dirhams",
        type="float",
    ),
    AttributeInfo(
        name="category",
        description="La catégorie du produit (ex: 'Café en grains', 'Machine', 'Accessoire')",
        type="string",
    ),
    AttributeInfo(
        name="brand",
        description="La marque du produit",
        type="string",
    ),
    AttributeInfo(
        name="averageRating",
        description="La note moyenne d'un produit, de 0 à 5",
        type="float",
    ),
    AttributeInfo(
        name="origin",
        description="Le pays ou la région d'origine du café",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="La note donnée dans un avis (review), de 1 à 5",
        type="integer",
    ),
]
document_content_description = "Informations textuelles sur un produit de café ou un avis client."

# Créer le SelfQueryRetriever avec ChromaDB (supporte les requêtes complexes !)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
    search_kwargs={"k": 10}
)

print("✅ SelfQueryRetriever configuré avec PostgreSQL (pgvector) !")

# 7. Configuration de la chaîne RAG
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever, # Utilisation du nouveau retriever
    return_source_documents=True
)

print("✅ Assistant RAG prêt !")

# Exemple d'utilisation
def ask_question(question: str):
    """Pose une question à l'assistant RAG"""
    print(f"\n❓ Question: {question}")
    
    # Invoquer la chaîne
    result = rag_chain.invoke({"query": question})
    
    print("\n💬 Réponse générée :")
    print(result["result"])
    
    print("\n📚 Documents utilisés :")
    if result["source_documents"]:
        for doc in result["source_documents"]:
            print(f"--- Document (Type: {doc.metadata.get('type', 'N/A')}, Nom: {doc.metadata.get('name', 'N/A')}) ---")
            print(doc.page_content)
            print("-" * (len(doc.page_content.split('\n')[0]) + 2))
    else:
        print("Aucun document n'a été utilisé.")
    
    return result

# Test avec quelques questions
if __name__ == "__main__":
    questions = [
        # La question originale, qui fonctionnera maintenant
        "Quels produits coûtent moins de 1000 dirhams ?", 
        
        # Exemples de questions plus complexes
        "Liste les cafés d'origine 'Ethiopia' avec une note moyenne supérieure à 4",
        "Y a-t-il des avis négatifs (note inférieure à 3) ?",
        "Quels sont les produits de la marque 'Bialetti' ?"
    ]
    
    for question in questions:
        ask_question(question)
        print("\n" + "="*50 + "\n")
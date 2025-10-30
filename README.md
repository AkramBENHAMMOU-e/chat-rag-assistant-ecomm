# 🧠 RAG Assistant - Service d'Assistance Intelligente

Service FastAPI qui implémente un système RAG (Retrieval-Augmented Generation) pour répondre aux questions sur les produits d'e-commerce en utilisant LangChain, PostgreSQL/pgvector et Google Gemini.

## 📋 Table des Matières

- [Présentation](#présentation)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Démarrage](#démarrage)
- [Utilisation](#utilisation)
- [API Documentation](#api-documentation)
- [Tests](#tests)
- [Dépannage](#dépannage)

## 🎯 Présentation

Le RAG Assistant est un service qui permet de répondre intelligemment aux questions des utilisateurs concernant les produits, en combinant :
- **Recherche vectorielle** dans PostgreSQL/pgvector
- **Intelligence artificielle** via Google Gemini
- **Analyse de contexte** des produits et avis clients

### Fonctionnalités

- ✅ Réponses basées sur les données réelles des produits
- ✅ Analyse des avis clients pour contexte enrichi
- ✅ Recherche vectorielle optimisée avec pgvector
- ✅ Support du filtrage par métadonnées (prix, catégorie, etc.)
- ✅ API RESTful avec FastAPI
- ✅ Documentation interactive Swagger/OpenAPI

## 🏗️ Architecture

```
┌─────────────────┐
│   Client/User   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│  Spring Boot    │─────▶│  FastAPI RAG    │
│   (Backend)     │      │   (Service)     │
└─────────────────┘      └────────┬────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
         ┌──────────────┐  ┌──────────┐  ┌──────────┐
         │ PostgreSQL   │  │ Hugging  │  │ Google   │
         │ + pgvector   │  │   Face   │  │  Gemini  │
         └──────────────┘  └──────────┘  └──────────┘
```

## 📦 Prérequis

- **Python** 3.9 ou supérieur
- **PostgreSQL** 12+ avec l'extension pgvector
- **Google Gemini API Key** (gratuite sur https://makersuite.google.com/app/apikey)
- **Backend Spring Boot** démarré sur port 8080

## 🔧 Installation

### 1. Cloner le dépôt

```bash
cd rag-assistant
```

### 2. Créer l'environnement virtuel

```bash
python -m venv rag_env
rag_env\Scripts\activate

```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configuration

Copiez le fichier `.env.example` vers `.env` :

```bash
# Windows
copy .env.example .env
```

Puis éditez `.env` avec vos paramètres :

```env
# Backend Spring Boot URL
BACKEND_URL=http://localhost:8080

# Google Gemini API Key (REQUIRED)
GOOGLE_API_KEY=your_api_key_here

# Configuration PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ecommerceInt
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin123

# Port FastAPI
PORT=8081
```

### 5. Configurer PostgreSQL

Connectez-vous à PostgreSQL et créez l'extension pgvector :

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## 🚀 Démarrage

### Développement

```bash
uvicorn app:app --host 0.0.0.0 --port 8081 --reload
```

### Production

```bash
uvicorn app:app --host 0.0.0.0 --port 8081
```

Le service sera accessible sur : http://localhost:8081

## 📖 Utilisation

### Test de santé

```bash
curl http://localhost:8081/health
```

Réponse attendue :
```json
{
  "status": "healthy"
}
```

### Test d'initialisation

```bash
curl http://localhost:8081/
```

Réponse attendue :
```json
{
  "status": "ok",
  "message": "RAG Assistant API is running",
  "rag_initialized": true
}
```

### Poser une question

```bash
curl -X POST http://localhost:8081/rag-query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels sont les meilleurs cafés en grains ?"}'
```

Réponse :
```json
{
  "answer": "Basé sur notre catalogue, nous recommandons..."
}
```

## 📚 API Documentation

Une fois le service démarré, accédez à :
- **Swagger UI** : http://localhost:8081/docs
- **ReDoc** : http://localhost:8081/redoc
- **OpenAPI JSON** : http://localhost:8081/openapi.json

## 🧪 Tests

Exécutez les tests depuis la racine du projet :

```bash
# Test Python simple
python test_rag.py

# Test PowerShell
powershell -ExecutionPolicy Bypass -File test_rag.ps1
```

## 🔍 Dépannage

### Erreur : GOOGLE_API_KEY not defined

**Solution :** Vérifiez que votre fichier `.env` contient une clé API valide.

### Erreur : PostgreSQL connection failed

**Solution :** 
1. Vérifiez que PostgreSQL est démarré
2. Vérifiez les identifiants dans `.env`
3. Vérifiez que l'extension pgvector est installée

### Erreur : Cannot connect to backend

**Solution :** 
1. Vérifiez que Spring Boot est démarré sur http://localhost:8080
2. Vérifiez la valeur de `BACKEND_URL` dans `.env`

### Slow performance

**Solution :**
1. Augmentez les ressources PostgreSQL
2. Optimisez les index de la base vectorielle
3. Utilisez un modèle d'embedding plus léger

## 🛠️ Développement

### Structure du Projet

```
rag-assistant/
├── app.py                 # Application FastAPI principale
├── main.py               # Script d'initialisation (optionnel)
├── requirements.txt      # Dépendances Python
├── .env                  # Variables d'environnement (NE PAS COMMITTER)
├── .env.example          # Template de configuration
├── .gitignore           # Fichiers ignorés par Git
├── README.md            # Documentation
└── rag_env/             # Environnement virtuel Python (IGNORÉ)
```

### Ajout de nouvelles fonctionnalités

1. Modifiez `app.py` pour ajouter de nouveaux endpoints
2. Testez localement avec `--reload`
3. Documentez dans ce README

## 📝 Variables d'Environnement

| Variable | Description | Requis | Défaut |
|----------|-------------|--------|---------|
| `BACKEND_URL` | URL du backend Spring Boot | Non | http://localhost:8080 |
| `GOOGLE_API_KEY` | Clé API Google Gemini | **Oui** | - |
| `EMBEDDING_MODEL` | Modèle d'embedding HuggingFace | Non | all-MiniLM-L6-v2 |
| `LLM_MODEL` | Modèle LLM Gemini | Non | gemini-2.5-flash |
| `POSTGRES_HOST` | Hôte PostgreSQL | Non | localhost |
| `POSTGRES_PORT` | Port PostgreSQL | Non | 5432 |
| `POSTGRES_DB` | Nom de la base de données | Non | ecommerceInt |
| `POSTGRES_USER` | Utilisateur PostgreSQL | Non | admin |
| `POSTGRES_PASSWORD` | Mot de passe PostgreSQL | Non | admin123 |
| `COLLECTION_NAME` | Nom de la collection vectorielle | Non | products_reviews |
| `PORT` | Port FastAPI | Non | 8081 |

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche feature
3. Faire un commit de vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence [MIT](LICENSE).

## 👤 Auteur

**Votre Nom**  
- Email: votre.email@example.com
- GitHub: [@votre-username](https://github.com/votre-username)

## 🙏 Remerciements

- [FastAPI](https://fastapi.tiangolo.com/) - Framework web moderne
- [LangChain](https://www.langchain.com/) - Framework LLM
- [pgvector](https://github.com/pgvector/pgvector) - Extension PostgreSQL pour vecteurs
- [Google Gemini](https://gemini.google.com/) - Modèle de langage
- [Hugging Face](https://huggingface.co/) - Modèles d'embedding

---

**Dernière mise à jour** : 2025-10-30  
**Version** : 1.0.0


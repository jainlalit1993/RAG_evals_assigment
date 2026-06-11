import chromadb
from chromadb.config import Settings
import yaml
from colorama import Fore, Style, init

init(autoreset=True)

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Connect to ChromaDB
client = chromadb.PersistentClient(path=config['vectordb']['persist_directory'])
collection = client.get_collection(name=config['vectordb']['collection_name'])

print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
print(f"{Fore.CYAN}🔍 CHROMADB DATABASE INSPECTOR{Style.RESET_ALL}")
print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

# Get collection info
count = collection.count()
print(f"{Fore.YELLOW}📊 Collection Stats:{Style.RESET_ALL}")
print(f"   • Collection name: {config['vectordb']['collection_name']}")
print(f"   • Total vectors: {count}")
print(f"   • Storage path: {config['vectordb']['persist_directory']}\n")

# Sample some documents
print(f"{Fore.YELLOW}📄 Sample Documents (first 5):{Style.RESET_ALL}\n")
results = collection.get(limit=5, include=['documents', 'metadatas'])

for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas']), 1):
    print(f"{Fore.CYAN}Document {i}:{Style.RESET_ALL}")
    print(f"   Source: {meta.get('source', 'N/A')}")
    print(f"   Page: {meta.get('page', 'N/A')}")
    print(f"   Content preview: {doc[:150]}...")
    print()

# Show unique sources
all_data = collection.get(include=['metadatas'])
sources = set(meta.get('source', 'Unknown') for meta in all_data['metadatas'])

print(f"{Fore.YELLOW}📚 Unique Source Documents:{Style.RESET_ALL}")
for source in sorted(sources):
    source_count = sum(1 for meta in all_data['metadatas'] if meta.get('source') == source)
    print(f"   • {source}: {source_count} chunks")

print()
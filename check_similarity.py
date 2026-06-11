import sys
sys.path.insert(0, '.')

from app.agent import RAGAgent
from colorama import Fore, Style, init

init(autoreset=True)

agent = RAGAgent()

question = input(f"{Fore.GREEN}Enter a question: {Style.RESET_ALL}")

# Get results with scores
results = agent.vectorstore.similarity_search_with_score(question, k=5)

print(f"\n{Fore.CYAN}🎯 Top 5 Similar Chunks:{Style.RESET_ALL}\n")

for i, (doc, score) in enumerate(results, 1):
    print(f"{Fore.YELLOW}Rank {i} - Similarity: {1-score:.3f}{Style.RESET_ALL}")
    print(f"   Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
    print(f"   Content: {doc.page_content[:200]}...")
    print()
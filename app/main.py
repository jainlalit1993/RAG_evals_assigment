
"""
Main entry point for the RAG Fitness Agent
Interactive CLI for asking questions
"""
import sys
from colorama import Fore, Style, init
from app.agent import RAGAgent

# Initialize colorama
init(autoreset=True)

def print_banner():
    """Print welcome banner"""
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}� Healthcare Insurance Policy Assistant - Interactive Mode{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    print(f"{Fore.YELLOW}Ask questions about your healthcare insurance policy documents!{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Type 'exit' or 'quit' to stop{Style.RESET_ALL}\n")

def format_sources(sources):
    """Format source citations"""
    if not sources:
        return ""
    
    unique_sources = {}
    for src in sources:
        key = f"{src['source']}"
        if key not in unique_sources:
            unique_sources[key] = []
        unique_sources[key].append(src['page'])
    
    formatted = []
    for source, pages in unique_sources.items():
        pages_str = ", ".join(map(str, sorted(set(pages))))
        formatted.append(f"📄 {source} (pages: {pages_str})")
    
    return "\n".join(formatted)

def main():
    """Main interactive loop"""
    print_banner()
    
    # Initialize agent
    print(f"{Fore.YELLOW}🔧 Loading RAG agent...{Style.RESET_ALL}")
    try:
        agent = RAGAgent()
        print(f"{Fore.GREEN}✅ Agent ready!{Style.RESET_ALL}\n")
    except Exception as e:
        print(f"{Fore.RED}❌ Error initializing agent: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💡 Make sure you've run: python app/ingest.py{Style.RESET_ALL}")
        sys.exit(1)
    
    # Interactive loop
    while True:
        # Get user input
        print(f"{Fore.CYAN}{'─'*60}{Style.RESET_ALL}")
        question = input(f"{Fore.GREEN}❓ Your question: {Style.RESET_ALL}").strip()
        
        # Check for exit
        if question.lower() in ['exit', 'quit', 'q']:
            print(f"\n{Fore.CYAN}👋 Thanks for using RAG Fitness Agent!{Style.RESET_ALL}\n")
            break
        
        # Skip empty questions
        if not question:
            continue
        
        # Query the agent
        print(f"\n{Fore.YELLOW}🤔 Thinking...{Style.RESET_ALL}\n")
        
        try:
            result = agent.query(question)
            
            # Display answer
            print(f"{Fore.CYAN}💡 Answer:{Style.RESET_ALL}")
            print(f"{Fore.WHITE}{result['answer']}{Style.RESET_ALL}\n")
            
            # Display sources
            if result['sources']:
                print(f"{Fore.CYAN}📚 Sources:{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}{format_sources(result['sources'])}{Style.RESET_ALL}\n")
        
        except Exception as e:
            print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}\n")

if __name__ == "__main__":
    main()

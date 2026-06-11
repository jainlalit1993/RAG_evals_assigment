"""
Compare different chunking strategies side-by-side
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.chunking_strategies import ChunkingStrategy, compare_chunking_strategies
from pypdf import PdfReader
from colorama import Fore, Style, init

init(autoreset=True)

def main():
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}✂️  CHUNKING STRATEGY COMPARISON{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    # Load PDFs and find one with extractable text
    pdf_dir = Path("data/pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"{Fore.RED}❌ No PDF files found{Style.RESET_ALL}")
        return
    
    sample_text = None
    pdf_name = None
    
    # Try each PDF until we find one with text
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages[:3]:  # Try first 3 pages
                text = page.extract_text()
                if text and len(text.strip()) > 100:  # Found text
                    sample_text = text
                    pdf_name = pdf_path.name
                    break
            if sample_text:
                break
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️  Error reading {pdf_path.name}: {e}{Style.RESET_ALL}")
            continue
    
    if not sample_text:
        print(f"{Fore.RED}❌ No PDFs with extractable text found{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💡 Your PDFs might be scanned images. Consider using OCR.{Style.RESET_ALL}")
        return
    
    print(f"{Fore.YELLOW}📄 Analyzing: {pdf_name}{Style.RESET_ALL}\n")
    
    print(f"{Fore.CYAN}📊 Original Text Stats:{Style.RESET_ALL}")
    print(f"   • Total characters: {len(sample_text)}")
    print(f"   • Total words: {len(sample_text.split())}")
    print(f"   • Total lines: {len(sample_text.splitlines())}\n")
    
    # Compare strategies
    results = compare_chunking_strategies(sample_text)
    
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📊 CHUNKING STRATEGY RESULTS{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    for strategy_name, stats in results.items():
        print(f"{Fore.YELLOW}{strategy_name}:{Style.RESET_ALL}")
        print(f"   • Chunks created: {stats['chunks']}")
        print(f"   • Avg chunk size: {stats['avg_size']} characters")
        if stats['first_chunk_preview']:
            print(f"   • First chunk preview:")
            print(f"     {stats['first_chunk_preview']}...\n")
        else:
            print()
    
    # Recommendations
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}💡 RECOMMENDATIONS{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    print(f"{Fore.GREEN}Use 'small_precise' if:{Style.RESET_ALL}")
    print(f"   • You need exact fact extraction")
    print(f"   • Questions are specific (e.g., 'How much protein?')\n")
    
    print(f"{Fore.GREEN}Use 'semantic' (default) if:{Style.RESET_ALL}")
    print(f"   • General purpose RAG")
    print(f"   • Mixed question types\n")
    
    print(f"{Fore.GREEN}Use 'large_context' if:{Style.RESET_ALL}")
    print(f"   • Questions require broad context")
    print(f"   • Documents have complex relationships\n")
    
    print(f"{Fore.GREEN}Use 'paragraph' if:{Style.RESET_ALL}")
    print(f"   • Documents have clear paragraph structure")
    print(f"   • Want minimal redundancy\n")

if __name__ == "__main__":
    main()
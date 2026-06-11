"""
Enhanced Document Ingestion Pipeline with Advanced Chunking Strategies
Supports: Token-based, Semantic, Agentic, and Recursive chunking
"""
import os
import sys
import yaml
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from tqdm import tqdm
from colorama import Fore, Style, init

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.chunking_strategies import AdvancedChunkingStrategies

# Initialize colorama for Windows
init(autoreset=True)

# Load environment variables
load_dotenv()

class DocumentIngestion:
    def __init__(self, config_path: str = "config.yaml", strategy: str = "semantic"):
        """
        Initialize the ingestion pipeline
        
        Args:
            config_path: Path to configuration file
            strategy: Chunking strategy to use
                     Options: token_based, semantic, agentic, recursive
        """
        print(f"{Fore.CYAN}🚀 Initializing Document Ingestion Pipeline{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}   Chunking Strategy: {strategy.upper()}{Style.RESET_ALL}\n")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup paths
        self.doc_dir = Path(self.config.get('documents', {}).get('doc_dir', 'data/docs'))
        self.vectorstore_dir = Path(self.config['vectordb']['persist_directory'])
        
        # Initialize text splitter based on strategy
        self.strategy_name = strategy
        self.text_splitter = self._get_chunking_strategy(strategy)
        
        # Initialize embeddings (local model)
        print(f"{Fore.YELLOW}📥 Loading embedding model: {self.config['embeddings']['model_name']}{Style.RESET_ALL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['embeddings']['model_name']
        )
        print(f"{Fore.GREEN}✅ Embedding model loaded{Style.RESET_ALL}\n")
    
    def _get_chunking_strategy(self, strategy: str):
        """Get the appropriate chunking strategy"""
        chunk_size = self.config['chunking']['chunk_size']
        chunk_overlap = self.config['chunking']['chunk_overlap']
        
        strategies = {
            "token_based": AdvancedChunkingStrategies.token_based_with_overlap(
                chunk_size=512,
                chunk_overlap=128
            ),
            "semantic": AdvancedChunkingStrategies.semantic_chunking(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            ),
            "agentic": AdvancedChunkingStrategies.agentic_chunking(
                max_chunk_size=1500,
                overlap=200
            ),
            "recursive": AdvancedChunkingStrategies.recursive_chunking(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        }
        
        if strategy not in strategies:
            print(f"{Fore.YELLOW}⚠️  Unknown strategy '{strategy}', using 'semantic'{Style.RESET_ALL}")
            return strategies["semantic"]
        
        return strategies[strategy]
    
    def load_documents(self) -> List[Document]:
        """Load all supported documents from the data directory"""
        documents = []
        file_types = [ext.lower() for ext in self.config.get('documents', {}).get('file_types', ['.pdf', '.md', '.txt'])]
        all_files = [p for p in sorted(self.doc_dir.glob('**/*')) if p.suffix.lower() in file_types]

        if not all_files:
            print(f"{Fore.RED}❌ No supported documents found in {self.doc_dir}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}💡 Please add files to: {self.doc_dir.absolute()}{Style.RESET_ALL}")
            return documents

        print(f"{Fore.CYAN}📚 Found {len(all_files)} documents{Style.RESET_ALL}\n")

        for path in tqdm(all_files, desc="Loading documents", colour="green"):
            try:
                if path.suffix.lower() == ".pdf":
                    reader = PdfReader(path)
                    for page_num, page in enumerate(reader.pages, start=1):
                        text = page.extract_text() or ""
                        if text.strip():
                            doc = Document(
                                page_content=text,
                                metadata={
                                    "source": path.name,
                                    "page": page_num,
                                    "total_pages": len(reader.pages)
                                }
                            )
                            documents.append(doc)
                else:
                    text = path.read_text(encoding="utf-8")
                    if text.strip():
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": path.name,
                                "page": 1,
                                "total_pages": 1
                            }
                        )
                        documents.append(doc)
            except Exception as e:
                print(f"{Fore.RED}❌ Error loading {path.name}: {e}{Style.RESET_ALL}")

        print(f"\n{Fore.GREEN}✅ Loaded {len(documents)} document chunks from {len(all_files)} files{Style.RESET_ALL}\n")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        print(f"{Fore.CYAN}✂️  Chunking documents with {self.strategy_name.upper()} strategy...{Style.RESET_ALL}")
        
        if self.strategy_name == "agentic":
            print(f"{Fore.YELLOW}   ⚠️  Agentic chunking uses LLM - this may take longer and use API credits{Style.RESET_ALL}")
        
        chunks = self.text_splitter.split_documents(documents)
        
        print(f"{Fore.GREEN}✅ Created {len(chunks)} chunks{Style.RESET_ALL}")
        chunk_sizes = [len(c.page_content) for c in chunks]
        print(f"{Fore.YELLOW}   Avg chunk size: ~{sum(chunk_sizes) // len(chunk_sizes)} characters{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}   Min chunk size: {min(chunk_sizes)} characters{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}   Max chunk size: {max(chunk_sizes)} characters{Style.RESET_ALL}\n")
        
        return chunks
    
    def create_vectorstore(self, chunks: List[Document]):
        """Create and persist ChromaDB vectorstore"""
        print(f"{Fore.CYAN}🗄️  Creating vector database...{Style.RESET_ALL}")
        
        # Create ChromaDB vectorstore
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.config['vectordb']['collection_name'],
            persist_directory=str(self.vectorstore_dir)
        )
        
        print(f"{Fore.GREEN}✅ Vector database created{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}   Location: {self.vectorstore_dir.absolute()}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}   Collection: {self.config['vectordb']['collection_name']}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}   Total vectors: {len(chunks)}{Style.RESET_ALL}\n")
        
        return vectorstore
    
    def run(self):
        """Run the complete ingestion pipeline"""
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🏋️  RAG FITNESS AGENT - DOCUMENT INGESTION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        # Step 1: Load documents
        documents = self.load_documents()
        if not documents:
            return
        
        # Step 2: Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Step 3: Create vectorstore
        vectorstore = self.create_vectorstore(chunks)
        
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}🎉 INGESTION COMPLETE!{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}\n")
        
        # Show summary
        print(f"{Fore.CYAN}📊 Summary:{Style.RESET_ALL}")
        print(f"   • Strategy used: {self.strategy_name.upper()}")
        print(f"   • PDFs processed: {len(set(d.metadata['source'] for d in documents))}")
        print(f"   • Pages loaded: {len(documents)}")
        print(f"   • Chunks created: {len(chunks)}")
        print(f"   • Vectors stored: {len(chunks)}")
        print(f"\n{Fore.YELLOW}🚀 Ready to answer questions! Run: python -m app.main{Style.RESET_ALL}\n")

if __name__ == "__main__":
    # Get strategy from command line argument
    strategy = sys.argv[1] if len(sys.argv) > 1 else "semantic"
    
    ingestion = DocumentIngestion(strategy=strategy)
    ingestion.run()
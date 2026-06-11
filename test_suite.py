import sys
sys.path.insert(0, '.')

from app.agent import RAGAgent
from colorama import Fore, Style, init

init(autoreset=True)

def run_tests():
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🧪 RAG SYSTEM COMPREHENSIVE TEST SUITE{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    agent = RAGAgent()
    
    # Test cases
    tests = [
        {
            "name": "Factual Question (Should Find Answer)",
            "question": "What are post-workout recovery tips?",
            "expect": "should_answer"
        },
        {
            "name": "Specific Nutrient Query",
            "question": "How much protein in cottage cheese?",
            "expect": "should_answer"
        },
        {
            "name": "Unrelated Question (Should Say 'Don't Know')",
            "question": "How do I build a rocket ship?",
            "expect": "should_not_know"
        },
        {
            "name": "Partial Information Test",
            "question": "What are the benefits of sleep?",
            "expect": "should_answer"
        },
        {
            "name": "Multi-Document Query",
            "question": "Tell me about nutrition and fitness",
            "expect": "should_answer"
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(tests, 1):
        print(f"{Fore.YELLOW}Test {i}: {test['name']}{Style.RESET_ALL}")
        print(f"Question: {test['question']}")
        
        result = agent.query(test['question'])
        answer = result['answer'].lower()
        
        # Check expectations
        if test['expect'] == "should_answer":
            if "don't have" in answer or "not in the provided" in answer:
                print(f"{Fore.RED}❌ FAILED - Should have found answer{Style.RESET_ALL}")
                failed += 1
            else:
                print(f"{Fore.GREEN}✅ PASSED - Found answer{Style.RESET_ALL}")
                print(f"   Answer preview: {result['answer'][:100]}...")
                print(f"   Sources: {len(result['sources'])} documents")
                passed += 1
        
        elif test['expect'] == "should_not_know":
            if "don't have" in answer or "not in the provided" in answer:
                print(f"{Fore.GREEN}✅ PASSED - Correctly said 'don't know'{Style.RESET_ALL}")
                passed += 1
            else:
                print(f"{Fore.RED}❌ FAILED - Should have said 'don't know'{Style.RESET_ALL}")
                failed += 1
        
        print()
    
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📊 TEST SUMMARY{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✅ Passed: {passed}/{len(tests)}{Style.RESET_ALL}")
    print(f"{Fore.RED}❌ Failed: {failed}/{len(tests)}{Style.RESET_ALL}")
    print()

if __name__ == "__main__":
    run_tests()
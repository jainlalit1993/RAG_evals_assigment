# Assignment: Domain-Specific RAG Agent With LangGraph Evaluation

## Project Goal

Build a new RAG-based agent for a completely different knowledge base than the current fitness/nutrition project. Your agent must ingest domain-specific documents, retrieve relevant context, generate answers, evaluate search quality, evaluate answer quality, and use LangGraph retry logic when quality falls below defined thresholds.

Each student must use their assigned use case from the table at the end of this document.

## Learning Outcomes

By completing this project, you will practice:

- Building a RAG agent for a new domain and knowledge base.
- Changing and comparing embedding models.
- Applying different chunking strategies.
- Implementing search evaluation metrics.
- Implementing answer evaluation metrics.
- Designing LangGraph workflows with evaluator nodes, thresholds, and retry paths.
- Reporting measurable improvements using experiments.

## Required Tasks

### 1. Create A New Knowledge Base

Replace the existing fitness/nutrition documents with documents for your assigned use case.

Requirements:

- Use at least 3 documents.
- Documents may be PDFs, Markdown files, text files, or web-exported documentation.
- Your documents must be relevant to your assigned domain.
- Add a short note in your submission explaining what documents you used and why.

Do not use the original fitness/nutrition knowledge base as your final project knowledge base.

### 2. Build A Domain-Specific Agent

Update the agent behavior for your assigned use case.

Your agent should:

- Answer questions only from the provided knowledge base.
- Mention when the answer is not supported by retrieved context.
- Return useful source/context references where possible.
- Be adapted to your domain, not just renamed.

Examples:

- A legal policy assistant should answer using policy/legal documents.
- A travel assistant should answer using travel guides, itineraries, and rules.
- A product support assistant should answer using manuals and FAQs.

### 3. Change The Embedding Model

Change the embedding model used in the project.

You must:

- Replace the current embedding model with another embedding model.
- Document the old model and new model.
- Explain why you selected the new model.
- Compare retrieval quality before and after the change using at least 3 test questions.

Suggested options:

- OpenAI embedding model
- Hugging Face sentence-transformer model
- Google embedding model
- Cohere embedding model
- Any other suitable embedding model supported by your stack

### 4. Change The Chunking Strategy

Implement or configure a different chunking strategy from the original project.

You may use one or more of:

- Fixed-size chunking
- Recursive character chunking
- Semantic chunking
- Markdown/header-based chunking
- Sentence-based chunking
- Parent-child chunking

You must compare at least 2 strategies and report:

- Chunk size
- Chunk overlap
- Number of chunks created
- Retrieval quality difference
- Which strategy you selected for the final agent and why

### 5. Add Search Evaluation

Implement search evaluation as LangGraph node(s). These nodes should evaluate the retrieved documents before answer generation or before final response approval.

Required search metrics:

| Metric | Question Answered | Range | Good Value | Use Case |
| --- | --- | --- | --- | --- |
| MRR | How fast do we find the first relevant document? | 0-1 | >0.8 | First result quality |
| MAP | Overall ranking quality across all relevant documents? | 0-1 | >0.7 | Complete ranking |
| Precision | What percent of retrieved documents are relevant? | 0-1 | >0.8 | Retrieval accuracy |
| Recall | What percent of relevant documents did we find? | 0-1 | >0.7 | Retrieval completeness |
| F1 | Balance of precision and recall? | 0-1 | >0.7 | Overall retrieval |
| NDCG | Ranking quality with position weighting? | 0-1 | >0.8 | Position-aware ranking |

Minimum requirement:

- Implement at least 3 search metrics.
- At least one metric must evaluate ranking order, such as MRR, MAP, or NDCG.

### 6. Add Answer Evaluation

Implement answer evaluation as LangGraph node(s). These nodes should evaluate the generated answer before returning it to the user.

Required answer metrics:

| Metric | Question Answered | Range | Good Value | Use Case |
| --- | --- | --- | --- | --- |
| ROUGE-1 | Word overlap using unigrams? | 0-1 | >0.5 | Basic text matching |
| ROUGE-2 | Phrase overlap using bigrams? | 0-1 | >0.3 | Phrase-level matching |
| ROUGE-L | Longest common subsequence? | 0-1 | >0.4 | Structural similarity |
| Fuzzy | String similarity with typo tolerance? | 0-100 | >70 | Typo-tolerant matching |
| Grounding % | What percent of sources support the answer? | 0-100 | >80% | Source relevance |
| TP % | What percent of claims are supported? | 0-100 | >90% | Factual accuracy |
| FP % | What percent of claims are hallucinated? | 0-100 | <10% | Hallucination detection |
| Relevance | Is the answer direct and helpful? | 0 or 1 | 1 | Answer helpfulness |

Minimum requirement:

- Implement at least 3 answer metrics.
- At least one metric must check grounding or hallucination risk.

### 7. Use LangGraph Nodes With Retry And Threshold Strategy

Your final workflow must use LangGraph.

Minimum graph design:

```text
User Question
    -> Retrieve Documents
    -> Search Evaluation
    -> Generate Answer
    -> Answer Evaluation
    -> Decision Node
        -> If passed: Final Answer
        -> If failed: Retry With Improved Retrieval Or Prompt
```

Required retry behavior:

- Define thresholds for your selected search and answer metrics.
- If search score is below threshold, retry retrieval with one change.
- If answer score is below threshold, retry generation with one change.
- Limit retries to avoid infinite loops.
- Include the final evaluation scores in logs or console output.

Example threshold strategy:

```text
If precision < 0.8:
    increase top_k or rewrite query, then retrieve again

If MRR < 0.8:
    try reranking or query expansion

If grounding < 80:
    regenerate answer with stricter source-only instruction

If FP percentage > 10:
    reject unsupported claims and regenerate
```

### 8. Build A Test Set

Create a small evaluation dataset for your use case.

Minimum:

- 10 test questions.
- Expected answer or reference answer for each question.
- Relevant document IDs or source names for each question.
- At least 2 questions where the answer should not be available in the knowledge base.

You may store this as:

- `evaluation_questions.json`
- `evaluation_questions.yaml`
- `evaluation_questions.csv`
- Markdown table

### 9. Final Report

Submit a short project report in Markdown.

Your report must include:

- Assigned use case.
- Knowledge base description.
- Embedding model comparison.
- Chunking strategy comparison.
- LangGraph workflow diagram or text flow.
- Search evaluation results.
- Answer evaluation results.
- Retry strategy explanation.
- 3 example questions with final answers and evaluation scores.
- Limitations and future improvements.

## Suggested Folder Structure

You may keep the existing project structure, but your final submission should be easy to review.

Suggested structure:

```text
project/
  app/
    agent.py
    graph.py
    ingest.py
    chunking_strategies.py
    evaluation.py
  data/
    docs/
  evaluation_questions.json
  config.yaml
  PROJECT_REPORT.md
  README.md
```

## Submission Checklist

- New domain-specific knowledge base added.
- Agent answers questions for the assigned use case.
- Embedding model changed and compared.
- At least 2 chunking strategies compared.
- LangGraph workflow implemented.
- Search evaluation node added.
- Answer evaluation node added.
- Retry and threshold strategy implemented.
- Evaluation dataset with at least 10 questions added.
- Final Markdown report completed.
- Code runs without errors.

## Grading Rubric

| Area | Marks |
| --- | ---: |
| New knowledge base and domain adaptation | 15 |
| Embedding model change and comparison | 10 |
| Chunking strategy implementation and comparison | 10 |
| LangGraph workflow design | 15 |
| Search evaluation metrics | 15 |
| Answer evaluation metrics | 15 |
| Retry and threshold strategy | 10 |
| Test set and final report quality | 10 |
| Total | 100 |

## Student Use Case Assignment

| Student | Assigned Use Case |
| --- | --- |
| Chan Wei Khjan | University course policy assistant |
| Gurleen Kaur | Mental wellness resource assistant |
| Komal Patil | Personal finance FAQ assistant |
| Anived Mishra | Cybersecurity awareness assistant |
| Lalit Jain | Healthcare insurance policy assistant |
| Gurkamal Singh | HR employee handbook assistant |
| Joseph | Travel guide and itinerary assistant |
| Siddhesh Sawant | Legal rental agreement assistant |
| Karthik Balaje R | Product manual support assistant |
| Sai Sankar | Cloud computing documentation assistant |
| Bala Krishna Yenumula | Agriculture advisory assistant |
| Beadon Roy | Automobile maintenance assistant |
| Sagar Sable | Government scheme information assistant |
| Ankith Dasu | E-commerce return and refund assistant |
| Tilottama Pawar | Academic research paper assistant |
| Mini Yadav | Food safety and nutrition label assistant |
| Purnima Sambasivan | Banking compliance FAQ assistant |
| Jocelyn Jose | Environmental sustainability assistant |

## Example Questions For Students To Adapt

Each student should create their own questions, but these examples show the expected style:

- What does the policy say about eligibility?
- Which section supports this answer?
- What are the required steps for completing this process?
- What exceptions are mentioned in the documents?
- Is this claim supported by the knowledge base?
- What information is missing from the knowledge base?

## Important Rules

- Do not submit only prompt changes. You must modify the pipeline.
- Do not use the same knowledge base as another student.
- Do not skip evaluation. Evaluation is a core part of this assignment.
- Do not return unsupported answers as facts.
- Keep all API keys and secrets out of GitHub.


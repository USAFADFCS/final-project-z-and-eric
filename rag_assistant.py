"""
rag_assistant.py

Minimal multi-agent RAG assistant for CS471-style agentic project.

- DocumentSearchTool: TF-IDF search over a folder of text files.
- QuestionRouterAgent: rewrites query + decides top_k and type.
- RetrievalAgent: calls the search tool.
- AnswerAgent: generates a grounded answer with citations.
- CriticAgent: checks for hallucinations and fixes the answer.
- RAGOrchestrator: glues everything together.

Replace call_llm() with your course's LLM interface if needed.
"""

import os
import json
import glob
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# If you use OpenAI:
# pip install openai python-dotenv
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------------------------
# LLM helper
# ------------------------------

def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Thin wrapper around OpenAI Chat Completions.
    Replace this with your Fair-LLM / local model if needed.
    """
    if OPENAI_API_KEY is None:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def extract_json_block(text: str) -> Dict[str, Any]:
    """
    Extracts the first {...} JSON object from a string and parses it.
    Simple but usually good enough for controlled prompts.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in LLM output: {text}")
    json_str = text[start : end + 1]
    return json.loads(json_str)


# ------------------------------
# Document search tool
# ------------------------------

@dataclass
class DocumentChunk:
    doc_id: str
    chunk_id: int
    text: str
    score: float


class DocumentSearchTool:
    """
    Simple TF-IDF document search over a folder of .txt files.
    You can replace this with embeddings/FAISS if you want.
    """

    def __init__(self, corpus_dir: str, chunk_size: int = 500, chunk_overlap: int = 100):
        self.corpus_dir = corpus_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        self._build_index()

    def _load_documents(self) -> Dict[str, str]:
        docs = {}
        for path in glob.glob(os.path.join(self.corpus_dir, "*.txt")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                docs[os.path.basename(path)] = f.read()
        return docs

    def _split_into_chunks(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Simple sliding-window chunking by characters.
        For more sophistication, you could do sentence-based splitting.
        """
        chunks = []
        start = 0
        chunk_id = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunks.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": chunk_text.replace("\n", " "),
                }
            )
            chunk_id += 1
            start = end - self.chunk_overlap  # overlap
        return chunks

    def _build_index(self):
        docs = self._load_documents()
        all_chunks: List[Dict[str, Any]] = []
        for doc_id, text in docs.items():
            doc_chunks = self._split_into_chunks(text, doc_id)
            all_chunks.extend(doc_chunks)

        self.chunks = all_chunks

        corpus_texts = [c["text"] for c in self.chunks]
        if not corpus_texts:
            raise RuntimeError(f"No .txt files found in corpus_dir: {self.corpus_dir}")

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)
        print(f"[DocumentSearchTool] Indexed {len(self.chunks)} chunks from {len(docs)} docs.")

    def search(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("Index not built.")

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_indices = sims.argsort()[::-1][:top_k]

        results: List[DocumentChunk] = []
        for idx in top_indices:
            c = self.chunks[idx]
            results.append(
                DocumentChunk(
                    doc_id=c["doc_id"],
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    score=float(sims[idx]),
                )
            )
        return results


# ------------------------------
# Agents
# ------------------------------

class QuestionRouterAgent:
    """
    Takes user question, returns:
      - rewritten_query
      - question_type
      - top_k
    """

    SYSTEM_PROMPT = """You are a question analysis agent.
Given a user question, you:
1. Rewrite it into a retrieval-friendly search query.
2. Classify the question type as one of:
   - factual
   - explain
   - compare
   - multi-part
3. Recommend an integer top_k (between 3 and 10) for how many chunks
   to retrieve from the document store.

Respond ONLY with a JSON object with keys:
- rewritten_query (string)
- question_type (string)
- top_k (int)
"""

    def run(self, question: str) -> Dict[str, Any]:
        user_prompt = f"User question:\n{question}\n"
        raw = call_llm(self.SYSTEM_PROMPT, user_prompt)
        return extract_json_block(raw)


class RetrievalAgent:
    """
    Calls DocumentSearchTool with the rewritten query and returns chunks.
    """

    def __init__(self, search_tool: DocumentSearchTool):
        self.search_tool = search_tool

    def run(self, rewritten_query: str, top_k: int) -> List[DocumentChunk]:
        return self.search_tool.search(rewritten_query, top_k=top_k)


class AnswerAgent:
    """
    Produces a draft answer grounded in the retrieved chunks with citations.
    """

    SYSTEM_PROMPT = """You are a research assistant that answers questions using ONLY the provided document chunks.

Instructions:
- Use ONLY the information in the chunks.
- If the chunks are insufficient to fully answer, say so explicitly.
- For every factual sentence you write, include at least one citation
  in the form [doc_id:chunk_id].
- Do not invent sources or content that is not clearly supported by the chunks.

Your output should be plain text answer paragraphs with inline citations.
"""

    def run(self, question: str, chunks: List[DocumentChunk]) -> str:
        context_lines = []
        for c in chunks:
            context_lines.append(
                f"[{c.doc_id}:{c.chunk_id}] (score={c.score:.4f}) {c.text}"
            )
        context_txt = "\n".join(context_lines)

        user_prompt = (
            f"User question:\n{question}\n\n"
            f"Here are the retrieved document chunks:\n{context_txt}\n\n"
            "Now write a grounded answer with inline citations."
        )
        return call_llm(self.SYSTEM_PROMPT, user_prompt)


class CriticAgent:
    """
    Checks the draft answer for hallucinations and correctness.
    Returns a corrected answer if needed.
    """

    SYSTEM_PROMPT = """You are a strict fact-checking critic agent.

You are given:
- A user question
- A set of document chunks (with IDs like [doc_id:chunk_id])
- A draft answer that uses citations.

Your job:
1. Verify that each factual claim in the draft answer is supported by
   the provided chunks.
2. If you find unsupported or hallucinated claims, either:
   - Remove them, or
   - Soften them and explicitly mark uncertainty.
3. Ensure every remaining factual sentence has at least one citation
   of the form [doc_id:chunk_id].
4. Do NOT introduce new facts from outside the chunks.

Return ONLY the final corrected answer (no commentary).
"""

    def run(self, question: str, chunks: List[DocumentChunk], draft_answer: str) -> str:
        context_lines = []
        for c in chunks:
            context_lines.append(
                f"[{c.doc_id}:{c.chunk_id}] (score={c.score:.4f}) {c.text}"
            )
        context_txt = "\n".join(context_lines)

        user_prompt = (
            f"User question:\n{question}\n\n"
            f"Document chunks:\n{context_txt}\n\n"
            f"Draft answer:\n{draft_answer}\n\n"
            "Now return the corrected final answer."
        )
        return call_llm(self.SYSTEM_PROMPT, user_prompt)


# ------------------------------
# Orchestrator
# ------------------------------

class RAGOrchestrator:
    """
    Manager that wires together all agents for end-to-end QA.
    """

    def __init__(self, corpus_dir: str):
        self.search_tool = DocumentSearchTool(corpus_dir=corpus_dir)
        self.router = QuestionRouterAgent()
        self.retriever = RetrievalAgent(self.search_tool)
        self.answer_agent = AnswerAgent()
        self.critic_agent = CriticAgent()

    def answer_question(self, question: str) -> Dict[str, Any]:
        # 1. Route / analyze question
        route_info = self.router.run(question)
        rewritten_query = route_info.get("rewritten_query", question)
        top_k = int(route_info.get("top_k", 5))
        qtype = route_info.get("question_type", "unknown")

        # 2. Retrieve chunks
        chunks = self.retriever.run(rewritten_query, top_k=top_k)

        # 3. Draft answer
        draft_answer = self.answer_agent.run(question, chunks)

        # 4. Critic / verification
        final_answer = self.critic_agent.run(question, chunks, draft_answer)

        # 5. Return full trace (useful for debugging & evaluation)
        return {
            "question": question,
            "question_type": qtype,
            "rewritten_query": rewritten_query,
            "top_k": top_k,
            "retrieved_chunks": [c.__dict__ for c in chunks],
            "draft_answer": draft_answer,
            "final_answer": final_answer,
        }


# ------------------------------
# Simple CLI entry point
# ------------------------------

if __name__ == "__main__":
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(description="Simple RAG Assistant CLI")
    parser.add_argument(
        "--corpus_dir",
        type=str,
        default="corpus",
        help="Directory containing .txt files for the RAG corpus.",
    )
    args = parser.parse_args()

    orchestrator = RAGOrchestrator(corpus_dir=args.corpus_dir)

    print("RAG Assistant ready. Type your question, or 'exit' to quit.\n")
    while True:
        try:
            q = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if q.strip().lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        result = orchestrator.answer_question(q)
        print("\n--- Final Answer ---")
        print(textwrap.fill(result["final_answer"], width=100))
        print("\n--- Debug Info ---")
        print(f"Question type: {result['question_type']}")
        print(f"Rewritten query: {result['rewritten_query']}")
        print(f"Top_k: {result['top_k']}")
        print("--------------------\n")

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import ConfigDict, PrivateAttr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_TRANSCRIPT = """Operator
Welcome to the earnings call Q&A session.

CFO
Revenue grew 12% year over year, driven by enterprise demand and a stronger renewal base.

CEO
We continue to invest in product and go-to-market efficiency while preserving margins.

Analyst
Can you talk about the drivers behind the gross margin expansion?

CFO
The main drivers were better cloud infrastructure utilization, pricing discipline, and a favorable product mix.
"""


def read_transcript(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return DEFAULT_TRANSCRIPT


def build_documents(transcript: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    return splitter.create_documents([transcript])


class TfidfRetriever(BaseRetriever):
    """Simple local retriever that avoids any embedding-model dependency."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: list[Document]
    k: int = 4

    _vectorizer: TfidfVectorizer = PrivateAttr()
    _matrix = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(
            [document.page_content for document in self.documents]
        )

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        ranked_indices = scores.argsort()[::-1]
        return [self.documents[index] for index in ranked_indices[: self.k]]


def ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def ollama_chat_model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.1")


def build_retriever(documents: Iterable[Document]) -> BaseRetriever:
    return TfidfRetriever(documents=list(documents), k=4)


def build_retrieval_chain(retriever: BaseRetriever):
    llm = ChatOllama(
        model=ollama_chat_model(),
        temperature=0,
        base_url=ollama_base_url(),
    )
    contextualize_q_system = (
        "Given the chat history and a follow-up question, rewrite the question "
        "to be standalone if needed. Do not answer the question."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system = (
        "You answer questions about an earnings call transcript. "
        "Use only the supplied context. If the answer is not in the context, say you do not know."
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system),
            ("human", "Context:\n{context}\n\nQuestion: {input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def run_retrieval(transcript: str, question: str) -> None:
    documents = build_documents(transcript)
    retriever = build_retriever(documents)
    chain = build_retrieval_chain(retriever)
    result = chain.invoke({"input": question, "chat_history": []})
    print(result["answer"])


def run_conversation(transcript: str, question: str) -> None:
    llm = ChatOllama(
        model=ollama_chat_model(),
        temperature=0,
        base_url=ollama_base_url(),
    )
    prompt = PromptTemplate.from_template(
        "You are answering questions about an earnings call transcript.\n\n"
        "Transcript:\n{transcript}\n\n"
        "Conversation so far:\n{history}\n\n"
        "Human: {input}\n"
        "AI:"
    )
    chain = ConversationChain(
        llm=llm,
        prompt=prompt.partial(transcript=transcript),
        verbose=False,
    )
    print(chain.predict(input=question))


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Basic LangChain earnings call Q&A scaffold."
    )
    parser.add_argument(
        "--mode",
        choices=("retrieval", "conversation"),
        default="retrieval",
        help="Choose the chain style to run.",
    )
    parser.add_argument(
        "--transcript",
        help="Path to a text file containing a transcript excerpt.",
    )
    parser.add_argument(
        "--question",
        default="What drove the gross margin expansion?",
        help="Question to ask about the transcript.",
    )
    args = parser.parse_args()

    if not os.getenv("OLLAMA_MODEL"):
        print("Using default Ollama chat model: llama3.1")

    transcript = read_transcript(args.transcript)

    if args.mode == "retrieval":
        run_retrieval(transcript, args.question)
    else:
        run_conversation(transcript, args.question)


if __name__ == "__main__":
    main()

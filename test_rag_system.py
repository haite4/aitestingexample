import requests
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import ContextualRecallMetric

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_chroma import Chroma
import re
from langchain_huggingface import HuggingFaceEmbeddings
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

class LocalLLM(DeepEvalBaseLLM):

    def __init__(self, base_url="http://127.0.0.1:1234", model="llama-3.1-8b-instruct"):
        self.base_url = base_url
        self.model = model

    def get_model_name(self):
        return self.model
    
    def load_model(self) -> None:
        pass
    
    def generate(self, prompt: str):
        response = requests.post(f'{self.base_url}/v1/chat/completions',
                                 json={
                                     "messages": [{"role": "user", "content": prompt}],
                                      "model": self.model
                                 })
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return "Error"
        
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
                                         
localLLM= LocalLLM()

loader = WebBaseLoader(
    web_paths=("https://luxequality.com/collaboration/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            name=True,
            attrs={"class": re.compile("ModuleCheckList_itemTextBlock")}
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
all_splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./data"
)

_ = vector_store.add_documents(documents=all_splits)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    simple_prompt = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {state["question"]} 
    Context: {docs_content} 
    Answer:"""
    
    response = localLLM.generate(simple_prompt)
    return {"answer": response}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

question = "Support"

response = graph.invoke({"question": question})
actual_output = response["answer"]

retrieved_docs = vector_store.similarity_search(question)
retrieval_context = [doc.page_content for doc in retrieved_docs]


print("Retrieal_context", retrieval_context)
print("Actual output ", actual_output)

expected_output = """Our expert team is here to back you up, ensuring smooth project transitions and execution."""

contextual_recall_metric = ContextualRecallMetric(
    threshold=0.6,  
    model=localLLM,  
)

test_case = LLMTestCase(
    input=question,
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=retrieval_context
)


assert_test(test_case, [contextual_recall_metric])

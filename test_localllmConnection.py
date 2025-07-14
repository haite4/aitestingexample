import requests
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval import evaluate
from deepeval.synthesizer import Synthesizer

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
    
                                        
localLLM = LocalLLM()

synthesizer = Synthesizer(model=localLLM, max_concurrent=4)

docs = synthesizer.generate_goldens_from_docs(document_paths=["Company_FAQ.pdf"])

test_cases = []  
for doc in docs:
    test_case = LLMTestCase(
        input=doc.input,
        actual_output=localLLM.generate(doc.input),
        context=doc.context,
        expected_output=doc.expected_output  
 
    )
    test_cases.append(test_case) 

answer_relevancy = AnswerRelevancyMetric(model=localLLM, threshold=0.7)

result = evaluate(test_cases=test_cases, metrics=[answer_relevancy])
print(f"Passed: {sum(r.success for r in result.test_results)}/{len(test_cases)} tests")




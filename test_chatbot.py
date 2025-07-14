from dotenv import load_dotenv
import os
from openai import OpenAI

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def get_response(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        max_tokens=150
    )
    return response.choices[0].message.content

question_one = "What is Tesla"
question_two = "What is Apple"

first_test_case = LLMTestCase(
    input=question_one, 
    actual_output=get_response(question_one)
)

second_test_case = LLMTestCase(
    input=question_two, 
    actual_output=get_response(question_two)
)

dataset = EvaluationDataset(
    test_cases=[first_test_case, second_test_case]
)

@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_example(test_case: LLMTestCase):
    metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [metric])
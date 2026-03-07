from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.output_parsers import PydanticOutputParser
from llm_utils import get_score_runs, invoke_with_retries

# Prompt template for the coherence evaluation task. 
# Given passage.
# Output: JSON object with score.
SCORE_PROMPT = '''
Analyze the following passage and rate its coherence using an integer score from 1 to 10:
<passage>
{response}
<\passage>
Wrap the output in `json` tags\n{format_instructions}
'''

SCORE_SCHEMA = {
    "title": "Score",
    "description": "Coherence score.",
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "description": "Analyze the given passage and rate its coherence using an integer score from 1 to 10.",
        },
    },
    "required": ["score"]
}

PROBLEM_PROMPT = '''
For the task: {task}
You got the following passage: {response}
You are also providing its coherence score, ranging from 1 to 10, with 10 being the highest and 1 the lowest: {score}
Please help me point out the problems with the coherence of the current passage to improve its score.
Limit your answer in 50 words.
'''


class Score(BaseModel):
    """
    Coherence score (1 – 10).
    """

    score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Analyze the given passage and rate its coherence using an integer score from 1 to 10.",
    )

    model_config = ConfigDict(
        title="Score",
        description="Coherence score.",
    )


def evaluate(llm, task, response):
    """True rating function. evaluate the coherence of the response. If the score is less than 10, return the problems with the response. Otherwise, return an empty string with the response.

    Args:
        llm (_type_): LLM model to be used.
        task (string): task description.
        response (string): response of the workflow function.

    Returns:
        (int, string): final_score, problem
    """
    parser = PydanticOutputParser(pydantic_object=Score)
    score_prompt = PromptTemplate(
        input_variables=["response"],
        template=SCORE_PROMPT
    ).partial(format_instructions=parser.get_format_instructions())
    chain = score_prompt | llm | parser
    input = {
        "response": response,
    }

    score_runs = get_score_runs()
    scores = 0
    for _ in range(score_runs):
        score = invoke_with_retries(chain, input, description="CW coherence scoring")
        score = score.model_dump(by_alias=True)
        scores += score['score']
    
    final_score = scores / score_runs

    if final_score < 10 and task != None:
        problem_prompt = PromptTemplate(
            input_variables=["task", "response", "score"],
            template=PROBLEM_PROMPT
        )
        chain = problem_prompt | llm | StrOutputParser()
        input = {
            "task": task,
            "response": response,
            "score": final_score
        }
        problem = invoke_with_retries(chain, input, description="CW problem analysis")
    else:
        problem = ''
    
    return final_score, problem

def get_fitness(results):
    """calc the fitness of the results, which is the average score of the results

    Args:
        results (_type_): _description_

    Returns:
        int: average score of the results
    """
    scores = [result['score'] for result in results]
    fitness = sum(scores) / len(scores)
    return fitness

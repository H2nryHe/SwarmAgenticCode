import json
from logger import log
from langchain_core.prompts import PromptTemplate
from llm_utils import invoke_with_retries


FEEDBACK_TEMPLATE = '''You are an expert assistant.
You are tasked with analyzing the given workflow to identify where issues occurred, leading to the problem. You must provide a detailed explanation of the cause of the error.

The team is solving the following task:
<task>
{task}
</task>

The roles are collaborative in the following workflow:
<workflow>
{workflow}
</workflow>

You are also provided with the problem in the team result:
<problem>
{evaluation}
</problem>

Please provide a detailed explanation of the root cause of the <problem> at the identified step(s) with by referencing the detail information of the <task>, while considering factors such as incorrect execution, missing information, or deviations from the intended process.
Think a bit step by step and limit your answer in 150 words.
'''

schema = {
    "title": "Explanation",
    "description": "Explain the root cause of the <problem>.",
    "type": "object",
    "properties": {
        "explanation": {
            "type": "string",
            "description": "Provide a detailed explanation of the root cause of the <problem> at the identified step(s) with by referencing the detail information of the <task>, while considering factors such as incorrect execution, missing information, or deviations from the intended process."
        },
    },
    "required": ["explanation"]
}

def give_feedback(llm, logger, task, workflow, evaluation):

    prompt = PromptTemplate(
            input_variables=["task", "workflow", "evaluation"],
            template=FEEDBACK_TEMPLATE,
    )

    chain = prompt | llm.with_structured_output(schema)
    input = {
        "task": task, 
        "workflow": json.dumps(workflow, indent=2),
        "evaluation": evaluation,
    }
    res = invoke_with_retries(chain, input, "Generate feedback")

    log(logger, 'Give Feedback', prompt.format(**input), res["explanation"])
    return res["explanation"]

import json
from logger import log
from langchain_core.prompts import PromptTemplate
from llm_utils import invoke_with_retries

# Prompt template for the failure identification function. 
# Given feedback (flaw) and velocity.
# Output: list of identified flaws and failed adjustments.
IDENTIFY_FAILURE_TEMPLATE = '''
You are a strategic advisor focused on enhancing team performance. Your role is to carefully analyze the feedback provided and identify failed adjustments with previous adjustment plan.
You are given the following feedback, areas for team improvement:
<feedback> 
{feedback}
</feedback>
 
You are also provided with the previous adjustment plan, the measures taken to enhance team performance:
<previous adjustment plan>
{velocity}
</previous adjustment plan>

# Instruction
   
For each flaw in <feedback>, please apply the following steps:
1. Identified Flaw: 
    - Clearly outline the specific flaw identified in the <feedback> section.
2. Thought: 
    - Carefully think if there is any **Proposed Adjustment** in the <previous adjustment plan> section for the exact same **Identified Flaw**. 
3. Failed Adjustment: 
    - Based on your **Thought**, quote the exact **Proposed Adjustment** as described in <previous adjustment plan> if there is any **Proposed Adjustment** for the same kind of Identified Flaw in <previous adjustment plan>. Otherwise, say 'None' here.
'''

schema = {
    "title": "Identification",
    "description": "Identify previous failed adjustments.",
    "type": "object",
    "properties": {
        "Failed Adjustments": {
            "type": "array",
            "description": "For each flaw in <feedback>, identify its previous failed adjustment.",
            "items": {
                "type": "object",
                "properties": {
                    "Identified Flaw": {
                        "type": "string",
                        "description": "Clearly outline the specific flaw identified in the <feedback> section."
                    },
                    "Thought": {
                        "type": "string",
                        "description": "Carefully think if there is any **Proposed Adjustment** in the <previous adjustment plan> section for the exact same **Identified Flaw**."
                    },                      
                    "Failed Adjustment": {
                        "type": "string",
                        "description": '''Based on your **Thought**, quote the exact **Proposed Adjustment** as described in <previous adjustment plan> if there is any **Proposed Adjustment** for the same kind of Identified Flaw in <previous adjustment plan>. Otherwise, say 'None' here.'''
                    },
                },
                "required": ["Identified Flaw", "Thought", "Failed Adjustment"]
            }
        }
    },
    "required": ["Failed Adjustments"]
}

def identify_failure(llm, logger, evaluation, velocity):
    prompt = PromptTemplate(
            input_variables=["feedback", "velocity"],
            template=IDENTIFY_FAILURE_TEMPLATE,
    )
    chain = prompt | llm.with_structured_output(schema)
    input = {
        "feedback": '\n'.join(f'{i+1}. {item["flaw type"]}: {item["description"]}' for i, item in enumerate(evaluation)),
        "velocity": json.dumps(velocity, indent=4), 
    }
    res = invoke_with_retries(chain, input, description="Identify failure")
    while len(res["Failed Adjustments"]) != len(evaluation):
        res = invoke_with_retries(chain, input, description="Identify failure")
    
    failed_adjustments = '\n'.join(json.dumps({"Identified Flaw": item["Identified Flaw"], "Failed Adjustment": item["Failed Adjustment"]}, indent=4) for item in res["Failed Adjustments"])
    clean_output = f'''{failed_adjustments}'''

    res["Failed Adjustments"] = '\n'.join(json.dumps(item,indent=4) for item in res["Failed Adjustments"])
    res = '\n'.join(f'**{key}**:\n{value}\n' for key, value in res.items())

    log(logger, 'Identify Failure', prompt.format(**input), res)
    return clean_output

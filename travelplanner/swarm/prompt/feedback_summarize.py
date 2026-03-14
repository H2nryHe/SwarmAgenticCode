import json
from logger import log
from langchain_core.prompts import PromptTemplate
from llm_utils import invoke_with_retries


SUMMARIZE_TEMPLATE = '''You are an expert assistant tasked with reflect on feedback and indicate specific flaws in the current team.
Given the following feedback:
<feedback> 
{feedback}
</feedback>

The team to optimize is as follows, including its roles and collaborative workflow:
<current team>
{team}
</current team>

# Instruction

Based on the <feedback>, identify the specific flaws in the roles or workflow steps that directly contributed to the <feedback>. The flaw should be within following types:
    1. Inappropriate Output: Before identifying an output as inappropriate, verify whether the requested output falls within the role's scope of responsibility. If the requested output exceeds the role's responsibility, reassign the task to an existing role better suited for it or create a new role specifically responsible for the output if no such role exists. Only when the required output is within the role's responsibility and still incorrect, missing, or incomplete should it be classified as inappropriate output for that role. Clearly specify the role responsible for the step and the exact step number where the output was inappropriate.
    2. Missing Role: Were there missing roles in the team that left certain tasks inadequately addressed or overlooked? Clearly specify which role may be needed.
    3. Redundant Role: Were there redundant roles in the team that are unnecessary? Clearly indicate the specific role that is redundant.
    4. Role Policy Deficiency: If the policy of the role is sufficiently instructive, clear, and effective. Are there gaps, ambiguities, or contradictions in the policy that affect role performance? Clearly specify the name of the role.
    5. Missing Workflow Step: Were there missing steps in the workflow that left certain tasks inadequately addressed or overlooked? Clearly specify between which two steps the missing step should have occurred.
    6. Redundant Workflow Step: Were there redundant steps in the workflow that are unnecessary? Clearly indicate the specific role and the exact step number that is redundant.
    7. Insufficient Input: Were the inputs insufficient for the workflow steps? Assess if it includes all necessary information needed to get the role's output with its responsibility effectively. Clearly specify the role responsible for the step and the exact step number where the input was insufficient. 
'''

schema = {
    "title": "Reflection",
    "description": "Reflection on the current team and workflow.",
    "type": "object",
    "properties": {
        "Reflect on Team Flaws": {
            "type": "array",
            "description": "**Reflect on Team Flaws**.",
            "items": {
                "type": "object",
                "properties": {
                    "flaw type": {
                        "type": "string",
                        "description": "The identified flaw type.",
                        "enum": ["Missing Role", "Redundant Role", "Role Policy Deficiency", "Missing Workflow Step", "Redundant Workflow Step", "Insufficient Input", "Inappropriate Output"]
                    }, 
                    "description": {
                        "type": "string",
                        "description": "Description of the flaw."
                    },                      
                },
                "required": ["flaw type", "description"]
            },
        },
    },
    "required": ["Reflect on Team Flaws"]
}

def summarize_feedback(llm, logger, feedback, team):

    prompt = PromptTemplate(
            input_variables=["feedback", "team"],
            template=SUMMARIZE_TEMPLATE,
    )

    chain = prompt | llm.with_structured_output(schema)
    input = { 
        "feedback": feedback,
        "team": team
    }
    res = invoke_with_retries(chain, input, "Summarize feedback")
    
    log_content = '\n'.join(f'{i+1}. {item["flaw type"]}: {item["description"]}' for i, item in enumerate(res["Reflect on Team Flaws"]))
    log(logger, 'Summarize Feedback', prompt.format(**input), log_content)
    return res["Reflect on Team Flaws"]

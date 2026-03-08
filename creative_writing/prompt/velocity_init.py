import json
from logger import log
from langchain_core.prompts import PromptTemplate
from llm_utils import invoke_with_retries

# Prompt template for the velocity initialization.
# Given current team and flaw.
# Output: List of flaw and adjustment.
INIT_VELOCITY_TEMPLATE = '''You are tasked with optimizing a multi-agent team setup to enhance its performance in solving a specific task.
The team to optimize is as follows, including its roles and collaborative workflow: 
<current team>
{current_team}
</current team>

However, the <current team>'s performance is insufficient and must be improved based on the following feedback:
<feedback> 
{feedback} 
</feedback>

# Instruction

Follow the instruction to generate your response:
- Use the following OPERATIONS to refine roles within <current team>: 
    * Add Role: Introduce a new role when an existing subtask becomes overly complex or burdensome, requiring a specialized responsibility that cannot be integrated into current roles without disrupting their primary responsibilities. Define the role's:
        - Name: A clear name that reflects its specific responsibility.
        - Responsibility: Specific tasks or functions the role will handle.
        - Policy: Operational guidelines for fulfilling the role's duties.
    * Modify Role: Adjust the policy of an existing role for improved role execution, when the identified inefficiencies or gaps can be addressed through manageable refinements to its policy, ensuring the changes do not overburden the role and be within the scope of its responsibiity.
    * Delete Role: Remove roles that are redundant, unnecessary, or conflict with the team's primary objectives.
- Use the following OPERATIONS to optimize the workflow of the <current team>:
    * Add Step: Add a new step if a gap exists in the workflow that hinders overall efficiency, coordination, or goal achievement. Ensure the new step does not duplicate the functions of existing steps and adds clear value to the process. Define the step's:
        - Role: The role responsible for acting in this step.
        - Input: The input for this step must be the output produced by one or more roles in previous steps.
        - Output: What output expected from the role in this step.
    * Modify Input: Adjust the input of an existing workflow step to ensure that it comprehensively incorporates outputs from previous steps to support the current step.
    * Modify Output: Modify the output of an existing workflow step to ensure that it fully aligns with the expected deliverables of the step and supports the inputs of subsequent steps.
    * Delete Step: Delete a step if it has become redundant, no longer contributes to team goals, or overlaps with other steps in the workflow. Ensure the removal of the step does not impact other steps' efficiency or completeness in achieving objectives.
    * Re-order Steps: Re-order steps if their current sequence causes inefficiencies or coordination issues within the workflow. Ensure the new order improves logical flow without compromising the integrity or dependencies of other steps.
- For each identified flaw in <feedback>, apply the following steps:
    * Identified Flaw: Clearly outline the specific flaw identified in the <feedback> section.
    * Proposed Adjustment: Specify the exact OPERATIONS to address the **Identified Flaw**.
'''

schema = {
    "title": "Plan",
    "description": "Makae an adjustment plan to the roles and workflow in the current team.",
    "type": "object",
    "properties": {
        "Adjustments": {
            "type": "array",
            "description": "Plan to adjustment the roles and workflow in the current team.",
            "items": {
                "type": "object",
                "properties": {
                    "Identified Flaw": {
                        "type": "string",
                        "description": "Clearly outline the specific flaw identified in the <feedback> section."
                    },
                    "Proposed Adjustment": {
                        "type": "string",
                        "description": "Specify the exact OPERATIONS to address the **Identified Flaw**."
                    },                       
                },
                "required": ["Identified Flaw", "Proposed Adjustment"]
            },
        }
    },
    "required": ["Adjustments"]
}


def initialize_velocity(llm, logger, team, evaluation):
    prompt = PromptTemplate(
            input_variables=["current_team", "feedback"],
            template=INIT_VELOCITY_TEMPLATE,
    )
    chain = prompt | llm.with_structured_output(schema)
    input = {
        "current_team": team,
        "feedback": '\n'.join(f'{i+1}. {item["flaw type"]}: {item["description"]}' for i, item in enumerate(evaluation))
,
    }
    res = invoke_with_retries(chain, input, description="Initialize velocity")

    log_output = '\n'.join(json.dumps(item,indent=4) for item in res["Adjustments"])
    log(logger, 'Initialize Velocity', prompt.format(**input), log_output)
    
    return res["Adjustments"]

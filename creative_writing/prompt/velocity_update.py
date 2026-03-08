import json
from logger import log
from prompt.base import TASK_MINI
from langchain_core.prompts import PromptTemplate
from llm_utils import invoke_with_retries

# Prompt template for the velocity update function. 
# Given the current team, task, and Reflection from their personal best and global best.
# Output the adjustments to optimize the roles and workflow in the current team.

UPDATE_VELOCITY_TEMPLATE = '''You are tasked with optimizing a multi-agent team setup to enhance its performance in solving a specific task.
The team to optimize is as follows, including its roles and collaborative workflow:
<current team>
{team}
</current team>

This team is designed to solve the following type of tasks:
<task>
{task}
</task> 

# Objective

Develop a detailed adjustment plan focused on optimizing roles and the collaborative workflow to maximize the <current team>'s performance in addressing the specified <task>. 
The adjustments must be based on following feedback:
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
- For each Identified Flaw in <feedback>, apply the following steps:
    * Identified Flaw: Clearly outline the specific Identified Flaw in the <feedback> section.
    * Proposed Adjustment: Based on the **Recommended Adjustment**, **Best Team Insights**, and **Past Best Setup Reflection**, generate a final adjustment plan that directly addresses the **Identified Flaw**, while avoiding any repetition of the **Failed Adjustments**. 
'''


schema = {
    "title": "Plan",
    "description": "Makae a plan to adjustment the roles and workflow in the current team.",
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
                        "description": "Based on the **Recommended Adjustment**, **Best Team Insights**, and **Past Best Setup Reflection**, generate a final adjustment plan that directly addresses the **Identified Flaw**, while avoiding any repetition of the **Failed Adjustments**."
                    },                       
                },
                "required": ["Identified Flaw", "Proposed Adjustment"]
            },

        }
    },
    "required": ["Adjustments"]
}

def combine_input(velocity, g_best, p_best):
    combined_input = []
    try:
        for i in range(len(velocity)):
            if g_best is None:
                global_best = 'None' 
            else:
                global_best = g_best[i]["Proposed Adjustment"] if "Proposed Adjustment" in g_best[i].keys() else 'None'
            if p_best is None:
                personal_best = 'None' 
            else:
                personal_best = p_best[i]["Proposed Adjustment"] if "Proposed Adjustment" in p_best[i].keys() else 'None'

            combined_input.append(
                {
                    "Identified Flaw": velocity[i]["Identified Flaw"],
                    "Failed Adjustment": velocity[i]["Failed Adjustment"] if "Failed Adjustment" in velocity[i].keys() else 'None',
                    "Recommended Adjustment": velocity[i]["Proposed Adjustment"],
                    "Best Team Insights": global_best,
                    "Past Best Setup Reflection": personal_best
                }   
            ) 
    except:
        print(f'{len(velocity), len(g_best), len(p_best)}')
    return json.dumps(combined_input, indent=4)

def update_velocity(llm, logger, team, velocity, g_best, p_best):
    prompt = PromptTemplate(
            input_variables=["team", "task", "feedback"],
            template=UPDATE_VELOCITY_TEMPLATE,
    )
    chain = prompt | llm.with_structured_output(schema)

    input = { 
        "team": team,
        "task": TASK_MINI,
        "feedback": combine_input(velocity, g_best, p_best),
    }
    res = invoke_with_retries(chain, input, description="Update velocity")

    log_output = '\n'.join(json.dumps(item,indent=4) for item in res["Adjustments"])
    log(logger, 'Update Velocity', prompt.format(**input), log_output)
    
    return res["Adjustments"]

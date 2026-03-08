import json
from langchain_core.prompts import PromptTemplate

from logger import log
from prompt.base import TASK_MINI
from llm_utils import invoke_with_retries

# Prompt template for the reflection from global best function.
# Given team, task, flaw, and global best team.
# Output the 
# reflection: {Identified Flaw, Thought, Comparative Insights, Proposed Adjustment}
GLOBAL_BEST_TEMPLATE = '''You are a strategic assistant tasked with improving a team's performance by analyzing the strengths of a higher-performing example team.
Your objective is to understand the specific practices and configurations of the more optimized team that are directly relevant to solving the current team's issues. You will suggest practical improvements to the current team without copying outright.

You are tasked with improving the current team's roles and collaborative workflow:
<current team>
{current_team}
</current team>

This team is designed to solve the following type of tasks:
<task>
{task}
</task>

However, the <current team>'s performance is insufficient and must be improved based on the following feedback:
<feedback>
{feedback}
</feedback>

You have been provided with details of a globally recognized high-performing team, optimized specifically for solving the same type of <task> as the <current team>:
<global best team>
{g_best}
</global best team>

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
    * Thought: What can we learn from the <global best team>'s descriptions to do better in the **Identified Flaw**.
    * Comparative Insights: 
        - Extract specific elements from the <global best team>'s descriptions that demonstrate excellence in the **Identified Flaw**. 
        - Present these elements as part of a structured sentence, explicitly quoting the key phrases from their role responsibilities, role policies, step inputs, step outputs or step orders. 
        - Ensure the response integrates the quoted descriptions into a coherent sentence without adding commentary, assumptions, or analysis. 
        - If nothing helpful to solve the **Identified Flaw**, say 'None'.
    * Proposed Adjustment: The adjustment must directly reflect and utilize the specific phrases quoted in the **Comparative Insights**. The wording and content of the adjustment must align with these insights without introducing unrelated suggestions, rephrased ideas, or unquoted elements. The response must clearly demonstrate how the adjustment directly incorporates the practices described in **Comparative Insights**. If **Comparative Insights** is 'None', say 'None' here.
'''

schema = {
    "title": "Reflection",
    "description": "Reflection on the global best team to adjust roles and workflow.",
    "type": "object",
    "properties": {
        "Adjustments": {
            "type": "array",
            "description": "Suggestions of adjustments.",
            "items": {
                "type": "object",
                "properties": {
                    "Identified Flaw": {
                        "type": "string",
                        "description": "Clearly outline the specific flaw identified in the <feedback> section."
                    },
                    "Thought": {
                        "type": "string",
                        "description": "What can we learn from the <global best team>'s descriptions to do better in the **Identified Flaw**."
                    },                    
                    "Comparative Insights": {
                        "type": "string",
                        "description": "Extract specific elements from the <global best team>'s descriptions that demonstrate excellence. Present these elements as part of a structured sentence, explicitly quoting the key phrases from their role responsibilities, role policies, step inputs, step outputs or step orders. Ensure the response integrates the quoted descriptions into a coherent sentence without adding commentary, assumptions, or analysis. If nothing helpful to solve the **Identified Flaw**, say 'None'.."
                    },
                    "Proposed Adjustment": {
                        "type": "string",
                        "description": "The adjustment must directly reflect and utilize the specific phrases quoted in the **Comparative Insights**. The wording and content of the adjustment must align with these insights without introducing unrelated suggestions, rephrased ideas, or unquoted elements. The response must clearly demonstrate how the adjustment directly incorporates the practices described in **Comparative Insights**. If **Comparative Insights** is 'None', say 'None' here."
                    },                       
                },
                "required": ["Identified Flaw", "Comparative Insights", "Proposed Adjustment"]
            },

        },
    },
    "required": ["Adjustments"]
}

def reflect_from_global_best(llm, logger, team, evaluation, g_best):
    prompt = PromptTemplate(
            input_variables=["current_team", "task", "feedback", "g_best"],
            template=GLOBAL_BEST_TEMPLATE,
    )
    chain = prompt | llm.with_structured_output(schema)
    input = {
        "current_team": team,
        "task": TASK_MINI, 
        "feedback": '\n'.join(f'{i+1}. {item["flaw type"]}: {item["description"]}' for i, item in enumerate(evaluation)),
        "g_best": g_best
    }
    res = invoke_with_retries(chain, input, description="Reflect from global best")
    while len(res["Adjustments"]) != len(evaluation):
        res = invoke_with_retries(chain, input, description="Reflect from global best")

    log_output = '\n'.join(json.dumps(item,indent=4) for item in res["Adjustments"])
    log(logger, 'Global Best', prompt.format(**input), log_output) 
    
    return res["Adjustments"]

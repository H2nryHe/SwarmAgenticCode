import re
import json
from logger import log
from prompt.base import TASK_MINI
from langchain_core.prompts import PromptTemplate
from llm_utils import invoke_with_retries

# Prompt template for the team update function.
# Given team, workflow, task, and plan (velocity).
# Output: refined team with updated roles and workflow.
UPDATE_TEAM_TEMPLATE = '''You are an expert assistant, and writer. You are tasked with generating a refined team from a existing team according to the reflection.
You are given the roles within the current team:
<roles>
{team}
</roles>

You are also provided the workflow of the current team:
<workflow>
{workflow}
</workflow>

The team is solving the following type of tasks:
<task>
{task}
</task>

# Instruction

Your job is to update the <roles> and <workflow> of the team based on the following plan:
<plan>
{plan}
</plan>

Use these guidelines when generating the answer:
<system-guidelines>
1. If a role is not required modification in the plan, it must be retained in the final “roles” list with its original “Name,” “Responsibility,” and “Policy.”
2. If the plan specifies that a role should be modified, only update the “Policy”; do not change the “Name” or “Responsibility.”
3. If the plan specifies that a role should be removed, then remove it from the final “roles” list.
4. If the plan specifies adding a new role, include it in the final “roles” list with its “Name,” “Responsibility,” and “Policy.”
5. When generating the final answer, verify the total number of roles to ensure:
   - All roles that do not require modification remain unchanged.
   - Roles marked for removal are actually removed.
   - Newly added roles appear in the final list.
   - Modified roles are correctly updated.
6. The information flow must be strictly modular, with each step primarily receiving structured input from the outputs of previous steps. Steps can refer to the initial task definition implicitly as needed, but it should not be treated as a direct input for workflow dependencies.
7. Each step's output must be structured and usable as a direct input for subsequent steps, creating a clear, step-by-step workflow.
8. Each step can only be assigned to a single role and cannot involve multiple roles simultaneously.
9. The final step in the workflow must produce the exact deliverable specified in the <task> without referencing any intermediate steps.
</system-guidelines>

**Before generating the final answer, please check whether all roles that appear in the workflow are already in the roles list.**
'''

schema = {
    "title": "new_team",
    "description": "Team afer refining.",
    "type": "object",
    "properties": {
        "roles": {
            "type": "array",
            "description": "Roles of the refined team.",
            "items": {
                "type": "object",
                "properties": {
                    "Name": {
                         "type": "string",
                         "description": "A clear name that reflects its specific responsibility.",
                    },
                    "Responsibility": {
                         "type": "string",
                         "description": "Specific tasks or functions the role will handle.",
                    },
                    "Policy": {
                        "type": "string",
                        "description": "Operational guidelines for fulfilling the role's duties. Provide step-by-step instructions in a numbered list format (1., 2., 3., etc.) for how the role can achieve their goal.",
                    },
                },
                "required": ["Name", "Responsibility", "Policy"],
            }
        },
        "workflow": {
            "type": "array",
            "description": "A detailed workflow on how to solve the task with the roles.",
            "items": {
                "type": "object",
                "properties": {
                    "Step": {
                         "type": "string",
                         "description": "The sequence number of the step in the process.",
                    },
                    "Role": {
                         "type": "string",
                         "description": "The role responsible for acting in this step.",
                    },
                    "Input": {
                        "type": "string",
                        "description": "The input for this step must be the output produced by one or more roles in previous steps. If this step does not have an upstream dependency, the input should be defined as empty.",
                    },
                    "Output": {
                        "type": "string",
                        "description": "The output must be structured and usable as a direct input for subsequent steps.",
                    },
                },
                "required": ["Step", "Role", "Input", "Output"],
            }
        },
    },
    "required": ["roles", "workflow"],
}


def update_team(llm, logger, team, workflow, plan):
    prompt = PromptTemplate(
            input_variables=["team", "workflow", "task", "plan"],
            template=UPDATE_TEAM_TEMPLATE,
    )
    chain = prompt | llm.with_structured_output(schema)
    input = {
        "team": team, 
        "workflow": json.dumps(workflow, indent=2),
        "task": TASK_MINI,
        "plan": '\n'.join(f'{i+1}. {item["Proposed Adjustment"]}' for i, item in enumerate(plan))
    }
    res = invoke_with_retries(chain, input, description="Update team")

    try:
        lines = re.split(r'(\d+\.\s)', res["workflow"])
        lines = [lines[i] + lines[i + 1].strip() for i in range(1, len(lines), 2)]
        res["workflow"] = "\n".join(lines)
    except:
        pass

    log(logger, 'Update Team', prompt.format(**input), res)
    return res

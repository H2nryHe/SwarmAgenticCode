import re
from logger import log
from prompt.base import TASK_MINI
from langchain_core.prompts import PromptTemplate
from llm_utils import invoke_with_retries

# Prompt template for initializing a team to solve a task
# Given task.
# Output: JSON object with roles and workflow.


INIT_TEAM_TEMPLATE = '''You are an expert in designing a highly efficient, specialized, and collaborative multi-agent team for a specific task.

**Requirements:**
- The team must break down the task into highly specialized, modular roles.
- Each role should have a focused domain of responsibility, handling only one primary aspect of the task.
- The information flow must be strictly modular, with each step primarily receiving structured input from the outputs of previous steps. Steps can refer to the initial task definition implicitly as needed, but it should not be treated as a direct input for workflow dependencies.
- Each step's output must be structured and usable as a direct input for subsequent steps, creating a clear, step-by-step workflow.
- Each step can only be assigned to a single role and cannot involve multiple roles simultaneously.
- The resulting team structure should allow for easy scalability and clarity, ensuring that each module can be independently optimized or replaced without affecting other parts of the system.

**Deliverables:**
1. Define Each Role:
   - Name: A clear and descriptive title.
   - Responsibility: A narrowly focused set of tasks aligned with that domain.
   - Policy: Specific operational guidelines for fulfilling these tasks.
2. Collaboration Structure:
   - Clearly outline how roles interact and pass information to one another.
   - Ensure that information flows from one role to another in a well-defined manner. Each role should clearly know which role's output it relies on, if any. If there is no upstream role, it operates independently (with no input).
3. Sequential Workflow:
   - Illustrate a concrete workflow from start to finish.
   - For each step:
      * Specify the single role responsible for that step.
      * Define its input, which must come from previous roles' outputs or be empty.
      * Define its output, which will be used as input for subsequent steps.
   - Ensure there is a designated role at the end to integrate all components into the final deliverable.

Now, giving the following task:
<task>
{task}
</task>

Please design a detailed multi-agent collaborative team that could efficiently solve the <task>.
'''

schema = {
    "title": "Plan",
    "description": "Plan a team to solve the task.",
    "type": "object",
    "properties": {
        "roles": {
            "type": "array",
            "description": "Details of the needed roles",
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
 

def init_team(llm, logger) -> str:
    prompt = PromptTemplate(
            input_variables=["task", "size"],
            template=INIT_TEAM_TEMPLATE,
    )

    chain = prompt | llm.with_structured_output(schema)
    input = {"task": TASK_MINI}
    res = invoke_with_retries(chain, input, description="Init team")

    try:
        lines = re.split(r'(\d+\.\s)', res["workflow"])
        lines = [lines[i] + lines[i + 1].strip() for i in range(1, len(lines), 2)]
        res["workflow"] = "\n".join(lines)
    except:
        pass

    log(logger, 'Init Team', prompt.format(**input), res)
    return res

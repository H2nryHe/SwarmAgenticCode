import json
import copy
import tiktoken
from typing import List
from dataclasses import dataclass

from prompt.base import TASK_NOTICE
from prompt.team_init import init_team
from llm_utils import invoke_with_retries

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


ROLE_PROMPT = '''You are {name}. You are working in a team solving the following specific task:
<task instance>
{instance}
</task instance>

You are also provided with the helpful information from other team members: 
<helpful information>
{information}
</helpful information>

# Instruction
Based on the <task instance> and <helpful information>, your responsibility is: {responsibility}
Please follow the instruction step by step to give an answer: 
<instruction>
{policy}
</instruction>

# Output Guidance
Your answer only needs to include: {output}
Think step by step and limit your answer in 400 words.
'''
# Think step by step internally, but only output the final result.
# Think a bit step by step and limit your answer in 200 words.
@dataclass
class Message:
    role: str
    subtask: str
    content: str


@dataclass
class MessagePool:
    messages: List[Message]
    
    def add_message(self, message):
        self.messages.append(message)      
    
    def reset_message(self):
        self.messages = []


class Role():
    """
    """

    def __init__(self, role: dict, llm) -> None:
        self.name = role['Name']
        self.responsibility = role['Responsibility']
        self.policy = role['Policy']
        
        self.llm = llm
        self.message = Message(
            role=role['Name'],
            subtask = role['Responsibility'], 
            content=''
        )
        self.description = json.dumps(role, indent=4)

    def init_message(self):
        self.message = Message(
            role=self.name,
            subtask = self.responsibility,
            content=''
        )
    
    def parse_inputs(self, inputs: List) -> str:
        others_outputs = ""
        for i, mes in enumerate(inputs):
            if i == 0: 
                task_instance = mes.content
            else:
                if isinstance(mes, Message):
                    others_outputs += mes.content
                else:
                    others_outputs += str(mes)
        return task_instance, others_outputs
        
    def response(self, task_instance, others_outputs, output):                    
        prompt = PromptTemplate(
            input_variables=["name", "responsibility", "policy", "instance", "information", "output"],
            template=ROLE_PROMPT
        )
        chain = prompt | self.llm | StrOutputParser()
        input = {
            "name": self.name,
            "responsibility": self.responsibility,
            "policy": self.policy,
            "instance": task_instance + f'\n\n**Important Notice:**{TASK_NOTICE}',
            "information": others_outputs,
            "output": output
        }
        response = invoke_with_retries(chain, input, f"Role response for {self.name}")

        log = (f'Role - {self.name}', prompt.format(**input), response)
        self.message.content = f'''\n\n## {output.split('.')[0]} from {self.name}: \n{response}\n'''
        return self.message.content, log
    
    def to_str(self):
        return self.__repr__()
    
    def to_dict(self):
        return {
            "Name": self.name, 
            "Responsibility": self.responsibility, 
            "Policy": self.policy,  
        }

    def __repr__(self):
        return self.description

    def __call__(self, inputs, output):
        task_instance, others_outputs = self.parse_inputs(inputs)
        return self.response(task_instance, others_outputs, output)
    

class Team():
    def __init__(self, llm, logger) -> None:
        self.llm = llm
        self.roles = None
        self.workflow = None
        self.task = None
        self.message_pool = None
        self.logger = logger
        self.logs = None

    def init(self, llm):
        res = init_team(llm, self.logger)
        init_roles = res['roles']
        roles = []
        for role in init_roles:
            roles.append(Role(role=role, llm=self.llm))
        self.roles = roles
        self.workflow = res['workflow']

    def copy(self, team):
        self.roles = team.roles
        for role in self.roles:
            role.llm=self.llm
        self.workflow = team.workflow

    def deepcopy(self):
        new_team = Team(self.llm, logger=self.logger)
        new_team.roles = []
        for role in self.roles:
            copy_role = Role(json.loads(role.description), self.llm)
            new_team.roles.append(copy_role)
        new_team.workflow = self.workflow
        return new_team

    def reset_task(self, task):
        self.task = Message(role='user', subtask='', content=f'{task}')
        self.logs = []

        # Clear message history
        self.message_pool = MessagePool(messages=[self.task])
        for role in self.roles:
            role.init_message()

    def to_str(self):
        roles_str = ''
        for role in self.roles:
            roles_str += f'{role.to_str()}\n' 
        return roles_str

    def __repr__(self):
        return self.to_str() 

    def call(self, required_role: str, inputs: List = [], output: str = ""):
        for role in self.roles:
            if role.name == required_role:
                inputs = [self.task] + inputs
                response, log = role(inputs, output)
                self.logs.append(log) 
                self.message_pool.add_message(role.message)
                return response

        return f"Call an unexisting Role {required_role}."
    
    def update(self, new_team: dict):
        roles = []
        for role in new_team['roles']:
            roles.append(Role(role=role, llm=self.llm))
        self.roles = roles
        self.workflow = new_team['workflow']

        self.task = None
        self.logs = None
        self.message_pool = None

    def save_into_dict(self) -> dict:
        team_dict = {"roles": [], "workflow": self.workflow}
        for role in self.roles:
            team_dict["roles"].append(role.to_dict())
        return team_dict
    
    def patch_result_and_workflow(self):
        workflow = copy.deepcopy(self.workflow)
        messages = self.message_pool.messages[1:]
        if len(workflow) == len(messages):
            for i, mes in enumerate(messages):
                workflow[i]['Result'] = mes.content
        for step in workflow:
            for role in self.roles:
                if role.name == step['Role']:
                    step['Role'] = role.to_dict()
        return workflow
    

import json
import random
from langchain_core.prompts import PromptTemplate

from prompt.base import TASK_OUTPUT_SCHEMA
from prompt.write_forward import build_forward
from llm_utils import invoke_with_retries

# If an attraction comes with its own City field, use that value verbatim for the city part of the “Name, City;” pair — do not infer or correct it from the address or your own knowledge. For example, output “Seattle Aquarium, Washington;” because its City field is “Washington,” even though the address shows Seattle.
EXTRACT_PALN_TEMPLATE = """
Please assist me in extracting valid information from a given natural language text and reconstructing it in JSON format, as demonstrated in the following example.
- Use a ';' to separate different attractions, with each attraction formatted as 'Name, City;', for example "Peoria Historical Society, Peoria;Peoria Holocaust Memorial, Peoria;".
- If there 's information about transportation, ensure that the 'current_city' aligns with the destination mentioned in the transportation details (i.e., the current city should follow the format 'from A to B'). 
- If the transportation entry is a flight, format it exactly as 'Flight Number: 12345, from OriginCity to DestinationCity, Departure Time: HH:MM, Arrival Time: HH:MM', placing a comma immediately after the destination city; for all non-flight transportation, keep the original wording.
- Also, ensure that all flight numbers and costs are followed by a colon (i.e., 'Flight Number:' and 'Cost:'), consistent with the provided example. 
- Each item should include ['day', 'current_city', 'transportation', 'breakfast', 'attraction', 'lunch , 'dinner', 'accommodation']. 
- Replace non-specific information like 'eat at home/on the road ' with '-'. Additionally , delete any '$' symbols.
- Please retain the original formatting of names exactly as provided in the natural language text. Do Not Correct Capitalization or Punctuation in Names. 
- If the accommodation information mentions multiple rooms (e.g., "2 rooms"), do not mention the room quantity in the output.
- Output one and only one accommodation for each day. If multiple options are mentioned, choose the most suitable single listing and omit the rest.
- When listing attractions, take the city exactly as it appears in the source data (e g., the “City” field).

<example>
[{{ 
        "days": 1, 
        "current_city": "from Dallas to Peoria", 
        "transportation": "Flight Number: 4044830, from Dallas to Peoria, Departure Time: 13:10, Arrival Time: 15:01", 
        "breakfast": "-", 
        "attraction": "Peoria Historical Society, Peoria;Peoria Holocaust Memorial, Peoria;", 
        "lunch": "-", 
        "dinner": "Tandoor Ka Zaika, Peoria", 
        "accommodation": "Bushwick Music Mansion, Peoria" 
    }}, 
    {{ 
        "days": 2, 
        "current_city": "Peoria", 
        "transportation ": "-", 
        "breakfast": "Tandoor Ka Zaika, Peoria", 
        "attraction": "Peoria Riverfront Park, Peoria;The Peoria PlayHouse, Peoria;Glen Oak Park, Peoria;", 
        "lunch": "Cafe Hashtag LoL, Peoria", 
        "dinner": "The Curzon Room - Maidens Hotel, Peoria", 
        "accommodation": "Bushwick Music Mansion, Peoria" 
    }}, 
    {{ 
        "days": 3, 
        "current_city": "from Peoria to Dallas", 
        "transportation": "Flight Number: 4045904, from Peoria to Dallas, Departure Time: 07:09, Arrival Time: 09:20", 
        "breakfast": "-", 
        "attraction": "-", 
        "lunch": "-", 
        "dinner": "-", 
        "accommodation ": "-" 
}}] 
</example>

Now, giving the following natural language text:
<natural language text>
{result}
</natural language text>

Please extract valid information from the given <natural language text> and reconstruct it in JSON format, as demonstrated in the <example>.
"""

def read_jsonl(file_path) -> list:
    "read jsonl file into list"
    data = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for jsonstr in f.readlines():
            jsondict = json.loads(jsonstr)
            data.append(jsondict)
    return data

def write_jsonl(file_path, data, mode):
    with open(file_path, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_dataset(path, shuffle=False, seed_index=0):
    dataset = read_jsonl(path)
    if shuffle:
        random.seed(seed_index)
        random.shuffle(dataset)
    return dataset

def load_ref_info(path):
    return read_jsonl(path)

def get_forward(llm, logger, roles, workflow):
    try:
        next_solution = build_forward(llm, logger, roles, workflow)
    except Exception as e:
        print("During LLM generate new solution:")
        print(e)
    return next_solution

def check_format(plan):
    """Apply some rules to the travel plan to make it suitable for the bechmark.

    Args:
        plan (_type_): _description_

    Returns:
        _type_: _description_
    """
    for i_day, day in enumerate(plan):
        if i_day != day["days"]-1:
            plan.pop(i_day)

    for i_day, day in enumerate(plan):
        try:
            if i_day == 0:
                if 'from' not in day["current_city"]:
                    day["current_city"] = f'from {day["current_city"]} to {plan[i_day+1]["current_city"]}'
                day["breakfast"] = "-"
                day["lunch"] = "-"
                day["dinner"] = "-"
                day["attraction"] = "-"

            if i_day == len(plan)-1:
                if 'from' not in day["current_city"]:
                    org = plan[0]["current_city"].split('from ')[1].split(' to ')[0].strip()
                    day["current_city"] = f'from {day["current_city"]} to {org}'
                day["breakfast"] = "-"
                day["lunch"] = "-"
                day["dinner"] = "-"
                day["attraction"] = "-"
                day["accommodation"] = "-"
            
            if day["transportation"] != '-' and 'from' not in day["transportation"]:
                day["transportation"] = f'{day["transportation"]}, {day["current_city"]}'
            
            if 'from' in day["current_city"]:
                day["breakfast"] = "-"
                day["lunch"] = "-"
                day["dinner"] = "-"
                day["attraction"] = "-"                           
        except:
            pass
        
    return plan
    
def extract_plan(llm, result):
    prompt = PromptTemplate(
        input_variables=["result"],
        template=EXTRACT_PALN_TEMPLATE
    )
    chain = prompt | llm.with_structured_output(TASK_OUTPUT_SCHEMA)
    input = {"result": result}
    res = invoke_with_retries(chain, input, "Extract travel plan")
    raw_plan = json.dumps(res, indent=4)
    checked_plan = check_format(res['travel_plan'])
    
    log = ('Team Final Answer', prompt.format(**input), f'{raw_plan}\n\n{json.dumps(checked_plan, indent=4)}')
    return checked_plan, log

def set_forward(next_solution):
    namespace = {}
    exec(next_solution, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    return func

def load_particles(idx: int):
    archive = read_jsonl('save.jsonl')
    particles = archive[idx]['archive']
    return particles

def save_particles(particles):
    archive = []
    for particle in particles:
        archive.append({"team": particle.best_position[0], "code": particle.best_position[1], "score": particle.best_fitness})
    write_jsonl("save.jsonl", [{"archive": archive}], 'a')
    
def save_state(particles, global_best_position, global_best_fitness, global_best_trend):
    particles_to_save = []
    for particle in particles:
        particles_to_save.append({
            "team": particle.position[0].save_into_dict(), 
            "code": particle.position[1], 
            "velocity": particle.velocity,
            "fitness": particle.fitness,
            "fitness_history": particle.fitness_history,
            "best_position": particle.best_position,
            "best_fitness": particle.best_fitness
        })
    
    state = [{
        "particles": particles_to_save,
        "global_best_position": global_best_position,
        "global_best_fitness": global_best_fitness, 
        "global_best_trend": global_best_trend
    }]
    write_jsonl("save_state.jsonl", state, 'a')

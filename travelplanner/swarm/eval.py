from typing import Dict

import os
import sys
sys.path.append("..")
from func import read_jsonl

EVAL_IMPORT_ERROR = None
_ORIGINAL_CWD = os.getcwd()
try:
    from evaluation.commonsense_constraint import evaluation as commonsense_eval
    from evaluation.hard_constraint import evaluation as hard_eval
    from evaluation.eval import eval_score
    os.chdir(_ORIGINAL_CWD)
except Exception as e:
    os.chdir(_ORIGINAL_CWD)
    commonsense_eval = None
    hard_eval = None
    eval_score = None
    EVAL_IMPORT_ERROR = e

# map constraint to description
KEY_MAP = {
    'Within Current City': '''All scheduled activities for the day must be located within that day's city(s).''',
    'Within Sandbox': 'All information in the plan must be within the given reference information.',
    'Reasonable City Route': 'Route Changes in cities during the trip must be reasonable.',
    'Diverse Restaurants': 'Restaurant choices should not be repeated throughout the trip.',
    'Non-conflict Transportation': 'Transportation choices within the trip having both “self-driving” and “flight” would be considered a conflict.',
    'Diverse Attractions': 'Attraction choices should not be repeated throughout the trip.',
    'Minimum Nights Stay': "The number of consecutive days booked in a specific accommodation during the trip must satisfy the corresponding accommodation's required minimum nights.",
    'Complete Information': 'No key information should be left out of the plan.',
    'Within Budget': 'Total cost of travel should be within budget.',
    'Valid Cuisine': "Cuisine of selected restaurants must meet user's preferences.",
    'Valid Room Type': "Room type of selected accommodations must meet user's preferences.",
    'Valid Room Rule': "Room rule of selected accommodations must meet user's preferences.",
    'Valid Transportation': "Transportation method must meet user's preferences."
}

# map function to constraint
ITEM_MAP = {
    'is_valid_information_in_current_city': 'Within Current City',
    'is_valid_information_in_sandbox': 'Within Sandbox',
    'is_reasonalbe_visiting_city': 'Reasonable City Route',
    'is_valid_restaurants': 'Diverse Restaurants',
    'is_valid_transportation': 'Non-conflict Transportation',
    'is_valid_attractions': 'Diverse Attractions',
    'is_valid_accommodation': 'Minimum Nights Stay',
    'is_not_absent': 'Complete Information',
    'valid_cost': 'Within Budget',
    'valid_room_rule': 'Valid Room Rule',
    'valid_cuisine': 'Valid Cuisine',
    'valid_room_type': 'Valid Room Type',
    'valid_transportation': 'Valid Transportation'
}

def evaluate(data, result) -> Dict:
    if commonsense_eval is None:
        return {}, f"Evaluator unavailable: {EVAL_IMPORT_ERROR}"

    if type(data) == str:
        data = eval(data)
    if type(data['local_constraint']) == str:
        data['local_constraint'] = eval(data['local_constraint'])

    commonsense_info_box = commonsense_eval(data, result)
    try:
        hard_info_box = hard_eval(data, result)
    except:
        hard_info_box = {
            'valid_cuisine': (None, None),
            'valid_room_rule': (None, None),
            'valid_transportation': (None, None),
            'valid_room_type': (None, None),
            'valid_cost': (False, None)
        }

    false_items = {}
    problems = ""
    for key, val in {**commonsense_info_box, **hard_info_box}.items():
        if val[0] is False:
            # constraint
            cons = ITEM_MAP[key]
            # description
            desc = KEY_MAP[cons]
            false_items[cons] = "False. Reason: " + str(val[1])
            problems += f'''For the constraint "{cons}": {desc} {false_items[cons]}\n'''

    return false_items, problems

def get_scores(dataset, file_path):
    if eval_score is None:
        results = read_jsonl(file_path)
        score = sum(item.get("score", 0.0) for item in results) / len(results) if results else 0.0
        fallback_scores = {
            'Commonsense Constraint Micro Pass Rate': score,
            'Hard Constraint Micro Pass Rate': score,
        }
        return fallback_scores, {"fallback_reason": str(EVAL_IMPORT_ERROR)}

    scores, detailed_scores = eval_score('train', file_path, dataset)   
    
    total_hard_constraints = 0
    for data in dataset:
        total_hard_constraints += 1
        if isinstance(data["local_constraint"], str):
            data["local_constraint"] = eval(data["local_constraint"])
        for _, val in data["local_constraint"].items():
            if val is not None:
                total_hard_constraints += 1
    
    scores['Hard Constraint Micro Pass Rate'] /= total_hard_constraints
    
    return scores, detailed_scores

def weight_constraints(false_item_list, data_number):
    constraint_scores = {val: 0 for val in ITEM_MAP.values()}
    
    for false_item in false_item_list:
        for cons in false_item.keys():
            for key in constraint_scores.keys():
                if key == cons:
                    constraint_scores[key] += 1
    
    for key in constraint_scores.keys():
        constraint_scores[key] = 1 - constraint_scores[key]/data_number

    target_cons = min(constraint_scores, key=constraint_scores.get)
    return  {"cons": target_cons, "desc": KEY_MAP[target_cons]}, constraint_scores

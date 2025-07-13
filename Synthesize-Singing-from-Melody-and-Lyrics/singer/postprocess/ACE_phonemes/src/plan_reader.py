import json
import os


current_file_path = os.path.dirname(__file__)
ALL_PLAN_PATH = os.path.join(current_file_path, '../resources/all_plans.json')

with open(ALL_PLAN_PATH, "r") as f:
    all_plan_dict = json.load(f)

assert all_plan_dict is not None

jp_word2romaji = all_plan_dict["jp_word2romaji"]

for plan in all_plan_dict["plans"]:
    if plan["language"] == "zh":
        zh_plan = plan
    elif plan["language"] == "jp":
        jp_plan = plan
    else:
        en_plan = plan

assert zh_plan is not None 
assert jp_plan is not None 
assert en_plan is not None 

import json

MODEL_LIST = ["BLIP2", "INSTRUCT_BLIP", "LLAVA", "GIT", "KOSMOS2", "VILT"]
with open("output/final_output.json", 'r') as json_file:
    data = json.load(json_file)

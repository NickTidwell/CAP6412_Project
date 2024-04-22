from EloSystem import EloRatingSystem
from model_info import MODEL_LIST, data

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import os
import random

print(torch.cuda.device_count())
device='cuda'

#################################
# Exclude BLIP2 from MODEL_LIST #
#################################
MODEL_LIST = ["INSTRUCT_BLIP", "LLAVA", "GIT", "KOSMOS2", "VILT"]

##################
# Load ELO class #
##################
elo_system = EloRatingSystem(elo_csv_path="output/blip2autoelo.csv")

##################
# Load LLaVA-34b #
##################
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")

model = model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl").to(device)
print('model loaded')
print(torch.cuda.memory_summary())
'''
print('test input...')
image = Image.open('laion/data/000000194.jpg').convert('RGB')
prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nChoose between (A/B/Tie) Which answer is better for this image and a corresponding question? Question: What color is the building? A: The building is brown. B: brown <|im_end|><|im_start|>assistant\n"
print(prompt)
inputs = processor(prompt, image, return_tensors="pt").to(device)
# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=10)
vote = processor.decode(output[0], skip_special_tokens=True)
print(vote)
print()
print(torch.cuda.memory_summary())
'''

template = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nChoose between (A/B/Tie) Which answer is better for this image and a corresponding question? Question: {} A: {} B: {}<|im_end|><|im_start|>assistant\n"

template = "Choose between (A/B/Tie) Which answer is better for this image and a corresponding question? Question: {} A: {} B: {}"

def get_random_model_indices():
    index1, index2 = random.sample(range(len(MODEL_LIST)), 2)
    return index1, index2

def get_random_datapoint_indice():
    return random.randint(0, len(data) - 1)
    
def validate_data_exist(current_data_index, model1, model2):
    if "image_path" not in data[current_data_index]:
        logging.debug("Image path is missing.")
        return False
    if "question" not in data[current_data_index]:
        logging.debug("Question is missing.")
        return False
    if MODEL_LIST[model1] not in data[current_data_index]:
        logging.debug(f"{MODEL_LIST[model1]} is missing.")
        return False
    if MODEL_LIST[model2] not in data[current_data_index]:
        logging.debug(f"{MODEL_LIST[model2]} is missing.")
        return False
    return True
    
for i in range(10000):
    current_data_index = get_random_datapoint_indice()
    model1, model2 = get_random_model_indices()
    while not validate_data_exist(current_data_index, model1, model2):
        current_data_index = get_random_datapoint_indice()
        model1, model2 = get_random_model_indices()
    
    image = Image.open(os.path.join('laion', data[current_data_index]["image_path"])).convert('RGB')
    question = data[current_data_index]["question"]
    ans1 = data[current_data_index][MODEL_LIST[model1]]
    ans2 = data[current_data_index][MODEL_LIST[model2]]
    
    # generate answer
    prompt = template.format(question, ans1, ans2)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    inputs['max_length'] = 51
    generated_ids = model.generate(**inputs)
    vote = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print('DEBUG:', vote)
    if (len(vote) >= 3) and ((vote[:3] == 'Tie') or (vote[:3] == 'tie')):
        vote = 'Tie'
    elif vote[0] == 'A' or vote[0] == 'a':
        vote = 'A'
    elif vote[0] == 'B' or vote[0] == 'b':
        vote = 'B'

    if vote == 'A':
        elo_system.update_ratings(MODEL_LIST[model1], MODEL_LIST[model2], 1)
    elif vote == 'B':
        elo_system.update_ratings(MODEL_LIST[model1], MODEL_LIST[model2], 0)
    elif vote == 'Tie':
        elo_system.update_ratings(MODEL_LIST[model1], MODEL_LIST[model2], 0.5)
    else:
        print('INVALID RESPONSE')
    print()

elo_system.print_elo()    

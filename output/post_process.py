import json
import csv
def load_csv_as_indexed_list(csv_file):
    indexed_list = []
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            indexed_list.append(row[1])
    return indexed_list

csv_file = 'questions.csv'
indexed_questions = load_csv_as_indexed_list(csv_file)

# Function to process each JSON file
def process_json(json_file, output_dict):
    with open("raw_outputs/" + json_file, 'r') as f:
        data = json.load(f)
        
    for index, (key, value) in enumerate(data.items()):
        image_path = f"data/{value['img']}"
        question = indexed_questions[index]
        
        if (image_path, question) not in output_dict:
            output_dict[(image_path, question)] = {}
        
        if json_file == 'blip2_ans.json':
            output_dict[(image_path, question)]['BLIP2'] = value['ans']
        elif json_file == 'instructblip_ans.json':
            output_dict[(image_path, question)]['INSTRUCT_BLIP'] = value['ans']
        elif json_file == 'llava_ans.json':
            output_dict[(image_path, question)]['LLAVA'] = value['ans']
        elif json_file == 'git_ans.json':
            output_dict[(image_path, question)]['GIT'] = value['ans']
        elif json_file == 'kosmos2_ans.json':
            output_dict[(image_path, question)]['KOSMOS2'] = value['ans']
        else:
            # Handle other cases if necessary
            pass
    
# List of JSON files
json_files = ['blip2_ans.json', 'git_ans.json', 'instructblip_ans.json', 'kosmos2_ans.json', 'llava_ans.json']

# Dictionary to store output grouped by image path and question
output_dict = {}

# Process each JSON file and accumulate the results
for json_file in json_files:
    process_json(json_file, output_dict)

# Convert output_dict to a list of dictionaries
final_output = []
for (image_path, question), answers in output_dict.items():
    entry = {
        "image_path": image_path,
        "question": question
    }
    entry.update(answers)
    final_output.append(entry)

# Save the final output as a new JSON file
output_filename = 'final_output.json'
with open(output_filename, 'w') as f:
    json.dump(final_output, f, indent=4)

print(f"Final output saved as {output_filename}")

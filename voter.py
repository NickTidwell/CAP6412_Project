import tkinter as tk
from tkinter import messagebox
import os
from PIL import ImageTk, Image
import random
import logging
from EloSystem import EloRatingSystem, StatsCounter
from model_info import MODEL_LIST, data
# Set up logging
logging.basicConfig(level=logging.DEBUG)  # Change the level as needed

#Intialize Voting List to 0
voting_dict = {}
for model in MODEL_LIST:
    voting_dict[model] = 0

def get_random_model_indices():
    index1, index2 = random.sample(range(len(MODEL_LIST)), 2)
    return index1, index2

def get_random_datapoint_indice():
    return random.randint(0, len(data) - 1)


class ImageComparisonApp:
    def __init__(self, root):
        self.elo_system = EloRatingSystem()
        self.stats_counter = StatsCounter("output/stats_counter.json")
        self.root = root
        self.root.title("Image Comparison App")
        self.image_frame = tk.Frame(root)
        self.image_frame.pack()
        self.current_data_index = 0  # Initialize with rnd index of image/data
        self.update_form()
        self.create_vote_buttons()

    def load_images(self):
        current_directory = os.getcwd()
        self.image1 = ImageTk.PhotoImage(Image.open(os.path.join(current_directory, data[self.current_data_index]["image_path"])))
        self.image2 = ImageTk.PhotoImage(Image.open(os.path.join(current_directory, data[self.current_data_index]["image_path"])))

    def validate_data_exist(self) -> bool:
        if "image_path" not in data[self.current_data_index]:
            logging.debug("Image path is missing.")
            return False
        if "question" not in data[self.current_data_index]:
            logging.debug("Question is missing.")
            return False
        if MODEL_LIST[self.model1] not in data[self.current_data_index]:
            logging.debug(f"{MODEL_LIST[self.model1]} is missing.")
            return False
        if MODEL_LIST[self.model2] not in data[self.current_data_index]:
            logging.debug(f"{MODEL_LIST[self.model2]} is missing.")
            return False
        return True

    def create_image_labels(self):
        self.model1_label = tk.Label(self.image_frame, text="Model 1")
        self.model1_label.grid(row=0, column=0, padx=10)

        self.label1 = tk.Label(self.image_frame, image=self.image1)
        self.label1.grid(row=1, column=0, padx=10)

        self.model2_label = tk.Label(self.image_frame, text="Model 2")
        self.model2_label.grid(row=0, column=1, padx=10)

        self.label2 = tk.Label(self.image_frame, image=self.image2)
        self.label2.grid(row=1, column=1, padx=10)

    def create_caption_labels(self):
        self.caption1_label = tk.Label(self.image_frame, text="Question:")
        self.caption1_label.grid(row=2, column=0, padx=10, sticky="w")

        self.caption2_label = tk.Label(self.image_frame, text="Question:")
        self.caption2_label.grid(row=2, column=1, padx=10, sticky="w")

    def create_caption_text_boxes(self):
        self.caption1_text = tk.Text(self.image_frame, height=3, width=50, wrap="word")
        self.caption1_text.grid(row=3, column=0, padx=10, sticky="w")
        self.caption1_text.insert("1.0", data[self.current_data_index]["question"])
        self.caption1_text.config(state="disabled")

        self.caption2_text = tk.Text(self.image_frame, height=3, width=50, wrap="word")
        self.caption2_text.grid(row=3, column=1, padx=10, sticky="w")
        self.caption2_text.insert("1.0", data[self.current_data_index]["question"])
        self.caption2_text.config(state="disabled")

    def create_answer_labels(self):
        self.answer1_label = tk.Label(self.image_frame, text="Answer 1:")
        self.answer1_label.grid(row=4, column=0, padx=10, sticky="w")

        self.answer2_label = tk.Label(self.image_frame, text="Answer 2:")
        self.answer2_label.grid(row=4, column=1, padx=10, sticky="w")

    def create_answer_text_boxes(self):
        self.answer1_text = tk.Text(self.image_frame, height=10, width=50, wrap="word")
        self.answer1_text.grid(row=5, column=0, padx=10, sticky="w")
        self.answer1_text.insert("1.0", data[self.current_data_index][MODEL_LIST[self.model1]])
        self.answer1_text.config(state="disabled")  # Set to disabled after inserting text

        self.answer2_text = tk.Text(self.image_frame, height=10, width=50, wrap="word")
        self.answer2_text.grid(row=5, column=1, padx=10, sticky="w")
        self.answer2_text.insert("1.0", data[self.current_data_index][MODEL_LIST[self.model2]])
        self.answer2_text.config(state="disabled")

    def create_vote_buttons(self):
        self.vote_label = tk.Label(self.root, text="Vote for Best Model:")
        self.vote_label.pack(anchor="center")

        self.vote_var = tk.StringVar()
        self.vote_var.set("TIE")

        self.model1_button = tk.Button(self.root, text="Model 1", command=lambda: self.vote("Model 1", 1), width=10)
        self.model1_button.pack(side="left", padx=5)

        self.model2_button = tk.Button(self.root, text="Model 2", command=lambda: self.vote("Model 2", 2), width=10)
        self.model2_button.pack(side="left", padx=5)

        self.tie_button = tk.Button(self.root, text="TIE", command=lambda: self.vote("TIE", 0), width=10)
        self.tie_button.pack(side="left", padx=5)

        # Create the "Print Dictionary" button
        self.print_dict_button = tk.Button(self.root, text="Display Votes", command=self.display_results, width=15)
        self.print_dict_button.pack(side="right", padx=5)

    def update_form(self):
        self.current_data_index = get_random_datapoint_indice()
        self.model1, self.model2 = get_random_model_indices()
        while not self.validate_data_exist():
            self.update_form()
        self.load_images()
        self.create_image_labels()
        self.create_caption_labels()
        self.create_caption_text_boxes()
        self.create_answer_labels()
        self.create_answer_text_boxes()

    def vote(self, selected_model, vote_id):
        logging.info(f"Voted for: {selected_model}")
        logging.info(f"Model 1 was: {MODEL_LIST[self.model1]}")
        logging.info(f"Model 2 was: {MODEL_LIST[self.model2]}")
        logging.info("=======================================")

        if(vote_id == 1):
            voting_dict[MODEL_LIST[self.model1]] += 1
            self.elo_system.update_ratings(MODEL_LIST[self.model1], MODEL_LIST[self.model2], 1)
        elif(vote_id == 2):
            voting_dict[MODEL_LIST[self.model2]] += 1
            self.elo_system.update_ratings(MODEL_LIST[self.model1], MODEL_LIST[self.model2], 0)
        else:
            self.elo_system.update_ratings(MODEL_LIST[self.model1], MODEL_LIST[self.model2], 0.5)
        self.stats_counter.update_count(MODEL_LIST[self.model1], MODEL_LIST[self.model2])
        self.update_form()

    def display_results(self):
        print(voting_dict)
        print("")
        self.elo_system.print_elo()
        print("")
        self.stats_counter.print_dictionary()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageComparisonApp(root)
    root.mainloop()

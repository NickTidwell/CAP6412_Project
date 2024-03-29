import tkinter as tk
import pandas as pd
from PIL import ImageTk, Image
import os
from tkinter import filedialog

class ImageLabelApp:
    def __init__(self, root, df):
        self.df = df
        self.root = root
        self.questions = [tk.StringVar() for _ in range(5)]  # Initialize a list for user input questions
        self.root.title("Image Label App")
        self.image_frame = tk.Frame(root)
        self.image_frame.pack()
        self.select_directory()
        self.image_files = [f for f in os.listdir(self.image_dir.get()) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.image_files =sorted(self.image_files, key=lambda x: int(x.split(".")[0]))
        self.current_image_index = 0
        self.create_frame()
        self.loop_frames()
        
    def create_frame(self):
        # Create label for image directory
        label_dir = tk.Label(self.image_frame, background='grey', text="Image Directory:")
        label_dir.grid(row=0, column=0, sticky="E")

        # Create entry widget for image directory
        entry_dir = tk.Entry(self.image_frame, textvariable=self.image_dir, width=100)
        entry_dir.grid(row=0, column=1, sticky="W")

        # Create button to select image directory
        button_dir = tk.Button(self.image_frame, text="Select Directory", command=self.select_directory)
        button_dir.grid(row=0, column=2, sticky="W")

        # Create entry widgets for user input questions
        for i in range(5):
            label_question = tk.Label(self.image_frame, text=f"Question {i + 1}:")
            label_question.grid(row=i + 1, column=0, sticky="E")
            entry_question = tk.Entry(self.image_frame, textvariable=self.questions[i], width=100)
            entry_question.grid(row=i + 1, column=1, sticky="W")

        # Create button to save and move to the next image
        self.button_save_next = tk.Button(self.image_frame, text="Save and Next", command=self.save_and_next, state=tk.DISABLED)
        self.button_save_next.grid(row=6, column=0, columnspan=3)

        # Check if all entry widgets have inputs
        for question in self.questions:
            question.trace_add("write", self.check_inputs)
    def select_directory(self):
        directory = filedialog.askdirectory()
        self.image_dir = tk.StringVar()
        if directory:
            self.image_dir.set(directory)
    def save_and_next(self):
        # Save user input for the current image
        image_name = self.image_files[self.current_image_index]
        question_list = [question.get() for question in self.questions]   

        if image_name not in self.df['Image'].values:
            for question in question_list:
                self.df = self.df.append({'Image': image_name, 'Question': question}, ignore_index=True)
            self.save_dataframe()
         # Clear the questions list
        for i in range(5):
            self.questions[i].set("")
        # Move to the next image not labeled
        while self.image_files[self.current_image_index] in self.df['Image'].values:
            self.current_image_index += 1
            self.display_next_image()
    def display_next_image(self):
        if self.current_image_index < len(self.image_files):
            image_path = os.path.join(self.image_dir.get(), self.image_files[self.current_image_index])
            img = Image.open(image_path)
            img = img.resize((400, 400), Image.ANTIALIAS)
            img_tk = ImageTk.PhotoImage(img)
            # Update the label with the new image
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
        else:
            print("All images processed!")
    def loop_frames(self):
        # Create a label widget to display images
        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        self.save_and_next()
    def check_inputs(self, *args):
        # Enable the "Save and Next" button if all entry widgets have non-empty values
        if all(question.get() for question in self.questions):
            self.button_save_next.config(state=tk.NORMAL)
        else:
            self.button_save_next.config(state=tk.DISABLED)
    def save_dataframe(self):
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "questions.csv")
        
        self.df['Image_Index'] = self.df['Image'].str.extract(r'(\d+)').astype(int)
        self.df.sort_values(by='Image_Index', inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.df.drop(columns=['Unnamed: 0'], inplace=True)
        self.df.drop(columns=['Image_Index'], inplace=True)
        self.df.to_csv(output_path, index=False)
        print(f"DataFrame saved to {output_path}")
        
# # Create dataframe
df = pd.DataFrame(columns=['Image', 'Question'])
# Check if the CSV file exists
csv_file_path = "output/questions.csv"
if os.path.exists(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    print("Existing CSV file loaded successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabelApp(root, df)
    root.mainloop()

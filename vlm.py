import os
os.environ["HF_HOME"] = "<pathtocahe>"

import torch
from PIL import Image
from PIL import Image, ImageOps
import math
import re



from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer




import csv
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import requests
import matplotlib.pyplot as plt
from transformers import pipeline

def read_csv(file_path):
    return pd.read_csv(file_path, encoding='latin-1')

def list_folders_with_numbers(directory):
    folders_with_numbers = []
    for folder in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, folder)):
            if any(char.isdigit() for char in folder):
                files_in_folder = os.listdir(os.path.join(directory, folder))
                image_files_count = sum(1 for file in files_in_folder if file.lower().endswith(('.png', '.jpg', '.jpeg')))
                if image_files_count > 1:
                    folders_with_numbers.append(folder)
    folders_with_numbers.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    return folders_with_numbers

def select_rows(csv_data, files_directory):
    selected_rows = []
    selected_file_paths = []
    files = list_folders_with_numbers(files_directory)
    for filename in files:
        try:
            file_index = int(filename.split('-')[1])
            selected_rows.append(csv_data.iloc[file_index-2])
            selected_file_paths.append(os.path.join(files_directory, filename))
        except (ValueError, IndexError):
            pass
    return selected_rows, selected_file_paths

if __name__ == "__main__":


    csv_file_path = '<pathtocsvfile>'
    files_directory = '<pathtodirectory>'

    csv_data = read_csv(csv_file_path)
    selected_rows, selected_file_paths = select_rows(csv_data, files_directory)
    folder_names = list_folders_with_numbers(files_directory)
    print(len(folder_names))
    print(folder_names)

    assert len(selected_rows) == len(selected_file_paths)

    custom_index = list(range(0,len(selected_rows)))
    df_syn = pd.DataFrame(selected_rows, columns=['Questions', 'answers'], index=custom_index)
    df_syn['file_path'] = selected_file_paths
    df_syn['filename'] = folder_names

IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']

def repeat_rows(df):
    new_rows = []
    for _, row in df.iterrows():
        path = row['file_path']
        image_files = [name for name in os.listdir(path)
                       if os.path.isfile(os.path.join(path, name))
                       and any(name.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)]
        for _ in range(len(image_files)):
            new_rows.append(row.tolist())
    return new_rows

rows_list = repeat_rows(df_syn)

df_testing = pd.DataFrame(rows_list)
df_testing_renamed = df_testing.rename(columns={0: 'Questions', 1: 'Answers', 2: 'file_path', 3: 'file_name'})

def extract_question_and_options(question_text):
    question_options = question_text.split('\n')
    question = question_options[0].strip()
    options = [option.strip() for option in question_options[1:] if option.strip()]
    return question, options

def read_csv():

    df = df_testing_renamed

    questions = []
    option_lists = []
    for index, row in df.iterrows():
        question_text = str(row[0])

        question, options = extract_question_and_options(question_text)
        questions.append(question)
        option_lists.append(options)
    return questions, option_lists


questions, options = read_csv()
questions[0], questions[-1], options[0], options[-1]

def sort_folders_by_number(folders):
    def extract_number(folder_name):
        try:
            return int(''.join(filter(str.isdigit, folder_name)))
        except ValueError:
            return float('inf')

    return sorted(folders, key=extract_number)

def get_image_paths(root_dir):
    image_paths = {}

    subdirectories = sort_folders_by_number(os.listdir(root_dir))
    for foldername in subdirectories:

        folder_path = os.path.join(root_dir, foldername)
        if os.path.isdir(folder_path):
            image_paths[foldername] = []

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if os.path.isfile(file_path) and any(file_path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                    image_paths[foldername].append(file_path)
    return image_paths

def load_images(image_paths):
    loaded_images = {}
    for foldername, paths in image_paths.items():
        loaded_images[foldername] = []
        for path in paths:
            img = Image.open(path).resize((512, 512))
            img = ImageOps.exif_transpose(img)
            loaded_images[foldername].append(img)
    return loaded_images

root_directory = '<pathtodirectory>'
image_paths = get_image_paths(root_directory)
image_paths.pop('.ipynb_checkpoints')
loaded_images = load_images(image_paths)

imgs_lst = []
for lst in loaded_images.values():
    imgs_lst.extend(lst)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_vqa, vis_processors_vqa, txt_processors_vqa = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)

answers_lst = []

for question, option, image in zip(questions, options, imgs_lst):
    question_tmp = txt_processors_vqa["eval"](question)

    image_tmp = vis_processors_vqa["eval"](image).unsqueeze(0).to(device)

    samples = {"image": image_tmp, "text_input": question_tmp}
    answer = model_vqa.predict_answers(samples, answer_list=option, inference_method="rank")
    answers_lst.append(answer[0])

answer_dataframe = pd.DataFrame(answers_lst)

def extract_word(text):
    match = re.search(r'(\w+)\s*\)', text)
    if match:
        return match.group(1)

answer_dataframe["cleaned_answers"] = answer_dataframe[0].apply(extract_word)

answer_dataframe["ground_truth"] = df_testing_renamed["Answers"]
answer_dataframe['accuracy'] = (answer_dataframe['ground_truth'] == answer_dataframe['cleaned_answers']).astype(int)
answer_dataframe['accuracy'].mean()

second_part = "BLIP"
file_name = second_part + "_" + f"habitat.csv"



directory_name = f"<pathtodirectory>/{second_part}"

print(directory_name)

if not os.path.exists(directory_name):

    os.makedirs(directory_name)
    print("Directory created:", directory_name)
else:
    print("Directory already exists:", directory_name)


full_path = f"{directory_name}/{file_name}"

answer_dataframe.to_csv(full_path, index=False)


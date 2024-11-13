import os
os.environ["HF_HOME"] = "<path to store cache>"

import torch
from PIL import Image
from PIL import Image, ImageOps
import math
import re
import ast




from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer




import csv
import pandas as pd
import numpy as np

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
    csv_file_path = '<pathtofile>'
    files_directory = '<pathtoimagedirectory>'

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

df_testing_renamed = df_testing.rename(columns={0: 'Questions', 1: 'Answers', 2: 'file_path', 3: 'folder_name'})

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

root_directory = '<pathtoimagefiles>'
image_paths = get_image_paths(root_directory)
image_paths.pop('.ipynb_checkpoints')
loaded_images = load_images(image_paths)

def modification_llava_16_7(text):
    return "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.  USER: <image>\n" + text +  ". Please answer with the option's letter from the given choices directly. ASSISTANT:"

df_testing_renamed['Questions'] = df_testing_renamed['Questions'].apply(modification_llava_16_7)

prompt_dict = {}
for index, row in df_testing_renamed.iterrows():
    key = row['folder_name']
    value = row['Questions']
    prompt_dict.setdefault(key, []).append(value)

differences = {}


for key in prompt_dict:
    if key in loaded_images:
        if len(prompt_dict[key]) != len(loaded_images[key]):
            differences[key] = (len(prompt_dict[key]), len(loaded_images[key]))
    else:
        differences[key] = (len(prompt_dict[key]), 0)

for key in loaded_images:
    if key not in prompt_dict:
        differences[key] = (0, len(loaded_images[key]))

print("Differences in list lengths:", differences)

lst_prompt = [len(lst) for lst in prompt_dict.values()]
lst_images =[len(lst) for lst in loaded_images.values()]
if lst_prompt == lst_images:
    print("both lists are equal")
else:
    print("not equal are not equal")

bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
)



model_id = "llava-v1.6-vicuna-7b-hf"

processor = LlavaNextProcessor.from_pretrained(model_id)




model = LlavaNextForConditionalGeneration.from_pretrained(model_id,
                                                          torch_dtype=torch.float16,
                                                          quantization_config = bnb_config,
                                                          low_cpu_mem_usage=True,
                                                          attn_implementation="flash_attention_2",
                                                          device_map="auto"
                                                         )



generated_text_dict = {}

for (prompt_key, prompt_value), (image_key, image_value) in zip(prompt_dict.items(), loaded_images.items()):

    inputs = processor(prompt_value, images=image_value, padding=True, return_tensors="pt").to(1, torch.float16)
    processor.tokenizer.padding_side = "left"
    output = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    generated_text_dict[prompt_key] = generated_text

def find_lists_with_none(dictionary):
    keys_with_none = []
    for key, value in dictionary.items():
        if None in value:
            keys_with_none.append(key)
    return keys_with_none

result = find_lists_with_none(generated_text_dict)

for key, value_list in generated_text_dict.items():
    for i in range(len(value_list)):
        split_value = value_list[i].split("assistant")[-1]
        value_list[i] = split_value

max_length = max(len(lst) for lst in generated_text_dict.values())

for key, value in generated_text_dict.items():
    if len(value) < max_length:
        generated_text_dict[key] += [None] * (max_length - len(value))

df = pd.DataFrame.from_dict(generated_text_dict)

df = df.applymap(lambda x: x.replace('\n', '') if isinstance(x, str) else x)

def extract_word(text):
    if text is not None:
        match = re.search(r'(\w+)\s*\)', text)
        if match:
            return match.group(1)
        else:
            return text

lst = []
for i in df.columns:
    lst.append(df[i].apply(extract_word))
df_na = pd.DataFrame(lst)

def row_to_list(row):
    return list(row)

df_na['temp'] = df_na.apply(lambda row: row_to_list(row), axis=1)

df_na["ground_truth_answers"] = df_syn["answers"].tolist()

def calculate_accuracy(row):
    my_list = row['temp']
    cleaned_list = [item.strip() if item is not None else None for item in my_list]
    cleaned_list_1 = [item.lower() if item is not None else None for item in cleaned_list]
    print(cleaned_list_1)
    value_to_compare = row['ground_truth_answers']


    filtered_list = [item for item in cleaned_list_1 if item is not None]
    total = len(filtered_list)
    correct = sum(1 for item in filtered_list if item == value_to_compare)
    if total == 0:
        return 0
    return correct / total


df_na['Accuracy'] = df_na.apply(calculate_accuracy, axis=1)

df_na["Accuracy"].mean()

second_part = model_id.split('/')[1]
file_name = second_part + "_" + f"habitat.csv"


directory_name = f"<pathtodirectory>/{second_part}"

print(directory_name)

if not os.path.exists(directory_name):
    os.makedirs(directory_name)
    print("Directory created:", directory_name)
else:
    print("Directory already exists:", directory_name)


full_path = f"{directory_name}/{file_name}"

df_na.to_csv(full_path, index=False)


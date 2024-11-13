import os
os.environ["HF_HOME"] = "<pathtocache>"

import torch
from PIL import Image
from PIL import Image, ImageOps
import math



from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from matplotlib import pyplot as plt
from lavis.common.gradcam import getAttMap
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
import numpy as np


import csv
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import requests
import matplotlib.pyplot as plt
from transformers import pipeline

main_folder = "<pathtoimagefolder>"

def read_csv(file_path):
    return pd.read_csv(file_path, encoding='latin-1')



def list_files(directory):
    return os.listdir(directory)

def list_folders_with_numbers(directory):
    folders_with_numbers = []

    folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

    for folder in folders:
        if any(char.isdigit() for char in folder):
            folders_with_numbers.append(folder)

    folders_with_numbers.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    return folders_with_numbers


def select_rows(csv_data, files_directory):
    selected_rows = []
    files = list_folders_with_numbers(files_directory)

    for filename in files:
        try:

            file_index = int(filename.split('-')[1])

            selected_rows.append(csv_data.iloc[file_index-2])


        except (ValueError, IndexError):
            pass

    return selected_rows

if __name__ == "__main__":

    csv_file_path = '<pathtofile>'
    files_directory = '<pathtodirectory>'

    csv_data = read_csv(csv_file_path)
    selected_rows = select_rows(csv_data, files_directory)


    custom_index = list(range(0,len(selected_rows)))
    df_syn = pd.DataFrame(selected_rows, columns=['Questions', 'answers'],index=custom_index)

def numerical_sort(subfolder):
    digits = ''.join(filter(str.isdigit, subfolder))
    return int(digits) if digits else float('inf')


def get_image_paths(main_folder, image_index=0):
    image_paths = []
    subfolders = sorted(next(os.walk(main_folder))[1], key=numerical_sort)
    for subdir in subfolders:
        full_subdir = os.path.join(main_folder, subdir)
        images_in_subfolder = [file for file in os.listdir(full_subdir) if file.endswith((".jpg", ".png", ".jpeg"))]
        if images_in_subfolder:
            selected_image = images_in_subfolder[min(image_index, len(images_in_subfolder) - 1)]
            image_path = os.path.join(full_subdir, selected_image)
            image_paths.append(image_path)
    return image_paths


image_index = 2
image_paths = get_image_paths(main_folder, image_index)


imgs_lst = []
for img in image_paths:

    imgs = Image.open(img)
    im = ImageOps.exif_transpose(imgs.resize((512, 512)))
    imgs_lst.append(im)

df_syn['image_path'] = image_paths


num_images = len(imgs_lst)
num_cols = 4
num_rows = math.ceil(num_images / num_cols)


fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))

for i, img in enumerate(imgs_lst):
    row = i // num_cols
    col = i % num_cols
    axes[row, col].imshow(img)
    axes[row, col].axis('off')


for i in range(num_images, num_rows * num_cols):
    row = i // num_cols
    col = i % num_cols
    axes[row, col].axis('off')



def extract_question_and_options(question_text):

    question_options = question_text.split('\n')

    question = question_options[0].strip()

    options = [option.strip() for option in question_options[1:] if option.strip()]
    return question, options

def read_csv():
    df = df_syn
    questions = []
    option_lists = []
    for index, row in df.iterrows():
        question_text = str(row[0])

        question, options = extract_question_and_options(question_text)
        questions.append(question)
        option_lists.append(options)

    return questions, option_lists


questions, options = read_csv()
questions[0], questions[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_vqa, vis_processors_vqa, txt_processors_vqa = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)

answers_lst = []

print(len(questions), len(options), len(imgs_lst))

for question, option, image in zip(questions, options, imgs_lst):
    question_tmp = txt_processors_vqa["eval"](question)


    image_tmp = vis_processors_vqa["eval"](image).unsqueeze(0).to(device)

    samples = {"image": image_tmp, "text_input": question_tmp}

    answer = model_vqa.predict_answers(samples, answer_list=option, inference_method="rank")
    answers_lst.append(answer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)

path = "pathtoimage"
raw_image = Image.open(path).convert("RGB")

raw_image = ImageOps.exif_transpose(raw_image)


dst_w = 720
w, h = raw_image.size
scaling_factor = dst_w / w

resized_img = raw_image.resize((int(w * scaling_factor), int(h * scaling_factor)))
norm_img = np.float32(resized_img) / 255

img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
txt = text_processors["eval"]("second bowl opening touches the bottom of the first bowl")

txt_tokens = model.tokenizer(txt, return_tensors="pt").to(device)
gradcam, _ = compute_gradcam(model, img, txt, txt_tokens, block_num=7)

avg_gradcam = getAttMap(norm_img, gradcam[0][1].numpy(), blur=True)

num_image = len(txt_tokens.input_ids[0]) - 2
num_image

gradcam_iter = iter(gradcam[0][2:-1].numpy())
token_id_iter = iter(txt_tokens.input_ids[0][1:-1])

plt.rcParams.update({'font.size':40})

plt.figure(figsize=(25,25))
for i, (gradcam, token_id) in enumerate(zip(gradcam_iter, token_id_iter)):
    print(gradcam.shape)
    word = model.tokenizer.decode([token_id], skip_special_tokens=True)
    print(word)
    gradcam_image = getAttMap(norm_img, gradcam, blur=True)

    plt.subplot(3, 4, i+1)
    normalized_attn_map = (gradcam_image - np.min(gradcam_image)) / (np.max(gradcam_image) - np.min(gradcam_image)) * 255
    normalized_attn_map = normalized_attn_map.astype(np.uint8)
    plt.imshow(normalized_attn_map)
    plt.yticks([])
    plt.xticks([])
    plt.xlabel(word)
    plt.margins(x=0)
    plt.margins(y=0)
    plt.tight_layout()

save_path = '<savepath>'
plt.savefig(f"{save_path}/normalized_attn_map.png")




import re
import sys
import os
import torch
import transformers
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import json
import random
from datasets import load_dataset
import numpy as np
import pandas as pd
import argparse
import templates, verbalizations
from sklearn.metrics import classification_report
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentTypeError

"""
python test.py --data_path "projecte-aina/tecla" \
               --model_path "symanto/xlm-roberta-base-snli-mnli-anli-xnli" \
               --data_split "test" \
               --text_field "text" \
               --label_field "label1" \
               --template_set 2 \
               --template_num 1 \
               --label_verbalization "verbalizations_templateset2_coarse_grained" \
                --output_dir "."
                
python test.py --data_path "tecla/tecla.py" --model_path "/gpfs/projects/bsc88/projects/entailment_irene/models/roberta-base-ca-v2/wiki_no_teca/output/roberta-base-ca-v2/tecla_nli.py_8_0.00003_date_22-12-21_time_01-40-19/checkpoint-2494" --data_split "test" --text_field "text" --label_field "label1" --template_num 7 --label_verbalization "verbalizations_templateset1_coarse_grained" --output_dir "pruebas_testpy/wiki-no-teca"
python test.py --data_path "tecla/tecla.py" --model_path "/gpfs/projects/bsc88/projects/entailment_irene/models/symanto_xlm-roberta-base-snli-mnli-anli-xnli" --data_split "test" --text_field "text" --label_field "label1" --template_num 13 --label_verbalization "verbalizations_templateset2_coarse_grained" --output_dir "pruebas_testpy/smax"
python test.py --data_path "tecla/tecla.py" --model_path "/gpfs/projects/bsc88/projects/entailment_irene/models/roberta-base-ca-v2-te" --data_split "test" --text_field "text" --label_field "label1" --template_num 16 --premise_shortening "first_sentence" --label_verbalization "verbalizations_templateset2_coarse_grained" --output_dir "pruebas_testpy/roberta-te"
"""

def add_base_arguments_to_parser(parser):
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the HF dataset or to dataloader.'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the HF model.'
    )
    parser.add_argument(
        '--data_split',
        type=str,
        required=True,
        help='Specify the dataset split to convert.'
    )
    parser.add_argument(
        '--text_field',
        type=str,
        required=True,
        help='Specify the name of the field/column in the dataset where we find the text (e.g. \'sentence\', \'text\').'
    )
    parser.add_argument(
        '--label_field',
        type=str,
        required=True,
        help='Specify the name of the field/column in the dataset where we find the label (e.g. \'label1\', \'label\').'
    )
    parser.add_argument(
        '--premise_shortening',
        nargs='?', 
        choices=['full_premise', 'first_sentence'],
        default='full',
        required=False,
        help='Specify the premise shortening setup: full_premise for no shortening and first_sentence to use only the first sentence as premise.'
    )
    parser.add_argument(
        '--template_num',
        type=int,
        required=True,
        help='Specify the number (int) of the template from templates.py.'
    )
    parser.add_argument(
        '--label_verbalization',
        type=str,
        required=True,
        help='Provide the name of a label verbalization mapping (dictionary) inside \'verbalizations.py\'; for instance, \'verbalizations_fine_grained\'. Create your own dictionary to use your own verbalizations.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to the directory where to save the json with the data converted into the entailment format.'
    )
    parser.add_argument(
        '--use_pipeline',
        type=str,
        default="True",
        required=False,
        help='Use the HF zero-shot text classification or not to perform inference. Values: True or False.'
    )



# Auxiliary function to split sentences by separators while keeping the separator in new list
def splitkeep(s, delimiter):
    split = s.split(delimiter)
    return [substr + delimiter for substr in split[:-1]]

def first_sent_premise(premise_to_shorten):
    all_premise_sents = splitkeep(premise_to_shorten, '.')
    premise = all_premise_sents[0]
    return ''.join(premise)



def main(args):
    # Load the dataset
    global data
    try:
        data = load_dataset(args.data_path)

    except FileNotFoundError:
        print("Invalid data path.")
    try:
        data = data[args.data_split]
    except KeyError:
        print("Invalid data split name. Valid names:", list(data.keys()))
    print("Data loaded")

    # Load the model and tokenizer
    nli_model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("Model loaded")

    # Use GPU whenever possible and transfer model to it
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    nli_model.to(device)

    def convert_ids_to_label_names(example):
        label_id = example[args.label_field] 
        label_name = data.features[args.label_field].int2str(label_id)
        example[args.label_field] = label_name 
        return example

    # Get the labels
    labels = data.features[args.label_field].names
    print("Labels:",labels)

    # Load the template and verbalizations
    template = templates.templates[args.template_num]
    selected_verbalizations = getattr(verbalizations, args.label_verbalization)
    selected_verbalizations_inv = {v: k for k, v in selected_verbalizations.items()}
    print("Template:", template)
    print("Verbalizations:", selected_verbalizations)
    print("Premise shortened to the first sentence:", args.premise_shortening == 'first_sentence')

    # Initialize list to store predicted labels, all entailment probabilities and all probabilities as dictionaries
    correct_labels = []
    predicted_labels = []
    all_entailment_probs = []
    #all_probs_list = []


    if args.use_pipeline == "False":   
        print("Inference with no HF pipeline.")    
        # Loop over the data and 
        for i, ex in tqdm(enumerate(data)):
            ex = convert_ids_to_label_names(ex)
            correct_label = ex[args.label_field]
            correct_labels.append(correct_label)
            premise = ex[args.text_field]

            # Apply premise shortening strategy
            if args.premise_shortening == 'first_sentence': 
                premise = first_sent_premise(premise)
             
            # Create the batch of premise-hypothesis pairs
            # for the ith example
            batch = []
            for label in labels:
                hypothesis = f"{template.format(selected_verbalizations[label])}"
                batch.append([premise, hypothesis])
            
            # 1) Encode input pairs
            batch_encoded = tokenizer(batch,padding=True,truncation=True,return_tensors="pt").to(device)

            # 2) Obtain the logits for [entailment, neutral, contradiction]
            classification_logits = nli_model(**batch_encoded).logits

            # 3) Calculate softmax probabilities
            # Softmax of entailment probabilities
            softmax_results = torch.softmax(classification_logits, dim=0).tolist()
            # Softmax within each [entailment, neutral, contradiction] (usually used for multi-label classification)
            #softmax_results_per_class = [torch.softmax(logits, dim=0).tolist() for logits in classification_logits]

            # Get only the entailment probabilities
            results_entailment = [ent_prob[0] for ent_prob in softmax_results]
            dic_ordered_entailment_probs = {label:results_entailment[j] for j,label in enumerate(labels)}
            all_entailment_probs.append(dic_ordered_entailment_probs)

            # Get the index of the element with highest probability and map it to the label, which will be the predicted label
            index_highest_prob = np.argmax(results_entailment)
            predicted_label = labels[index_highest_prob]

            # Save the predicted label in the list
            predicted_labels.append(predicted_label)

            # Store all softmax probabilities (entailment, neutral, contradiction) for each label
            #dic_ordered_all_probs = {label:softmax_results_per_class[j] for j,label in enumerate(labels)}
            #all_probs_list.append(dic_ordered_all_probs)

    else:
        print("Inference with HF pipeline.")    
        # Load the zero-shot-classification pipeline if chosen
        classifier = pipeline("zero-shot-classification", model=nli_model, tokenizer=tokenizer, hypothesis_template=template, device=0)
        candidates = [selected_verbalizations[label] for label in labels]
        # Loop over the data and 
        for i, ex in tqdm(enumerate(data)):
            ex = convert_ids_to_label_names(ex)
            correct_label = ex[args.label_field]
            correct_labels.append(correct_label)
            premise = ex[args.text_field]

            # Apply premise shortening strategy
            if args.premise_shortening == 'first_sentence': 
                premise = first_sent_premise(premise)

            results_entailment = classifier(premise, candidates)

            # Get the predicted label and save it in the list
            predicted_label = results_entailment['labels'][0]
            predicted_label = selected_verbalizations_inv[predicted_label]
            predicted_labels.append(predicted_label)

            # Get only the entailment probabilities
            all_entailment_probs = results_entailment['scores']
            #dic_ordered_entailment_probs = dict(sorted(dic_entailment_probs.items()))
            #all_entailment_probs.append(dic_ordered_entailment_probs)
            #breakpoint()

            # Calculate and store the softmax entailment probabilities within each [entailment, neutral, contradiction] for each label
            #softmax_results_per_class = classifier(premise, candidates, multi_label=True)
            #dic_all_probs = {candidate_labels_dict[label_template]:softmax_results_per_class['scores'][i] for i,label_template in enumerate(softmax_results_per_class['labels'])}
            #dic_ordered_all_probs = dict(sorted(dic_all_probs.items()))
            #all_probs_list.append(dic_ordered_all_probs)
            #breakpoint()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'predicted_labels.txt'), "w") as f:
        print(predicted_labels, file = f)
    with open(os.path.join(args.output_dir, 'entailment_probabilities.txt'), "w") as f:
        print(all_entailment_probs, file = f)
    with open(os.path.join(args.output_dir, 'gold_labels.txt'), "w") as f:
        print(correct_labels, file = f)
    report = classification_report(correct_labels, predicted_labels, digits=4)
    with open(os.path.join(args.output_dir, 'report.txt'), "w") as f:
        print(report, file = f)

    return 0



if __name__ == "__main__":
    description = """Code detection"""
    parser = ArgumentParser(description=description, formatter_class=ArgumentDefaultsHelpFormatter)
    add_base_arguments_to_parser(parser)
    args = parser.parse_args()
    main(args)


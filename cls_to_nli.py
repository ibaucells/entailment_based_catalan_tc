"""
Convert classification data to entailment-task data using a template and verbalization.

Example of use:
python cls_to_nli.py --data_path "projecte-aina/tecla" \
                     --data_split "test" \
                     --text_field "text" \
                     --label_field "label1" \
                     --premise_shortening "full_premise" \
                     --template_num 7 \
                     --label_verbalization "verbalizations_templateset1_coarse_grained" \
                     --output_dir "."
"""
import os
import argparse
from datasets import load_dataset
import templates, verbalizations
import random
import json
from tqdm import tqdm

def add_base_arguments_to_parser(parser):
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the HF dataset or to dataloader.'
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
        '--num_non_entailment',
        type=str,
        default='all',
        required=False,
        help='Number of non-enatilment instances created per each entailment one. Use \'all\' to create all possible non-entailment pairs (as many as the number of labels in the dataset).'
    )
    parser.add_argument(
        '--template_num',
        type=int,
        required=True,
        help='Specify the number (int) of the template in template.py.'
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


# Auxiliary function to split sentences by separators while keeping the separator in new list
def splitkeep(s, delimiter):
    split = s.split(delimiter)
    return [substr + delimiter for substr in split[:-1]]

def get_first_sent_premise(premise_to_shorten):
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
        exit()
    try:
        data = data[args.data_split]
    except KeyError:
        print("Invalid data split name. Valid names:", list(data.keys()))
        exit()
    print("Data loaded")

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
    print("Templates and verbalizations loaded")

    # Initialize the list to keep the transformed data
    nli_data = []

    # Loop over the data and 
    for i, ex in tqdm(enumerate(data), desc="Processing"):
        ex = convert_ids_to_label_names(ex)
        # Store the text as the new premise and shorten it if specified
        try:
            premise = ex[args.text_field]
        except KeyError:
            print("Invalid text name field. Valid names must be found in:", list(ex.keys()))
        if args.premise_shortening == "first_sentence":
            premise = get_first_sent_premise(premise)
        
        # Get the true label
        correct_label = ex[args.label_field]

        # Create the entailment pair and all possible non-entailment pairs if specified in args
        if args.num_non_entailment == 'all':
            for ii, label in enumerate(labels):
                label_bool = 0 if label == correct_label else 1
                hypothesis = f"{template.format(selected_verbalizations[label])}"
                example = {'id': str(i)+f'_{ii}', 'premise': premise, 'hypothesis': hypothesis, 'label': label_bool, 'cls_label': label}
                #nli_pairs.append(example)
                #nli_data.append(nli_pairs)
                nli_data.append(example)
        # Create the entailment pair and the number of non-entailment pairs per each specified in args
        else:
            for ii, label in enumerate(random.sample(labels, int(args.num_non_entailment))):
                label_bool = 0 if label == correct_label else 1
                hypothesis = f"{template.format(selected_verbalizations[label])}"
                example = {'id': str(i)+f'_{ii}', 'premise': premise, 'hypothesis': hypothesis, 'label': label_bool, 'cls_label': label}
                nli_data.append(example)
    
    # Save the entailment data into a json file
    with open(os.path.join(args.output_dir, f"nli_{args.data_split}.json"), "w") as outfile:
        json.dump(nli_data, outfile, indent=4)
    print("Data saved at:", os.path.join(args.output_dir, f"nli_{args.data_split}.json"))

if __name__ == "__main__":
    description = """Convert a classification dataset to the NLI format."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_base_arguments_to_parser(parser)
    args = parser.parse_args()
    main(args)
# Entailment-based Task Transfer for Catalan Text Classification in Small Data Regimes

This is the repo for the paper "[*Entailment-based Task Transfer for Catalan Text Classification in Small Data Regimes*]", which will be published in the SEPLN 2023 congress. 

1. Code to reformulate a Text Classification (TC) dataset as entailment. Arguments:

    --data_path: Specifies the path to the dataset in HF used for classification. 
    --data_split: Specifies the data split to be used (e.g., "train", "test", "validation").
    --text_field: Specifies the field name or key in the dataset that contains the text data.
    --label_field: Specifies the field name or key in the dataset that contains the label data.
    --template_num: Specifies the template (specified in templates.py) to be used for hypothesis generation.
    --label_verbalization: Specifies the label verbalizations (from verbalizations.py) to be used for the hypothesis generation.
    --output_dir: Specifies the directory where the converted dataset will be saved.

For instance, to reformulate the TeCla test set (coarse-grained task) using the template "Aquest text tracta sobre {}.", run the following command in the terminal:
```
python cls_to_nli.py --data_path "projecte-aina/tecla" \
                     --data_split "test" \
                     --text_field "text" \
                     --label_field "label1" \
                     --premise_shortening "full_premise" \
                     --template_num 7 \
                     --label_verbalization "verbalizations_templateset1_coarse_grained" \
                     --output_dir "."
```

2. Code to fine-tune and evaluate a model (pre-trained model or NLI model) on the specified TC dataset reformulated as entailment. Specify the desired hyperparameters in finetune.sh and run in the terminal:
```
sh finetune.sh
```

3. Code to evaluate an entailment model on a text classification task (conversion to NLI at inference time). The arguments needed are the same as those specified for cls_to_nli.py and the model_path (stored in local or in HF).
```
python test.py --data_path "projecte-aina/tecla" \
               --model_path "symanto/xlm-roberta-base-snli-mnli-anli-xnli" \
               --data_split "test" \
               --text_field "text" \
               --label_field "label1" \
               --template_num 7 \
               --label_verbalization "verbalizations_templateset1_coarse_grained" \
               --output_dir "."
```

Additionally, we have made accesible [CaWikiTC](https://huggingface.co/ibaucells/CaWikiTC), the Wikipedia-based dataset automatically created as the task transfer source, and the task-transfer [RoBERTa-ca-CaWikiTC](https://huggingface.co/ibaucells/RoBERTa-ca-CaWikiTC) model trained on this data converted to NLI .


## 📣 Citation
(pending)

## 📫 Contact

For further information, please contact <irene.baucells@bsc.es>.

## ⚖️ License

This work is distributed under a [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

import joblib
import numpy as np
import scipy.special as sp
from transformers import LukeTokenizer, LukeForEntityPairClassification
import json
import torch
from tqdm import trange
import pdb
import random
import argparse

def getF1Micro(key, prediction):
    correct_by_relation = ((key == prediction) & (prediction != 0)).astype(np.int32).sum()
    guessed_by_relation = (prediction != 0).astype(np.int32).sum()
    gold_by_relation = (key != 0).astype(np.int32).sum()

    prec_micro = 1.0
    if guessed_by_relation > 0:
        prec_micro = float(correct_by_relation) / float(guessed_by_relation)
    recall_micro = 1.0
    if gold_by_relation > 0:
        recall_micro = float(correct_by_relation) / float(gold_by_relation)
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    return f1_micro

# This function loads the dataset .json file and produce the list containing the relation extraction instances.
def load_examples(dataset_file):
    with open(dataset_file, "r") as f:
        data = json.load(f)

    examples = []
    for i, item in enumerate(data):
        tokens = item["token"]
        token_spans = dict(
            subj=(item["subj_start"], item["subj_end"] + 1),
            obj=(item["obj_start"], item["obj_end"] + 1)
        )

        if token_spans["subj"][0] < token_spans["obj"][0]:
            entity_order = ("subj", "obj")
        else:
            entity_order = ("obj", "subj")

        text = ""
        cur = 0
        char_spans = {}
        for target_entity in entity_order:
            token_span = token_spans[target_entity]
            text += " ".join(tokens[cur : token_span[0]])
            if text:
                text += " "
            char_start = len(text)
            text += " ".join(tokens[token_span[0] : token_span[1]])
            char_end = len(text)
            char_spans[target_entity] = (char_start, char_end)
            text += " "
            cur = token_span[1]
        text += " ".join(tokens[cur:])
        text = text.rstrip()

        examples.append(dict(
            text=text,
            entity_spans=[tuple(char_spans["subj"]), tuple(char_spans["obj"])],
            label=item["relation"],
            entity_type = (item['subj_type'], item['obj_type']),
        ))

    return examples

parser = argparse.ArgumentParser()

parser.add_argument("--input_file", default="test.json", type=str)
parser.add_argument("--output_file", default="luke_pred.output", type=str)

args = parser.parse_args()

# test.json is the file containing the test set of the TACRED dataset
test_examples = load_examples(args.input_file)

# Load the model checkpoint
model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
model.eval()
model.to("cuda")

luke_id_to_label = model.config.id2label
luke_label_to_id = {value_ : key_ for key_, value_ in luke_id_to_label.items()}
# org_to_luke = [luke_label_to_id[ID_TO_LABEL[i_]] for i_ in range(len(luke_label_to_id.values()))]

# Load the tokenizer
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")

batch_size = 128

# produce the original testing result on the test set of TACRED
pred_ls = []
label_ls = []

for batch_start_idx in trange(0, len(test_examples), batch_size):
    batch_examples = test_examples[batch_start_idx:batch_start_idx + batch_size]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]
    gold_labels = [example["label"] for example in batch_examples]
    label_ls.extend(gold_labels)

    inputs = tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)
    inputs = inputs.to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_indices = outputs.logits.argmax(-1)
    predicted_labels = [model.config.id2label[index.item()] for index in predicted_indices]
    pred_ls.append(outputs.logits.cpu().numpy())
    

luke_prob = np.concatenate(pred_ls, axis = 0)
luke_prob = sp.softmax(luke_prob, axis = 1)

keys = np.array([luke_label_to_id[label_] for label_ in label_ls])
luke_preds = luke_prob.argmax(1)
wrong_id_ls = [i_ for i_ in range(len(keys)) if keys[i_] != luke_preds[i_] and keys[i_] != 0]
wrong_example_ls = [test_examples[i_] for i_ in wrong_id_ls]

print('f1 micro score before bias mitigation: ', getF1Micro(keys, luke_preds))

batch_size = 512
# produce the counterfactual predictions to distill the entity bias
pred_ls = []

for batch_start_idx in trange(0, len(test_examples), batch_size):
    batch_examples = test_examples[batch_start_idx:batch_start_idx + batch_size]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]
    
    texts = [
            texts[i_][entity_spans[i_][0][0] : entity_spans[i_][0][1]] 
            + 
            ' '
            +
            texts[i_][entity_spans[i_][1][0] : entity_spans[i_][1][1]]
            for i_ in range(len(texts))
            ]
    entity_spans = [
                    [(0, entity_spans[i_][0][1] - entity_spans[i_][0][0]),
                    (entity_spans[i_][0][1] - entity_spans[i_][0][0] + 1, 
                     entity_spans[i_][0][1] - entity_spans[i_][0][0] + 1 + entity_spans[i_][1][1] - entity_spans[i_][1][0]
                    )]
                    for i_ in range(len(texts))
                   ]

    inputs = tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)
    inputs = inputs.to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)

    pred_ls.append(outputs.logits.cpu().numpy())
    
new_luke_prob_bias = np.concatenate(pred_ls, axis = 0)
new_luke_prob_bias = sp.softmax(new_luke_prob_bias, axis = 1)
new_luke_preds_bias = new_luke_prob_bias.argmax(1)

print('f1 micro score with only entity names: ', getF1Micro(keys, new_luke_preds_bias))

joblib.dump((keys, luke_id_to_label, luke_preds, new_luke_preds_bias, new_luke_prob_bias), args.output_file)
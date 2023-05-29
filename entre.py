import joblib
import numpy as np
import scipy.special as sp
from transformers import LukeTokenizer, LukeForEntityPairClassification
import json
import torch
from tqdm import trange
import pdb
import random
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch.cuda.amp import autocast
from transformers import AutoConfig, AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader

class REModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('roberta-large', config=config)
        hidden_size = config.hidden_size
        self.loss_fnt = nn.CrossEntropyLoss()
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0),
            nn.Linear(hidden_size, 42)
        )

    @autocast()
    def forward(self, input_ids=None, attention_mask=None, labels=None, ss=None, os=None):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = outputs[0]
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        ss_emb = pooled_output[idx, ss]
        os_emb = pooled_output[idx, os]
        h = torch.cat((ss_emb, os_emb), dim=-1)
        logits = self.classifier(h)
        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fnt(logits.float(), labels)
            outputs = (loss,) + outputs
        return outputs

def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
        return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token


class Processor:
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def tokenize(self, tokens, subj_type, obj_type, ss, se, os, oe):
        """
        Implement the following input formats:
            - entity_mask: [SUBJ-NER], [OBJ-NER].
            - entity_marker: [E1] subject [/E1], [E2] object [/E2].
            - entity_marker_punct: @ subject @, # object #.
            - typed_entity_marker: [SUBJ-NER] subject [/SUBJ-NER], [OBJ-NER] obj [/OBJ-NER]
            - typed_entity_marker_punct: @ * subject ner type * subject @, # ^ object ner type ^ object #
        """
        sents = []
        subj_type = self.tokenizer.tokenize(subj_type.replace("_", " ").lower())
        obj_type = self.tokenizer.tokenize(obj_type.replace("_", " ").lower())

        for i_t, token in enumerate(tokens):
            if i_t >= 1:
                tokens_wordpiece = self.tokenizer.tokenize(' ' + token)
            else:
                tokens_wordpiece = self.tokenizer.tokenize(token)

            if i_t == ss:
                new_ss = len(sents)
                tokens_wordpiece = ['@'] + ['*'] + subj_type + ['*'] + tokens_wordpiece
            if i_t == se:
                tokens_wordpiece = tokens_wordpiece + ['@']
            if i_t == os:
                new_os = len(sents)
                tokens_wordpiece = ["#"] + ['^'] + obj_type + ['^'] + tokens_wordpiece
            if i_t == oe:
                tokens_wordpiece = tokens_wordpiece + ["#"]

            sents.extend(tokens_wordpiece)

        sents = sents[:512 - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(sents)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        return input_ids, new_ss + 1, new_os + 1


class TACREDProcessor(Processor):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.LABEL_TO_ID = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}

    def read(self, file_in, id_ls):
        features = []
        with open(file_in, "r") as fh:
            data = json.load(fh)
            
        data = [data[id_] for id_ in id_ls]

        for d in tqdm(data):
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']

            tokens = d['token']
                        
            tokens = [convert_token(token) for token in tokens]

            input_ids, new_ss, new_os = self.tokenize(tokens, d['subj_type'], d['obj_type'], ss, se, os, oe)
            rel = self.LABEL_TO_ID[d['relation']]

            if 'org_id' in d:
                org_id = d['org_id']
            else:
                org_id = -1
            
            feature = {
                'input_ids': input_ids,
                'labels': rel,
                'ss': new_ss,
                'os': new_os,
                'relation' : d['relation'],
                'text' : tokens,
                'org_id' : org_id,
            }

            features.append(feature)
        return features
    
def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    ss = [f["ss"] for f in batch]
    os = [f["os"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    ss = torch.tensor(ss, dtype=torch.long)
    os = torch.tensor(os, dtype=torch.long)
    output = (input_ids, input_mask, labels, ss, os)
    return output

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

def replaceNameJson(input_file, output_file, subj_id_ls, obj_id_ls, subj_name_ls, obj_name_ls):
    with open(input_file, "r") as f:
        input_data = json.load(f)
        
    output_data = input_data[:]
    for i_ in range(len(subj_id_ls)):
        org_subj_name_len = output_data[subj_id_ls[i_]]['subj_end'] - output_data[subj_id_ls[i_]]['subj_start'] + 1
        output_data[subj_id_ls[i_]]['token'] = output_data[subj_id_ls[i_]]['token'][: output_data[subj_id_ls[i_]]["subj_start"]] + \
                                                subj_name_ls[i_] + \
                                                output_data[subj_id_ls[i_]]['token'][output_data[subj_id_ls[i_]]["subj_end"] + 1 :]
        output_data[subj_id_ls[i_]]['subj_end'] = output_data[subj_id_ls[i_]]['subj_start'] + len(subj_name_ls[i_]) - 1
        if output_data[subj_id_ls[i_]]['obj_start'] > output_data[subj_id_ls[i_]]['subj_start']:
            output_data[subj_id_ls[i_]]['obj_start'] = output_data[subj_id_ls[i_]]['obj_start'] + len(subj_name_ls[i_]) - org_subj_name_len
            output_data[subj_id_ls[i_]]['obj_end'] = output_data[subj_id_ls[i_]]['obj_end'] + len(subj_name_ls[i_]) - org_subj_name_len
            
    for i_ in range(len(obj_id_ls)):
        org_obj_name_len = output_data[obj_id_ls[i_]]['obj_end'] - output_data[obj_id_ls[i_]]['obj_start'] + 1
        output_data[obj_id_ls[i_]]['token'] = output_data[obj_id_ls[i_]]['token'][: output_data[obj_id_ls[i_]]["obj_start"]] + \
                                                obj_name_ls[i_] + \
                                                output_data[obj_id_ls[i_]]['token'][output_data[obj_id_ls[i_]]["obj_end"] + 1 :]
        output_data[obj_id_ls[i_]]['obj_end'] = output_data[obj_id_ls[i_]]['obj_start'] + len(obj_name_ls[i_]) - 1
        if output_data[obj_id_ls[i_]]['subj_start'] > output_data[obj_id_ls[i_]]['obj_start']:
            output_data[obj_id_ls[i_]]['subj_start'] = output_data[obj_id_ls[i_]]['subj_start'] + len(obj_name_ls[i_]) - org_obj_name_len
            output_data[obj_id_ls[i_]]['subj_end'] = output_data[obj_id_ls[i_]]['subj_end'] + len(obj_name_ls[i_]) - org_obj_name_len

    with open(output_file, "w") as f:
        json.dump(output_data, f)
        
def onlyNameJson(input_file, output_file, subj_id_ls, obj_id_ls, subj_name_ls, obj_name_ls):
    with open(input_file, "r") as f:
        input_data = json.load(f)
        
    org_subj_name_ls = [data_['token'][data_['subj_start'] : data_['subj_end'] + 1] for data_ in input_data]
    org_obj_name_ls = [data_['token'][data_['obj_start'] : data_['obj_end'] + 1] for data_ in input_data]
    
    new_subj_name_ls = org_subj_name_ls[:]
    new_obj_name_ls = org_obj_name_ls[:]
    
    for i_ in range(len(subj_id_ls)):
        new_subj_name_ls[subj_id_ls[i_]] = subj_name_ls[i_]
    
    for i_ in range(len(obj_id_ls)):
        new_obj_name_ls[obj_id_ls[i_]] = obj_name_ls[i_]
        
    output_data = input_data[:]
    for id_ in range(len(output_data)):
        output_data[id_]['token'] = new_subj_name_ls[id_] + new_obj_name_ls[id_]
        output_data[id_]['subj_start'], output_data[id_]['subj_end'] = 0, len(new_subj_name_ls[id_]) - 1
        output_data[id_]['obj_start'], output_data[id_]['obj_end'] = len(new_subj_name_ls[id_]), len(new_subj_name_ls[id_]) + len(new_obj_name_ls[id_]) - 1

    with open(output_file, "w") as f:
        json.dump(output_data, f)
        
def sampleName(input_file, subj_id_ls, obj_id_ls):
    with open(input_file, "r") as f:
        input_data = json.load(f)
        
    person_name_ls = joblib.load('./wiki_person.output')
    organization_name_ls = joblib.load('./wiki_organization.output')
    
    random.shuffle(person_name_ls)
    random.shuffle(organization_name_ls)
    
    person_counter = 0
    organization_counter = 0
    
    subj_name_ls = []
    for i_ in range(len(subj_id_ls)):
        if input_data[subj_id_ls[i_]]['subj_type'] == 'PERSON':
            subj_name_ls.append([person_name_ls[person_counter]])
            person_counter += 1
        elif input_data[subj_id_ls[i_]]['subj_type'] == 'ORGANIZATION':
            subj_name_ls.append([organization_name_ls[organization_counter]])
            organization_counter += 1
        else:
            assert 1 == 0
            
    random.shuffle(person_name_ls)
    random.shuffle(organization_name_ls)
    
    person_counter = 0
    organization_counter = 0
    
    obj_name_ls = []
    for i_ in range(len(obj_id_ls)):
        if input_data[obj_id_ls[i_]]['obj_type'] == 'PERSON':
            obj_name_ls.append([person_name_ls[person_counter]])
            person_counter += 1
        elif input_data[obj_id_ls[i_]]['obj_type'] == 'ORGANIZATION':
            obj_name_ls.append([organization_name_ls[organization_counter]])
            organization_counter += 1
        else:
            assert 1 == 0
            
    return subj_name_ls, obj_name_ls

def lukeInference(input_file, id_ls):
    test_examples = load_examples(input_file)
    test_examples = [test_examples[id_] for id_ in id_ls]

    # Load the model checkpoint
    model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
    model.eval()
    model.to("cuda")

    luke_id_to_label = model.config.id2label
    luke_label_to_id = {value_ : key_ for key_, value_ in luke_id_to_label.items()}

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

    return keys, luke_preds

def ireInference(input_file, id_ls):
    config = AutoConfig.from_pretrained(
        'roberta-large',
        num_labels=42,
    )
    config.gradient_checkpointing = True
    tokenizer = AutoTokenizer.from_pretrained(
        'roberta-large',
    )
    
    model = REModel(config)
    model.to(0)
    model.load_state_dict(torch.load('748_re.model')) 
    
    processor = TACREDProcessor(tokenizer)
    test_features = processor.read(input_file, id_ls)

    dataloader = DataLoader(test_features, batch_size=256, collate_fn=collate_fn, drop_last=False)
    keys, preds, pred_prob = [], [], []
    for i_b, batch in enumerate(dataloader):
        model.eval()
        inputs = {'input_ids': batch[0].to("cuda:0"),
                  'attention_mask': batch[1].to("cuda:0"),
                  'ss': batch[3].to("cuda:0"),
                  'os': batch[4].to("cuda:0"),
                  }
        keys += batch[2].tolist()
        with torch.no_grad():
            logit = model(**inputs)[0]
            pred = torch.argmax(logit, dim=-1)
        preds += pred.tolist()
        pred_prob.extend(F.softmax(logit).tolist())

    keys = np.array(keys, dtype=np.int64)
    preds = np.array(preds, dtype=np.int64)

    return keys, preds
    

_, subj_id_ls, obj_id_ls = joblib.load('final_id_resample_ls.output')
care_id_ls = sorted(list(set(subj_id_ls + obj_id_ls)))
replaceNameJson('test.json', 'output.json', [], [], [], [])
onlyNameJson('test.json', 'output_only_name.json', [], [], [], [])
key_array, pred_array = lukeInference('test.json', list(range(15509)))
# key_array, pred_array = ireInference('test.json', list(range(15509)))
print('original f1 score: ' , getF1Micro(key_array[care_id_ls], pred_array[care_id_ls]))

for replace_id_ in range(10000):
    print('replace time: ', replace_id_)
    right_id_ls = [id_ for id_ in care_id_ls 
                  if key_array[id_] == pred_array[id_]
                   and key_array[id_] != 0
                 ]
    subj_id_ls = [id_ for id_ in subj_id_ls if id_ in right_id_ls]
    obj_id_ls = [id_ for id_ in obj_id_ls if id_ in right_id_ls]
    subj_name_ls, obj_name_ls = sampleName('output.json', subj_id_ls, obj_id_ls)
    replaceNameJson('output.json', 'output.json', subj_id_ls, obj_id_ls, subj_name_ls, obj_name_ls)
    right_key_array, right_pred_array = lukeInference('output.json', right_id_ls)
    # right_key_array, right_pred_array = ireInference('output.json', right_id_ls)
    key_array[right_id_ls], pred_array[right_id_ls] = right_key_array, right_pred_array
    print('f1 score: ' , getF1Micro(key_array[care_id_ls], pred_array[care_id_ls]))

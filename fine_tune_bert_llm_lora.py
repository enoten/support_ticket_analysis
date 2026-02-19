from datasets import load_dataset
from datasets import DatasetDict, Dataset

import matplotlib.pyplot as plt

from transformers import (logging,
    AutoTokenizer, DataCollatorWithPadding, 
    AutoModelForSequenceClassification, TrainingArguments, Trainer
    )
import evaluate
import numpy as np
import pandas as pd

from peft import LoraConfig, get_peft_model, TaskType

# ignore warnings from transformers lib
logging.set_verbosity_error() 

support_tickets = load_dataset("phi-ai-info/support_tickets",name='alpha')
print(support_tickets)

def dataset_to_df(dataset_obj, 
                  label=None,
                  reference='class',
                  new_key='label'
                 ):
    cols = list(dataset_obj[0].keys())

    cols_values = {item:[] for item in cols}
    for json_rec in dataset_obj:
        for item in cols:
            cols_values[item].append(json_rec[item])
    
    if label is not None:
        cols += [new_key]
        cols_values[new_key] = []
        for json_rec in dataset_obj:
            cols_values[new_key].append(label[json_rec[reference]])
    
    return pd.DataFrame({item:cols_values[item] for item in cols})

label_class_dict = {'revoke access': 'access',
 'grant access': 'access',
 'access profile': 'access',
 'add user': 'user',
 'delete user': 'user',
 'create user': 'user',
 'modify user': 'user',
 'user role': 'user',
 'disk space': 'storage',
 'hard disk': 'storage',
 'disk full': 'storage',
 'ssd disk': 'storage',
 'disk error': 'storage',
 'shared disk': 'storage',
 'nas disk': 'storage',
 'printer functioning': 'printer',
 'printer driver': 'printer',
 'printer toner': 'printer',
 'printer paper': 'printer',
 'wifi functioning': 'network',
 'network functioning': 'network',
 'email server': 'servers',
 'web server': 'servers'}

label_name_dict = {item: i for i,item in enumerate(list(set(label_class_dict.values())))}
print(label_name_dict)

label_dict = {'grant access': 0,
 'revoke access': 0,
 'access profile': 0,
 'disk space': 1,
 'disk full': 1,
 'disk error': 1,
 'add user': 2,
 'delete user': 2,
 'create user': 2,
 'modify user': 2}

df_set = dataset_to_df(support_tickets['train'], 
                       label=label_name_dict)

#create random index for shuffling the dataset
random_index = np.random.permutation(df_set.index)

subsets = ['train','valid','test']
train_ratio = 0.6
lng = len(random_index)
train_start = 0
train_end = int(lng*train_ratio)

val_ratio = 0.2
val_start = train_end+1
val_end = int(lng*(train_ratio+val_ratio))

test_start = val_end + 1 
test_end = lng

split_ind = np.array([[train_start,train_end],
                      [val_start,val_end],
                      [test_start,test_end]])

ind2subset = {key:random_index[split_ind[i][0]:split_ind[i][1]] for i,key in enumerate(subsets)}

ds_splits = DatasetDict({
                        'train': Dataset.from_pandas(
                                                    df_set.iloc[ind2subset['train']]
                                                    ),
                        'valid': Dataset.from_pandas(
                                                    df_set.iloc[ind2subset['valid']]
                                                    ),
                        'test': Dataset.from_pandas(
                                                    df_set.iloc[ind2subset['test']]
                                                    )
                        })
print(ds_splits)

model_path = "google-bert/bert-base-uncased"
#model_path = 'distilbert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(model_path)

label2id = label_name_dict.copy()
id2label = {label2id[key]:key for key in label2id}
#id2label = {0: "access", 1: "disk", 2: "user"}
#label2id = {id2label[key]:key for key in id2label}

# model = AutoModelForSequenceClassification.from_pretrained(model_path, 
#                                                            num_labels=len(list(id2label.keys())), 
#                                                            id2label=id2label, 
#                                                            label2id=label2id
#                                                           )
# for name, param in model.named_parameters():
#    print(name, param.requires_grad)

model_path_dict = {}
model_path_dict['lora'] = 'distilbert-base-uncased'

label2id = label_name_dict.copy()
id2label = {label2id[key]:key for key in label2id}
#id2label = {0: "access", 1: "disk", 2: "user"}
#label2id = {id2label[key]:key for key in id2label}

model_dict = {}

model_dict['lora'] = AutoModelForSequenceClassification.from_pretrained(model_path_dict['lora'], 
                                                                        num_labels=len(list(id2label.keys())), 
                                                                        id2label=id2label, 
                                                                        label2id=label2id
                                                                       )

tokenizer_dict = {}
tokenizer_dict['lora'] = AutoTokenizer.from_pretrained(model_path_dict['lora'], 
                                                       add_prefix_space=True)

# add pad token if none exists
if tokenizer_dict['lora'].pad_token is None:
    tokenizer_dict['lora'].add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer_dict['lora']))


# create tokenize function
def tokenize_function_lora(examples):
    # extract text
    text = examples["description"]

    #tokenize and truncate text
    tokenizer_dict['lora'].truncation_side = "left"
    tokenized_inputs = tokenizer_dict['lora'](text,
                                              return_tensors="np",
                                              truncation=True,
                                              max_length=512
                                             )

    return tokenized_inputs

# tokenize training and validation datasets

tokenized_dataset_dict = {}
tokenized_dataset_dict['lora'] = ds_splits.map(tokenize_function_lora, 
                                               batched=True)
#tokenized_dataset_dict['lora']

# create data collator
data_collator_dict = {}
data_collator_dict['lora'] = DataCollatorWithPadding(tokenizer=tokenizer_dict['lora'])

# import accuracy evaluation metric
accuracy = evaluate.load("accuracy")

# define an evaluation function to pass into trainer later
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

for name, param in model_dict['lora'].named_parameters():
   print(name, param.requires_grad)


peft_config = LoraConfig(task_type="SEQ_CLS",
                        r=4,
                        lora_alpha=32,
                        lora_dropout=0.01,
                        target_modules = ['q_lin','k_lin','v_lin'])

model_dict['lora'] = get_peft_model(model_dict['lora'], 
                                    peft_config)
model_dict['lora'].print_trainable_parameters()

# hyperparameters
lr = 1e-3
batch_size = 16
num_epochs = 10

train_arg_dict = {}
# define training arguments
train_arg_dict['lora'] = TrainingArguments(
                                            output_dir= model_path_dict['lora'] + "-lora-text-classification",
                                            learning_rate=lr,
                                            per_device_train_batch_size=batch_size,
                                            per_device_eval_batch_size=batch_size,
                                            num_train_epochs=num_epochs,
                                            weight_decay=0.01,
                                            eval_strategy="epoch",
                                            save_strategy="epoch",
                                            load_best_model_at_end=True,
                                        )

trainer_dict = {}
trainer_dict['lora'] = Trainer(model=model_dict['lora'],
                               args=train_arg_dict['lora'],
                               train_dataset=tokenized_dataset_dict['lora']["train"],
                               eval_dataset=tokenized_dataset_dict['lora']["valid"],
                               #tokenizer=tokenizer_dict['lora'],
                               data_collator=data_collator_dict['lora'],
                               compute_metrics=compute_metrics)

trainer_dict['lora'].train()

model_dict['lora'].save_pretrained(model_path_dict['lora'] + "-lora-text-classification")
tokenizer_dict['lora'].save_pretrained(model_path_dict['lora'] + "-lora-text-classification")

### Inference on test set
split_key = "test"
# apply model to validation dataset
predictions = trainer_dict['lora'].predict(tokenized_dataset_dict['lora'][split_key])

# Extract the logits and labels from the predictions object
logits = predictions.predictions
labels = predictions.label_ids

# Use your compute_metrics function
#metrics = compute_metrics((logits, labels))
#print(metrics)
true_count = 0
for pred_l,actu_l in zip(labels,ds_splits[split_key]['label']): 
    if pred_l == actu_l:
        true_count += 1
print(f'Accuracy {true_count/len(labels)}')
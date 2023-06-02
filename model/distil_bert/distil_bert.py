import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer
import pandas as pd
from datasets import Dataset
import os
import wandb
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--current_dir", help="Current directory")
argParser.add_argument("-o", "--out_dir", help="Output directory")
args = argParser.parse_args()

wandb.login(key="<our-token-key>")

def deflog(text):
    print(f"[DEF-LOG] {text}")


def load_data(config):
    data = pd.read_csv("normalized_data.csv")
    data = data[data['review_content'].notna()]
    data = data[data['review_type'].notna()]
    x_data = data['review_content']
    y_data = data['review_type']
    y_data = [1 if x == "Fresh" else 0 for x in y_data]

    data = {'train': {'x': [], 'labels': []},
            'val': {'x': [], 'labels': []},
            'test': {'x': [], 'labels': []}}

    X_rem, data['test']['x'], y_rem, data['test']['labels'] = train_test_split(x_data, y_data, test_size=0.1, random_state=42, stratify=y_data)
    data['train']['x'], data['val']['x'], data['train']['labels'], data['val']['labels'] = train_test_split(X_rem, y_rem, test_size=0.1, random_state=42, stratify=y_rem)
    

    if config['train_p'] < 0.999:
        _, data['train']['x'], _, data['train']['labels'] = train_test_split(data['train']['x'], 
                                                data['train']['labels'], 
                                                test_size=config['train_p'], 
                                                random_state=42,
                                                stratify=data['train']['labels'])
    
    if config['test_p'] < 0.999:
        _, data['test']['x'], _, data['test']['labels'] = train_test_split(data['test']['x'], 
                                                data['test']['labels'], 
                                                test_size=config['test_p'], 
                                                random_state=42,
                                                stratify=data['test']['labels'])

    if config['val_p'] < 0.999:
        _, data['val']['x'], _, data['val']['labels'] = train_test_split(data['val']['x'], 
                                                data['val']['labels'], 
                                                test_size=config['val_p'], 
                                                random_state=42,
                                                stratify=data['val']['labels'])



    for phase in ['train', 'val', 'test']:
        data[phase]['x'] = list(data[phase]['x'])
        data[phase]['labels'] = list(data[phase]['labels'])
        data[phase] = Dataset.from_dict(data[phase])
    
    return data


def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}



def run_model(config):
    wandb.init(project=config['proj_name'], name=config['ex_name'], tags=['NEW'])

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def preprocess_function(examples):
        return tokenizer(examples['x'], return_tensors='pt', padding="max_length", max_length=200, truncation=True)

    data = load_data(config)

    tokenized_train = data['train'].map(preprocess_function, batched=True)
    tokenized_val = data['val'].map(preprocess_function, batched=True)
    tokenized_test = data['test'].map(preprocess_function, batched=True)
    tokenized_train = tokenized_train.remove_columns(['x'])
    tokenized_val = tokenized_val.remove_columns(['x'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    for idx, param in enumerate(model.parameters()):
        if idx < config['freeze_layer']:
            param.requires_grad = False
        else:
            param.requires_grad = True

    training_args = TrainingArguments(
            output_dir=f"{config['out_dir']}/{config['ex_name']}",
            learning_rate=config['lr'],
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            num_train_epochs=config['epoch_num'],
            weight_decay=config['weight_decay'],
            save_strategy="epoch",
            push_to_hub=False,
            logging_steps=1,
            evaluation_strategy='steps',
            eval_steps=50,
            metric_for_best_model='f1',
            report_to="wandb",
            lr_scheduler_type="cosine",
            run_name=config['ex_name']
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()

    wandb.finish()



def main():
    os.chdir(args.current_dir)
    config = {
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "train_p": 1,
        "val_p": 1,
        "test_p": 1,
        "epoch_num": 40,
        "batch_size": 1024,
        "class_num": 2,
        "run_id": 2,
        "out_dir": args.out_dir,
        "freeze_layer": 30,
        "proj_name": "BERT-transfomer",
        "ex_name": "distil-bert-transformer-30layer"
    }
    run_model(config)


if __name__ == "__main__":
    main()
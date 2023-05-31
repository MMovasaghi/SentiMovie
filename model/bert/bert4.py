import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import time
import random
import pickle
import os
import shutil
import copy
import sys
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, precision_score, recall_score, roc_curve, auc
from transformers import BertTokenizer, BertModel
import wandb
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--current_dir", help="Current directory")
argParser.add_argument("-o", "--out_dir", help="Output directory")
args = argParser.parse_args()


wandb.login(key="aec6fef7ba56ee445129472eb583718b8e529934")


def deflog(text):
    print(f"[DEF-LOG] {text}")


def load_data(config):
    train_set = pd.read_pickle("train.pkl")
    val_set = pd.read_pickle("val.pkl")
    test_set = pd.read_pickle("test.pkl")

    df_train = pd.DataFrame.from_dict(train_set)
    df_test =  pd.DataFrame.from_dict(test_set)
    df_val =  pd.DataFrame.from_dict(val_set)
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    df_val = df_val.dropna()

    data = {'train': {}, "test": {}, 'val': {}}
    data['train']['x'] = list(df_train['x'])
    data['train']['y'] = list(df_train['y'])
    data['test']['x'] = list(df_test['x'])
    data['test']['y'] = list(df_test['y'])
    data['val']['x'] = list(df_val['x'])
    data['val']['y'] = list(df_val['y'])

    if config['train_p'] < 0.999:
        _, data['train']['x'], _, data['train']['y'] = train_test_split(data['train']['x'], 
                                                data['train']['y'], 
                                                test_size=config['train_p'], 
                                                random_state=42,
                                                stratify=data['train']['y'])
    
    if config['test_p'] < 0.999:
        _, data['test']['x'], _, data['test']['y'] = train_test_split(data['test']['x'], 
                                                data['test']['y'], 
                                                test_size=config['test_p'], 
                                                random_state=42,
                                                stratify=data['test']['y'])

    if config['val_p'] < 0.999:
        _, data['val']['x'], _, data['val']['y'] = train_test_split(data['val']['x'], 
                                                data['val']['y'], 
                                                test_size=config['val_p'], 
                                                random_state=42,
                                                stratify=data['val']['y'])
    
    return data


class myModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(myModel, self).__init__()
        self.bert = bert_model
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(768, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.soft = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, cls_hs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.classifier(cls_hs)
        return x, self.soft(x)


def load_model(config):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = myModel(model)
    return model, tokenizer


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2


class MyDataset(Dataset):
    def __init__(self, X_data, labels, tokenizer):
        self.embedding = tokenizer(X_data, return_tensors='pt', 
                              padding="max_length", max_length=180, truncation=True)
        self.embedding['input_ids'] = self.embedding['input_ids'].squeeze()
        self.embedding['attention_mask'] = self.embedding['attention_mask'].squeeze()
        self.labels = np.unique(labels, return_inverse=True)[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_onehot = torch.tensor(self.labels[idx])
        return self.embedding['input_ids'][idx], self.embedding['attention_mask'][idx], label_onehot


def dataset_loader(data, tokenizer, config):
    dataset = {'train': MyDataset(data['train']['x'], data['train']['y'], tokenizer),
              'val': MyDataset(data['val']['x'], data['val']['y'], tokenizer),
              'test': MyDataset(data['test']['x'], data['test']['y'], tokenizer)}

    dataloaders = {x: DataLoader(dataset[x], batch_size=config['batch_size'], 
                                 num_workers=1) for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val', 'test']}
    return dataloaders, dataset, dataset_sizes


def train_model(model, criterion, optimizer, dataloaders, 
                dataset_sizes, device, scheduler, config, 
                earlyStopper, num_epochs=25):
    since = time.time()
    history = {'train_loss': [], 'val_loss': [], 
                    'train_acc': [], 'val_acc': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    step = 0

    for epoch in range(num_epochs):
        deflog(f"========\nEpoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for input_ids, attention_mask, labels in tqdm(dataloaders[phase]):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, probs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(probs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        step += 1

                running_loss_tmp = loss.item() * input_ids.size(0)
                running_loss += running_loss_tmp
                running_corrects_tmp = torch.sum(preds == labels.data)
                running_corrects += running_corrects_tmp

                #######################################
                ###           wandb logging         ###
                #######################################
                tmp = {f'{phase}_loss_s': running_loss_tmp / input_ids.size(0), 
                        f'{phase}_acc_s': running_corrects_tmp.double() / input_ids.size(0),
                        'step_dl': step,
                        'epoch': epoch}
                wandb.log(tmp)
                #######################################
                #######################################

                if phase == 'train':
                    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(float(epoch_acc.cpu().numpy()))

            deflog(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            #######################################
            ###           wandb logging         ###
            #######################################
            tmp = {f'{phase}_loss': epoch_loss, 
                    f'{phase}_acc': epoch_acc,
                    'step_dl': step,
                    'epoch': epoch}
            wandb.log(tmp)
            #######################################
            #######################################

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        if earlyStopper.early_stop(history['val_loss'][-1]):
            deflog("Early Stopping")
            break

    time_elapsed = time.time() - since
    deflog(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    deflog(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def save_results(model, history, run_id, base_path="output"):
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    history_file_path = f"{base_path}/bert-history{run_id}.pkl"
    with open(history_file_path, 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    deflog(f'[Log] history has been saved in \"{history_file_path}\"')

    model_file_path = f"{base_path}/bert-model{run_id}.pt"
    torch.save(model.state_dict(), model_file_path)
    deflog(f'[Log] model has been saved in \"{model_file_path}\"')


def testing(model, dataloaders, device):
    y_pred = []
    y_true = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for input_ids, attention_masks, labels in tqdm(dataloaders):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            outputs = model(input_ids, attention_mask=attention_masks)[0]
            probs = softmax(outputs)
            _, preds = torch.max(probs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return y_pred, y_true


def run_model(config):
    tags = ["BERT"]
    run = wandb.init(entity='sentimovie', project=f"BERT-testing-2", 
                     name=f"BERT-{config['run_id']}", config=config, tags=tags)

    data = load_data(config)

    # model
    model, tokenizer = load_model(config)
    deflog("Model loaded")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    deflog(f"Model device is: {device}")

    # dataset and dataloader
    deflog("Loading dataloader and dataset")
    dataloaders, _, dataset_sizes = dataset_loader(data, tokenizer, config)
    deflog(f"Load data: [train:{dataset_sizes['train']}] [val:{dataset_sizes['val']}] [test:{dataset_sizes['test']}]")

    earlyStopper = EarlyStopper(patience=5, min_delta=1e-2)
    deflog("EarlyStopper defined")

    class_weights = compute_class_weight('balanced', 
                                        classes=np.unique(data['train']['y']), 
                                        y=np.array(data['train']['y']))
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    deflog("Define loss class_weights")

    optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                            weight_decay=config['weight_decay'])
    deflog("Optimizer (Adam) defined")

    # scheduler = lr_scheduler.LinearLR(optimizer, 
    #                      start_factor = 1e-5,
    #                      total_iters = 50)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                              T_max = 1500, # Maximum number of iterations.
                              eta_min = 1e-7)
    deflog("Scheduler (cosin_lr_scheduler) defined")

    model_best, history = train_model(model, criterion, optimizer, dataloaders, 
                                dataset_sizes, device, scheduler, config,
                                earlyStopper, config['epoch_num'])
    
    save_results(model_best, history, config['run_id'], base_path=config['out_dir'])

    y_pred, y_true = testing(model, dataloaders['test'], device)
    deflog("testing finished")
    f1 = f1_score(y_true, y_pred, average='macro')
    deflog("F1 done")
    acc = accuracy_score(y_true, y_pred)
    deflog("acc done")
    wandb.sklearn.plot_confusion_matrix(y_true, 
                                        y_pred, 
                                        np.unique(data['train']['y']))
    deflog("Metric calculated")
    tmp = {
        "test_acc": acc, 
        "test_f1": f1,
        "test_size": dataset_sizes['test']
    }
    wandb.log(tmp)
    deflog("Run finished")
    wandb.finish()
    


def main():
    os.chdir(args.current_dir)
    config = {
        "lr": 1e-3,
        "weight_decay": 1e-7,
        "train_p": 0.01,
        "val_p": 1,
        "test_p": 1,
        "epoch_num": 30,
        "batch_size": 5200,
        "class_num": 2,
        "run_id": 4,
        "out_dir": args.out_dir
    }
    run_model(config)


if __name__ == "__main__":
    main()
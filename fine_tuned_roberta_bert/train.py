from sklearn.utils import class_weight
import torch
import pandas as pd
from torch.types import Device
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, RobertaForSequenceClassification, RobertaTokenizer, BertModel, RobertaModel
from par import par
import random
import numpy as np
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, classification_report
from tqdm import tqdm, trange
from sklearn.utils.class_weight import compute_class_weight
from transformers.utils.dummy_pt_objects import LogitsProcessor
from utils import *
from models import *
import warnings
from tqdm import tqdm




def train_multi_class(train_df, val_df, num_additional_features=0):

  model = {
    'bert' : [BertModel, BertTokenizer, 'bert-base-cased'],
    'roberta' : [RobertaModel, RobertaTokenizer, 'roberta-base'],
    'roberta-large' : [RobertaModel, RobertaTokenizer, 'roberta-large']
  }
  

  model_class = model[par.model][0]
  tokenizer_class = model[par.model][1]
  pretrained_weights = model[par.model][2]

  labels_list = get_labels()
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.cuda.empty_cache()
  n_gpu = torch.cuda.device_count()

  random.seed(par.seed)
  np.random.seed(par.seed)
  torch.manual_seed(par.seed)

  
  if n_gpu > 0:
      torch.cuda.manual_seed_all(par.seed)
      print("Identified {} GPUs".format(n_gpu))

  if par.model != 'roberta-large':
    model = ModelForMulticlassClassification(model_class, pretrained_weights, num_additional_features=num_additional_features)
  else:
    model = ModelForMulticlassClassification(model_class, pretrained_weights, train_language_model=False, num_additional_features=num_additional_features)

  tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=par.lowerCase)

  
  print('Creating new model...')
      
  model.to(device)

  if par.weightedTraining == True:
    print('Computing the class weights...')

    train_df_propaganda = train_df[train_df[labels_list].sum(axis=1) != 0]

    loss_sum_weights = get_loss_sum_weights(train_df).to(device)
    bce_weights = get_bce_weights(train_df).to(device)

    criterions = [torch.nn.BCEWithLogitsLoss(pos_weight=bce_weights[i]) for i in range (len(labels_list))]
  else:
    criterions = torch.nn.BCEWithLogitsLoss()

  train_loader, extended_features = get_loader(train_df, tokenizer)
  val_loader, _ = get_loader(val_df, tokenizer, val=True)

  optimizer = AdamW(model.parameters(), lr=par.learningRate, eps=1e-8)

  num_warmup_steps=0

  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=len(train_loader)*5)

  global_step = 0
  nb_tr_steps = 0
  loss = 0
  tr_loss = 0
  max_grad_norm = 1.0
  best = 0
  train_losses = []
  valid_losses = []
  f1_scores = []
  best_net = None
  best_score = float('-inf')
  best_epoch = 0

  for epoch in trange(par.nEpochs, desc="Epoch"):
      if(par.train):
          model.train()

          for iteration, batch in enumerate(train_loader):
            inputs = {
              'input_ids' : batch[0].to(device),
              'attention_mask' : batch[1].to(device)
            }

            if len(extended_features) != 0:
              additional_features = []
              for i, extended_feature in enumerate(extended_features):
                additional_features.append(batch[i+2].to(device))
            else:
              additional_features = None

            inputs['additional_features'] = additional_features

            labels = batch[len(extended_features)+2].to(device)

            
            outputs = model(**inputs)

            if par.weightedTraining == True:
              loss = sum([loss_sum_weights[i] * criterions[i](torch.flatten(outputs[i]), labels[:, i].float()) for i in range(len(labels_list))])
            else:
              loss = sum([criterions(torch.flatten(outputs[i]), labels[:, i].float()) for i in range(len(labels_list))])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()
            nb_tr_steps += 1

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if iteration%20 == 0 and iteration != 0:
              print(f"Current batch: {iteration}/{len(train_loader)}")
              print(f"Loss: {tr_loss/nb_tr_steps}")

          train_losses.append(tr_loss/nb_tr_steps)
          model.eval()

          eval_loss, eval_accuracy = 0, 0
          predictions = [[] for _ in range(len(labels_list))]
          true_labels = [[] for _ in range(len(labels_list))]
          eval_accuracies = [0 for _ in range(len(labels_list))]
          val_loss = 0
          nb_val_steps = 0
          loss_list = []

          with torch.no_grad():
              for count, batch in enumerate(tqdm(val_loader, desc="Iteration")):
                
                inputs = {
                  'input_ids' : batch[0].to(device),
                  'attention_mask' : batch[1].to(device),
                  'training' : False
                }

                if len(extended_features) != 0:
                  additional_features = []
                  for i, extended_feature in enumerate(extended_features):
                    additional_features.append(batch[i+2].to(device))
                else:
                  additional_features = None

                inputs['additional_features'] = additional_features

                labels = batch[len(extended_features)+2].to(device)

                outputs = model(**inputs)

                if par.weightedTraining == True:
                  loss = sum([loss_sum_weights[i] * criterions[i](torch.flatten(outputs[i]), labels[:, i].float()) for i in range(len(labels_list))])
                else:
                  loss = sum([criterions(torch.flatten(outputs[i]), labels[:, i].float()) for i in range(len(labels_list))])

                loss_list.append(loss.item())
                labels = labels.cpu()

                for j, label in enumerate(labels_list):
                  prediction_single_column = torch.flatten(outputs[j]).cpu().round()
                  predictions[j].extend(prediction_single_column)
                  true_labels[j].extend(labels[:, j])
                  eval_accuracies[j] += (prediction_single_column == labels[:, j]).sum()

          f1_macros = []
          precisions = []
          recalls = []

          for j in range(len(labels_list)):
            f1_macros.append(f1_score(true_labels[j], predictions[j], average='binary'))
            precisions.append(precision_score(true_labels[j], predictions[j], average='binary'))
            recalls.append(recall_score(true_labels[j], predictions[j], average='binary'))
           
          print(f"f1: {f1_macros}")
          print(f"precisions: {precisions}")
          print(f"recalls: {recalls}")

          mean_loss_val = np.mean(loss_list)

          if best_score < np.average(f1_macros):
            best_score = np.average(f1_macros)
            best_net = model
            best_epoch = epoch

          with open("{}/{}/{}/Epoch{}.txt".format(par.savingPath, par.classType, par.expID, epoch+1), "a+") as f:
            
            print("Avg Precision: {}, Avg Recall: {}, Avg F1 score (macro): {}".format( 
                                                                                       np.average(precisions), 
                                                                                       np.average(recalls), 
                                                                                       np.average(f1_macros)), file=f)
            for j, label in enumerate(labels_list):
              print(f"{label} :\nPrecision: {precisions[j]}\nRecall: {recalls[j]}\nF1 Score: {f1_macros[j]}\n", file=f)

            f.close()
          
  print("Got best model at epoch {} with score {}".format(best_epoch, best_score))
  
  if par.saveModel == True:
    torch.save(best_net.state_dict(), "{}/{}/{}/weights.pkl".format(par.savingPath, par.classType, par.expID))

  return model



########################### SINGLE CLASS ################################



def train_single_class(train_df, val_df, num_additional_features=0):

  model = {
    'bert' : [BertForSequenceClassification, BertTokenizer, 'bert-base-cased'],
    'roberta' : [RobertaForSequenceClassification, RobertaTokenizer, 'roberta-base']
  }
  
  model_class = model[par.model][0]
  tokenizer_class = model[par.model][1]
  pretrained_weights = model[par.model][2]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.cuda.empty_cache()
  n_gpu = torch.cuda.device_count()

 

  random.seed(par.seed)
  np.random.seed(par.seed)
  torch.manual_seed(par.seed)

  
  if n_gpu > 0:
      torch.cuda.manual_seed_all(par.seed)
      print("Identified {} GPUs".format(n_gpu))


  labels = train_df['label'].unique()
  labels.sort()
  class_weights = torch.from_numpy(compute_class_weight(class_weight='balanced', classes=labels, y=train_df['label'])).float().to(device)

  model = model_class.from_pretrained(pretrained_weights, num_labels=par.nLabels)

  tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=par.lowerCase)

  
  print('Creating new model')
      
  model.to(device)

  train_loader, _ = get_loader(train_df, tokenizer)
  val_loader, _ = get_loader(val_df, tokenizer)

  optimizer = AdamW(model.parameters(), lr=par.learningRate, eps=1e-8)

  num_warmup_steps=0

  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=len(train_loader)*5)

  global_step = 0
  nb_tr_steps = 0
  tr_loss = 0
  max_grad_norm = 1.0
  best = 0
  train_losses = []
  valid_losses = []
  f1_scores = []

  criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

  for epoch in trange(par.nEpochs, desc="Epoch"):
      
      if(par.train):
          model.train()

          for batch in tqdm(train_loader):
              inputs = {
                  'input_ids' : batch[0].to(device),
                  'attention_mask' : batch[1].to(device),
                  'labels' : batch[2].to(device)
              }

              outputs = model(**inputs)

              logits = outputs['logits']
              loss = criterion(logits, inputs['labels'])

              if n_gpu > 1:
                  loss = loss.mean()

              loss.backward()
              torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

              tr_loss += loss.item()
              nb_tr_steps += 1

              optimizer.step()
              scheduler.step()
              optimizer.zero_grad()
              global_step += 1

          logging.info(f'EPOCH {epoch+1} Training Loss {(tr_loss / nb_tr_steps)}')
          train_losses.append(tr_loss/nb_tr_steps)

          model.eval()

          eval_loss, eval_accuracy = 0, 0
          predictions, true_labels = [], []

          with torch.no_grad():
              for count, batch in enumerate(tqdm(val_loader, desc="Iteration")):
                  inputs = {
                      'input_ids' : batch[0].to(device),
                      'attention_mask' : batch[1].to(device),
                      'labels' : batch[2].to(device)
                  }

                  outputs = model(**inputs)
                  logits = outputs.logits
                  logits = logits.detach().cpu().numpy()
                  labels = inputs['labels'].to('cpu').numpy()

                  predictions.extend(np.argmax(logits, axis=1))
                  true_labels.extend(labels)

                  eval_loss += outputs.loss.item()
                  eval_accuracy += (outputs.logits.argmax(axis=1) == inputs['labels']).sum().item()

          f1_macro = f1_score(true_labels, predictions, average='macro')
          precision = precision_score(true_labels, predictions, average='macro')
          recall = recall_score(true_labels, predictions, average='macro')
          eval_accuracy /= len(val_loader)
          eval_loss /= len(val_loader)

          print(f"Validation loss: {eval_loss}")
          print("Accuracy: {}, Precision: {}, Recall: {}, F1 score (macro): {}".format(eval_accuracy, precision, recall, f1_macro))
          print(f"Classification Report for Epoch {epoch+1}")
          print(classification_report(true_labels, predictions))
          plot_confusion_matrix(true_labels, predictions)

          f1_scores.append(f1_macro)
          valid_losses.append(eval_loss)
          if f1_macro > best:
              best = f1_macro

  print("Best performance: {}".format(best))

  draw_curves(train_losses, valid_losses, f1_scores)

  if par.saveModel == True:
    torch.save(model.state_dict(), "{}/{}/{}/{}.pkl".format(par.savingPath, par.classType, par.expID, par.expID))

  return model


######################### BINARY TASK ########################

def train_binary_class(train_df, val_df, num_additional_features=0):

  model = {
    'bert' : [BertForSequenceClassification, BertTokenizer, 'bert-base-uncased'],
    'roberta' : [RobertaForSequenceClassification, RobertaTokenizer, 'roberta-base']
  }
  
  model_class = model[par.model][0]
  tokenizer_class = model[par.model][1]
  pretrained_weights = model[par.model][2]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.cuda.empty_cache()
  n_gpu = torch.cuda.device_count()

  random.seed(par.seed)
  np.random.seed(par.seed)
  torch.manual_seed(par.seed)

  
  if n_gpu > 0:
      torch.cuda.manual_seed_all(par.seed)
      print("Identified {} GPUs".format(n_gpu))

  model = model_class.from_pretrained(pretrained_weights, num_labels=2)

  tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=par.lowerCase)

  print('Creating new model')
      
  model.to(device)

  train_loader, _ = get_loader(train_df, tokenizer)
  val_loader, _ = get_loader(val_df, tokenizer)

  optimizer = AdamW(model.parameters(), lr=par.learningRate, eps=1e-8)

  num_warmup_steps=0

  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=len(train_loader)*5)

  global_step = 0
  nb_tr_steps = 0
  tr_loss = 0
  max_grad_norm = 1.0
  best = 0
  train_losses = []
  valid_losses = []
  f1_scores = []
  best_model = None

  criterion = torch.nn.CrossEntropyLoss()

  softmax = nn.Softmax(dim=1)

  for epoch in trange(par.nEpochs, desc="Epoch"):
    model.train()

    for iteration, batch in enumerate(train_loader):
      inputs = {
          'input_ids' : batch[0].to(device),
          'attention_mask' : batch[1].to(device),
          'labels' : batch[2].to(device)
          }
      
      outputs = model(**inputs)

      logits = outputs['logits']
      loss = criterion(logits, inputs['labels'])

      if n_gpu > 1:
          loss = loss.mean()

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

      tr_loss += loss.item()
      nb_tr_steps += 1

      if iteration%20 == 0 and iteration != 0:
        print("EPOCH {}, ITERATION {}, with loss: {}".format(epoch, iteration, tr_loss))
        tr_loss = 0


      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()
      global_step += 1

    train_losses.append(tr_loss/nb_tr_steps)

    model.eval()

    eval_loss, eval_accuracy = 0, 0
    predictions, true_labels = [], []
  
    with torch.no_grad():

      for count, batch in enumerate(tqdm(val_loader, desc="Iteration")):
        inputs = {
            'input_ids' : batch[0].to(device),
            'attention_mask' : batch[1].to(device),
            'labels' : batch[2].to(device)
        }

        outputs = model(**inputs)
        logits = outputs.logits
        logits = logits.detach().cpu()
        logits = softmax(logits).numpy()
        
        predicted_labels = [1 if output[1] > 0.5 else 0 for output in logits]
        labels = inputs['labels'].to('cpu').numpy()
        # predictions.extend(np.argmax(logits, axis=1))
        predictions.extend(predicted_labels)
        true_labels.extend(labels)

        eval_loss += outputs.loss.item()

      f1_macro = f1_score(true_labels, predictions, average='binary')
      precision = precision_score(true_labels, predictions, average='binary')
      recall = recall_score(true_labels, predictions, average='binary')
      eval_accuracy /= len(val_loader)
      eval_loss /= len(val_loader)

    with open("{}/{}/{}/Epoch{}.txt".format(par.savingPath, par.classType, par.expID, epoch+1), "a+") as f:

      print(f"Validation loss: {eval_loss}", file=f)
      print("Precision: {}, Recall: {}, F1 score (macro): {}".format( precision, recall, f1_macro), file=f)
      print(f"Classification Report for Epoch {epoch+1}", file=f)
      print(classification_report(true_labels, predictions))
    
    plot_confusion_matrix(true_labels, predictions, epoch)

    f1_scores.append(f1_macro)
    valid_losses.append(eval_loss)
    if f1_macro > best:
        best = f1_macro
        best_model = model

  print("Best performance: {}".format(best))

  # draw_curves(train_losses, valid_losses, f1_scores)

  if par.saveModel == True:
    best_model.save_pretrained("{}/{}/{}/weights.pkl".format(par.savingPath, par.classType, par.expID))

  return model




def main():
  
  warnings.filterwarnings('ignore')
  TRAIN_PATH=par.trainPath
  VAL_PATH=par.valPath

  num_additional_features = 0


  df_train = pd.read_csv(TRAIN_PATH, sep='\t')

  labels_list = get_labels()

  df_val = pd.read_csv(VAL_PATH, sep='\t')

  features = df_train.columns.tolist()

  if 'title' in features:
    num_additional_features += 1
  if 'length' in features:
    num_additional_features += 1
  if 'position' in features:
    num_additional_features += 1
  if 'sentiment' in features:
    num_additional_features += 1
  if 'joy' in features:
    num_additional_features += 7

  if not os.path.exists("{}/{}/{}".format(par.savingPath, par.classType, par.expID)):
    try:
        os.mkdir("{}/{}/{}".format(par.savingPath, par.classType, par.expID))
    except FileNotFoundError:
        os.mkdir("{}/{}".format(par.savingPath, par.classType))
        os.mkdir("{}/{}/{}".format(par.savingPath, par.classType, par.expID))

  
  if par.task == "multi":
    # df_train = df_train[df_train[labels_list].sum(axis=1) != 0]
    # df_val = df_val[df_val[labels_list].sum(axis=1) != 0]
    model = train_multi_class(df_train, df_val, num_additional_features=num_additional_features)
  elif par.task == "single":
    model = train_single_class(df_train, df_val, num_additional_features=num_additional_features)
  elif par.task == 'binary':
    model = train_binary_class(df_train, df_val, num_additional_features=num_additional_features)


  

if __name__ == '__main__':
    main()


            

        
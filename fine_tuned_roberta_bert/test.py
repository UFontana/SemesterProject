from sklearn.utils import class_weight
import torch
from torch.nn import Softmax
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
from utils import *
from models import *
import warnings
from tqdm import tqdm

labels_list = ['Whataboutism',
    'Causal_Oversimplification',
    'Exaggeration,Minimisation', 
    'Doubt', 
    'Loaded_Language',
    'Name_Calling,Labeling', 
    'Flag-Waving', 
    'Reductio_ad_hitlerum',
    'Slogans', 
    'Appeal_to_fear-prejudice', 
    'Repetition',
    'Thought-terminating_Cliches', 
    'Black-and-White_Fallacy',
    'Bandwagon', 
    'Appeal_to_Authority', 
    'Red_Herring',
    'Obfuscation,Intentional_Vagueness,Confusion', 
    'Straw_Men'
    ]

################# TESTING OF THE MODEL (BINARY+MULTILABEL) ####################
      
def test_model(dev_path, path_binaryl_model=None):

  binary_models = {
    'bert' : [BertForSequenceClassification, BertTokenizer, 'bert-base-uncased'],
    'roberta' : [RobertaForSequenceClassification, RobertaTokenizer, 'roberta-base']
  }

  multil_models = {
    'bert' : [BertModel, BertTokenizer, 'bert-base-uncased'],
    'roberta' : [RobertaModel, RobertaTokenizer, 'roberta-base'],
    'roberta-large' : [RobertaModel, RobertaTokenizer, 'roberta-large']
  }


  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.cuda.empty_cache()
  n_gpu = torch.cuda.device_count() 

  random.seed(par.seed)
  np.random.seed(par.seed)
  torch.manual_seed(par.seed)


  binary_model_class = binary_models[par.model][0]
  tokenizer_class = binary_models[par.model][1]
  pretrained_weights = binary_models[par.model][2]

  multil_model_class = multil_models[par.model][0]

  if par.binaryModel != "":
    print('LOADING THE BINARY MODEL...')
    binary_model = binary_model_class.from_pretrained(par.binaryModel).to(device)
    binary_model.eval()

  print('LOADING THE MULTICLASS MODEL...')
  multil_model = ModelForMulticlassClassification(multil_model_class, pretrained_weights)
  multil_model.load_state_dict(torch.load(par.multilModel))
  multil_model = multil_model.to(device)



  if n_gpu > 0:
      torch.cuda.manual_seed_all(par.seed)
      print("Identified {} GPUs".format(n_gpu))

  tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=par.lowerCase)

  df_val = pd.read_csv(dev_path, sep='\t')

  val_loader, _ = get_loader(df_val, tokenizer, val=True)

  multil_model.eval()

  predictions = [[] for _ in range(len(labels_list))]
  true_labels = [[] for _ in range(len(labels_list))]
  eval_accuracies = [0 for _ in range(len(labels_list))]

  zero_label = np.zeros(len(labels_list))

  with torch.no_grad():

    for batch in tqdm(val_loader):

      inputs = {
        'input_ids' : batch[0].to(device),
        'attention_mask' : batch[1].to(device)
      }

      labels = batch[2]
      softmax = Softmax(dim=1)

      outputs_multil = multil_model(**inputs, training=False)

      if par.binaryModel != "":
        outputs_binary = softmax(binary_model(**inputs).logits).detach().cpu().numpy()

        predicted_label = [1 if output[1] > 0.5 else 0 for output in outputs_binary]

        for i, pred in enumerate(predicted_label):
          if pred == 0:
            for j, o in enumerate(outputs_multil):
              o[i] = 0.0
              outputs_multil[j] = o

      for j, label in enumerate(labels_list):
        prediction_single_column = torch.flatten(outputs_multil[j]).cpu().round()
        predictions[j].extend(prediction_single_column)
        true_labels[j].extend(labels[:, j])
        eval_accuracies[j] += (prediction_single_column == labels[:, j]).sum()


  f1_scores = []
  precisions = []
  recalls = []
  for j, label in enumerate(labels_list):
    f1_score_singl = f1_score(true_labels[j], predictions[j], average='binary')
    precision_singl = precision_score(true_labels[j], predictions[j], average='binary')
    recall_singl = recall_score(true_labels[j], predictions[j], average='binary')

    f1_scores.append(f1_score_singl)
    precisions.append(precision_singl)
    recalls.append(recall_singl)

    print("PRECISION FOR LABEL {} IS {}".format(label, precision_singl))
    print("RECALL FOR LABEL {} IS {}".format(label, recall_singl))
    print("F1 SCORE FOR LABEL {} IS {}".format(label, f1_score_singl))
    print()

  print("Avg Precision: {}, Avg Recall: {}, Avg F1 score: {}".format(np.average(precisions), 
                                                                             np.average(recalls), 
                                                                             np.average(f1_scores)))


def main():
  dev_path = par.valPath
  test_model(dev_path)

if __name__ == '__main__':
  main()

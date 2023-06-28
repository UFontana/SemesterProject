import torch.nn as nn
import torch
from transformers import BertModel
from utils import *
from par import par

class ModelForMulticlassClassification(nn.Module):

  def __init__(self, language_model, pretrained_weights, train_language_model=True, num_additional_features=0):
    super(ModelForMulticlassClassification, self).__init__()
    self.language_model = language_model.from_pretrained(pretrained_weights)

    self.labels = get_labels()

    if not train_language_model:
      print("FREEZING MODEL PARAMTERS...")
      for _, param in self.language_model.named_parameters():
        param.requires_grad = False

    dim_in = self.language_model.config.hidden_size+num_additional_features

    self.classifier = nn.ModuleList([nn.Linear(dim_in, 1) for _ in range(len(self.labels))])
    self.sigmoid = nn.Sigmoid()

  
  def forward(self, input_ids, attention_mask, additional_features=None, training=True):

    outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)
    pooled_output = outputs.pooler_output
        
    if additional_features != None:

      for feature in additional_features:
        feature = feature.unsqueeze(1)
        pooled_output = torch.cat((pooled_output, feature), dim=1)
      
      
    logits = [self.classifier[i](pooled_output) for i in range(len(self.labels))]

    if training == False:
      logits = [self.sigmoid(l) for l in logits]

    return logits


    




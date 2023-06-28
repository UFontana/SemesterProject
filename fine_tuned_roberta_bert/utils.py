import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from par import par
import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler
from par import par

propaganda_labels = ['Whataboutism',
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

persuasion_labels = [
  'Justification',
  'Simplification',
  'Distraction',
  'Call',
  'Manipulative_Wording',
  'Attack_on_Reputation'
]


def plot_confusion_matrix(y_true, y_pred, epoch):

  palette = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)
  matrix = confusion_matrix(y_true, y_pred, normalize='all')

  fig, ax = plt.subplots(figsize=(10,8))
  ConfusionMatrixDisplay(matrix).plot(ax=ax, cmap=palette)

  plt.xlabel('Predicted labels')
  plt.ylabel('True labels')
  plt.title('Confusion Matrix')
  fig.savefig("{}/{}/{}/confusion_matrix_epoch_{}.png".format(par.savingPath, par.classType, par.expID, epoch))
  plt.close(fig)

  return


def draw_curves(train_losses, valid_losses, f1_scores):

  x = list(range(len(valid_losses)))

  fig, _ = plt.subplots(figsize=(10, 8))
  
  plt.plot(x, train_losses, label = "Train loss") 

  plt.plot(x, valid_losses, label = "Validation losses") 

  plt.plot(x, f1_scores, label = "F1 scores") 
  
  plt.xlabel('Epochs') 
  plt.ylabel('Metric') 
  plt.title('Training Curves') 
  plt.legend()

  fig.savefig("{}/learning_curve.png".format(par.savingPath))
  plt.close(fig)

  return

def get_loader(df, tokenizer, val=False):

  labels_list = get_labels()

  features = df.columns.tolist()
  additional_features = []

  if par.weightedSampler == True and val == False:
    print("USING WEIGHTED SAMPLER...")
    num_samples = len(df)
    weights =  num_samples/ df[labels_list].sum()
    shuffle = False

    weight_samples = np.zeros(len(df))

    for index, row in df.iterrows():

      label_cols = row[labels_list]
      label_cols_1 = label_cols[label_cols == 1]

      if not label_cols.empty:
          labels_names = label_cols.index.tolist()
          for label_name in labels_names:
            weight_samples[index] += weights[label_name]

    sampler = WeightedRandomSampler(weight_samples, len(weight_samples), replacement=True)
    shuffle = False

  else:
    sampler = None
    shuffle = True if val == False else False

  label_features = labels_list if par.task == "multi" else 'label'

  encodings = tokenizer(list(df['text']), truncation=True, padding=True, max_length=400)

  labels = torch.tensor(df[label_features].values)
    
  dataset_features = [torch.tensor(encodings['input_ids']), torch.tensor(encodings['attention_mask'])]

  if 'title_similarity_score' in features:
    title_similarity_scores = df['title_similarity_score']
    dataset_features.append(torch.tensor(title_similarity_scores.to_numpy(), dtype=torch.float32))
    additional_features.append('title_similarity_score')
  if 'position' in features:
    text_relative_position = df['position']
    dataset_features.append(torch.tensor(text_relative_position.to_numpy(), dtype=torch.float32))
    additional_features.append('position')
  if 'length' in features:
    length_text = df['position']
    dataset_features.append(torch.tensor(length_text.to_numpy(), dtype=torch.float32))
    additional_features.append('length')
  if 'sentiment' in features:
    sentiment_score = df['sentiment']
    dataset_features.append(torch.tensor(sentiment_score.to_numpy(), dtype=torch.float32))
    additional_features.append('sentiment')
  if 'joy' in features:
    joy_score = df['joy']
    neutral_score = df['neutral']
    surprise_score = df['surprise']
    disgust_score = df['disgust']
    sadness_score = df['sadness']
    fear_score = df['fear']
    anger_score = df['anger']
    additional_features.extend(['joy', 'neutral', 'surprise', 'disgust', 'sadness', 'fear', 'anger'])
    dataset_features.extend([torch.tensor(joy_score.to_numpy(), dtype=torch.float32), torch.tensor(neutral_score.to_numpy(), dtype=torch.float32), 
                            torch.tensor(surprise_score.to_numpy(), dtype=torch.float32), torch.tensor(disgust_score.to_numpy(), dtype=torch.float32),
                            torch.tensor(sadness_score.to_numpy(), dtype=torch.float32), torch.tensor(fear_score.to_numpy(), dtype=torch.float32), 
                            torch.tensor(anger_score.to_numpy(), dtype=torch.float32)])
    
  dataset_features.append(labels)

  dataset = torch.utils.data.TensorDataset(*dataset_features)

  loader = torch.utils.data.DataLoader(dataset, batch_size=par.batchSize, shuffle=shuffle, sampler=sampler)

  return loader, additional_features
 

def get_bce_weights(train_df):

  labels_list = get_labels()

  def get_col_name(row):
    b = (train_df.loc[row.name] == 1)
    c = list(b.index[b])
    return c
 
  y = train_df.apply(get_col_name, axis=1)
  binary_labels_all_classes = []

  for label in labels_list:
      binary_labels_one_class = []
      for row in y:
          if label in row:
              binary_labels_one_class.append("1")
          else:
              binary_labels_one_class.append("0")
      binary_labels_all_classes.append(binary_labels_one_class)

  binary_class_weights_list = torch.from_numpy(np.array(
      [binary_labels_all_classes[i].count("0")/binary_labels_all_classes[i].count("1") for i in
        range(len(binary_labels_all_classes))]))

  return binary_class_weights_list


def get_loss_sum_weights(train_df):

  labels_list = get_labels()

  def get_col_name(row):
    b = (train_df.loc[row.name] == 1)
    c = list(b.index[b])
    return c

  y = train_df.apply(get_col_name, axis=1)
  y_list = []

  for row in y:
      for label in row:
          if label in labels_list:
              y_list.append(label)

  class_weights = torch.from_numpy(
      compute_class_weight(class_weight='balanced', classes=labels_list, y=y_list)).float()

  return class_weights

def get_labels():
  if par.labelsType == 'persuasion':
    return persuasion_labels
  
  return propaganda_labels



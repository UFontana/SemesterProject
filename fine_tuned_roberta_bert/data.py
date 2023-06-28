import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import warnings
import sys
from utils import get_labels
from transformers import pipeline

from par import par


label_dict = {
    'None' : 0,
    'Whataboutism' : 1,
    'Causal_Oversimplification' : 2,
    'Exaggeration,Minimisation' : 3, 
    'Doubt' : 4, 
    'Loaded_Language' : 5,
    'Name_Calling,Labeling' : 6, 
    'Flag-Waving' : 7, 
    'Reductio_ad_hitlerum' : 8,
    'Slogans' : 9, 
    'Appeal_to_fear-prejudice' : 10, 
    'Repetition' : 11,
    'Thought-terminating_Cliches' : 12, 
    'Black-and-White_Fallacy' : 13,
    'Bandwagon' : 14, 
    'Appeal_to_Authority' : 15, 
    'Red_Herring' : 16,
    'Obfuscation,Intentional_Vagueness,Confusion' : 17, 
    'Straw_Men' : 18
}

id_dict = {
    0:'None',
    1:'Whataboutism',
    2:'Causal_Oversimplification',
    3:'Exaggeration,Minimisation', 
    4:'Doubt', 
    5:'Loaded_Language',
    6:'Name_Calling,Labeling', 
    7:'Flag-Waving', 
    8:'Reductio_ad_hitlerum',
    9:'Slogans', 
    10:'Appeal_to_fear-prejudice', 
    11:'Repetition',
    12:'Thought-terminating_Cliches', 
    13:'Black-and-White_Fallacy',
    14:'Bandwagon', 
    15:'Appeal_to_Authority', 
    16:'Red_Herring',
    17:'Obfuscation,Intentional_Vagueness,Confusion', 
    18:'Straw_Men'
}

propaganada_labels = ['Whataboutism',
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

persuasion_propaganda = {
  'Appeal_to_Authority' : 'Justification',
  'Appeal_to_fear-prejudice' : 'Justification',
  'Flag-Waving' : 'Justification',
  'Causal_Oversimplification' : 'Simplification',
  'Straw_Men' : 'Distraction',
  'Red_Herring' : 'Distraction',
  'Whataboutism' : 'Distraction',
  'Loaded_Language' : 'Manipulative_Wording',
  'Repetition' : 'Manipulative_Wording',
  'Exaggeration,Minimisation' : 'Manipulative_Wording',
  'Name_Calling,Labeling' : 'Attack_on_Reputation',
  'Doubt' : 'Attack_on_Reputation',
  'Slogans' : 'Call'
}


peruasion_labels = [
  'Justification',
  'Simplification',
  'Distraction',
  'Call',
  'Manipulative_Wording',
  'Attack_on_Reputation'
]

def extract_sentiment_score(sentence, pipeline):
  
  sentiment = pipeline(sentence)[0]
  
  # Sentiment score computed as positive score - negative score
  return sentiment[1]['score'] - sentiment[0]['score']

def extract_emotion(sentence, pipeline):
  
  result = {}
  emotion = pipeline(sentence)[0]

  for entry in emotion:
    result[entry['label']] = entry['score']

  return result


def read_articles(train_dir, columns):

  train_df = pd.DataFrame(columns=columns)

  sent_n = 0
  model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

  if par.emotion:
    emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

  if 'sentiment' in columns:
    sentiment_pipeline = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english", return_all_scores=True)

  print("READING DIRECTORY...")

  for filename in tqdm(os.listdir(train_dir)):
      if filename.endswith(".txt"):
          article_id = filename.split(".")[0][7:]
          with open(train_dir + filename, "r", encoding='utf-8') as f:
              content = f.read()
              sentences = text_to_sentences(content)
              sent_n = sent_n + len(sentences)
              
              output = []
              n = len(sentences)

              for i, sentence in enumerate(sentences):
                  entry = {}

                  start = content.find(sentence)
                  end = start + len(sentence) - 1

                  entry['doc_id'] = article_id
                  entry['start'] = int(start)
                  entry['end'] = int(end)
                  entry['text'] = sentence.strip()

                  if 'title' in columns:
                    if i == 0:
                        title = sentence
                        title_encoded = model.encode(sentence)
                        similarity_score = 1.0
                    else:
                      start = content.find(sentence)
                      end = start + len(sentence) - 1
                      sentence_encoded = model.encode(sentence)

                      
                      similarity_score = cosine_similarity(title_encoded.reshape(1, -1), sentence_encoded.reshape(1, -1))[0][0]

                    entry['title'] = title
                    entry['title_similarity_score'] = similarity_score

                  if 'position' in columns:
                    entry['position'] = i/n

                  if 'length' in columns:
                    entry['length'] = len(sentence)

                  if par.emotion:
                    entry.update(extract_emotion(sentence, emotion_pipeline))
                  
                  if 'sentiment' in columns:
                    entry['sentiment'] = extract_sentiment_score(sentence, sentiment_pipeline)
                  

                  train_df = train_df.append(entry, ignore_index = True)
                  start = end + 1

  return train_df

def label_name_to_id(text):
  labels = ""
  first = True
  for key in label_dict:
    if key in text:
      if not first:
        labels += ","
      labels += str(label_dict[key])
      first = False
  return labels

def id_to_label(text):
  labels = ""
  first = True
  arr = text.split(",")
  for key in arr:
    if not first:
      labels += ","
    labels += id_dict[int(key)]
    first = False
  return labels

def text_to_sentences(text):
  text_content = text
  text_string = text_content.replace("\n", ".")
  
  characters_to_remove = [";"]

  for item in characters_to_remove:
    text_string = text_string.replace(item,"")
    
  sentences_not_filtered = text_string.split(".")
  j = 0

  sentences = [sentence.lstrip() for sentence in sentences_not_filtered if len(sentence) >= 2]
  
  return sentences




#################### DATA MULTI LABELS #####################


def data_multi_label(train_dir, columns=['doc_id', 'start', 'end', 'text']):
  
  train_df = read_articles(train_dir, columns)

  labels = get_labels()
  
  labels_df = pd.DataFrame(0, index=train_df.index, columns=labels)
  train_df = pd.concat([train_df, labels_df], axis=1)

  for filename in os.listdir(train_dir):
    if filename.endswith(".tsv"):
        tmp_df = pd.read_csv(train_dir + filename, sep='\t', names = ['id', 'type', 'start', 'end'])
        id = filename.split(".")[0][7:]

        for _, row in tmp_df.iterrows():
          tech_id = label_dict[row["type"]]
          v = train_df.loc[(train_df['doc_id'] == id) & 
                           ((train_df['start'] <= row['start']) & #p starts after beginning of the sentence and before the ending
                           (train_df['end'] >= row['start']) |
                           (train_df['start'] <= row['end']) & #p starts after beginning of the sentence and before the ending
                           (train_df['end'] >= row['end']))] 

          

          if len(train_df.loc[v.index, :].values) == 0:
            continue

          if par.labelsType == 'propaganda':
            train_df.loc[v.index, row["type"]] = 1
          if par.labelsType == 'persuasion' : 
            if row["type"] in persuasion_propaganda.keys():
              train_df.loc[v.index, persuasion_propaganda[row["type"]]] = 1

  return train_df



################### DATA SINGLE LABEL #######################



def data_single_label(train_dir, columns=['doc_id', 'start', 'end', 'text']):

  train_df = read_articles(train_dir, columns)
  train_df['label'] = "None"

  for filename in os.listdir(train_dir):
    if filename.endswith(".tsv"):
      tmp_df = pd.read_csv(train_dir + filename, sep='\t', names = ['id', 'type', 'start', 'end'])
      article_id = filename.split(".")[0][7:]

      for _, row in tmp_df.iterrows():
        tech_id = label_dict[row["type"]]
        v = train_df.loc[(train_df['doc_id'] == article_id) & 
                          ((train_df['start'] <= row['start']) & #p starts after beginning of the sentence and before the ending
                          (train_df['end'] >= row['start']) |
                          (train_df['start'] <= row['end']) & #p starts after beginning of the sentence and before the ending
                          (train_df['end'] >= row['end']))] 

        if len(train_df.loc[v.index, 'label'].values) == 0:
          continue

        current_label = train_df.loc[v.index, 'label'].values[0]

        if current_label == "None":
          if par.labelsType == 'propaganda':
            train_df.loc[v.index, 'label'] = str(label_dict[row["type"]])
          else:
            if row["type"] in persuasion_propaganda.keys():
              train_df.loc[v.index, 'label'] = str(persuasion_propaganda[row["type"]])
  
  return train_df


########################## BINARY LABEL #############################


def data_binary_label(train_dir, columns=['doc_id', 'start', 'end', 'text']):

  train_df = read_articles(train_dir, columns)
  train_df['label'] = 0

  for filename in os.listdir(train_dir):
    if filename.endswith(".tsv"):
      tmp_df = pd.read_csv(train_dir + filename, sep='\t', names = ['id', 'type', 'start', 'end'])
      article_id = filename.split(".")[0][7:]

      for _, row in tmp_df.iterrows():
    
        v = train_df.loc[(train_df['doc_id'] == article_id) & 
                          ((train_df['start'] <= row['start']) & #p starts after beginning of the sentence and before the ending
                          (train_df['end'] >= row['start']) |
                          (train_df['start'] <= row['end']) & #p starts after beginning of the sentence and before the ending
                          (train_df['end'] >= row['end']))] 

        if len(train_df.loc[v.index, 'label'].values) == 0:
          continue
          
        current_label = train_df.loc[v.index, 'label'].values[0]

        if current_label == 0:
          if par.labelsType == 'propaganda':
            train_df.loc[v.index, 'label'] = 1
          else:
            if row["type"] in persuasion_propaganda.keys():
              train_df.loc[v.index, 'label'] = 1
  
  return train_df



def main():

  columns = ['doc_id', 'start', 'end', 'text']

  warnings.filterwarnings("ignore")

  if par.labelsType not in ['propaganda', 'persuasion']:
    print("Please, set the --labelsType argument to a value [propaganda | persuasion]")
    sys.exit(-1)

  if par.title:
    columns.append('title')

  if par.length:
    columns.append('length')

  if par.position:
    columns.append('position')

  if par.sentiment:
    columns.append('sentiment')

  if par.taskType == 'single':
    df = data_single_label(par.articlesDir, columns=columns)
  elif par.taskType == 'multi':
    df = data_multi_label(par.articlesDir, columns=columns)
  elif par.taskType == 'binary':
    df = data_binary_label(par.articlesDir, columns=columns)
  else:
    print("Please, set the --taskType option to a value [single | multi | binary]")
    sys.exit(-1)
    
    
  df.to_csv(par.fileOut, sep='\t', index=False)

if __name__ == '__main__':
  main()
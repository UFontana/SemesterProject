import argparse

parser = argparse.ArgumentParser(description = 'PyTorch BERT model Training')

"------------------------- General options -------------------------"

parser.add_argument('--expID', default='0', type=str, help="Experiment ID")
parser.add_argument('--seed', default=42, type=int, help="Seed value")
parser.add_argument('--task', default="multi", type=str, 
                    help="multi | single | binary | persuasion")
parser.add_argument('--classType', default="single_label", type=str, help='multi_label_weighted | single_label | multi_label_not_weighted')
parser.add_argument('--train', default=True, type=bool, help="Set false not to train the model")
parser.add_argument('--trainPath', 
                    default='/content/drive/MyDrive/Colab Notebooks/semester-project-propaganda/semester-project-propaganda/code/data/csv/train_data_sentences.csv', 
                    type=str, help='Path to specify for training data')
parser.add_argument('--valPath',
                    default='/content/drive/MyDrive/Colab Notebooks/semester-project-propaganda/semester-project-propaganda/code/data/csv/val_data_sentences.csv', 
                    type=str, help='Path to specify for validation data')
parser.add_argument('--testPath',
                    default='/content/drive/MyDrive/Colab Notebooks/semester-project-propaganda/semester-project-propaganda/code/data/csv/test_data_sentences.csv', 
                    type=str, help='Path to specify for test data')
parser.add_argument('--saveModel', default=False, type=bool, help='Set True if you want to save the experiment')
parser.add_argument('--savingPath', 
                    default='/content/drive/MyDrive/Colab Notebooks/semester-project-propaganda/semester-project-propaganda/code/models', 
                    type=str, help='Path to specify for saving the model')
parser.add_argument('--model', default='bert', type=str, help='bert | roberta')
parser.add_argument('--extendedFeatures', default=False, type=bool, help='Set to true to insert new features. They should be associated with the right dataframe!')

"------------------------- Test options --------------------------"

parser.add_argument('--binaryModel', default="", type=str, help='Path of the weights of the binary classification model')
parser.add_argument('--multilModel', default="", type=str, help="Path of the weights of the multilabels classification model")

"------------------------- Model options -------------------------"

parser.add_argument('--nLabels', default=19, type=int, help="Number of Labels to Consider")
parser.add_argument('--lowerCase', default=False, type=bool, help="Set true if using uncased model")

"------------------------- Train options -------------------------"

parser.add_argument('--nEpochs', default=5, type=int, help="Number of Epochs to train for")
parser.add_argument('--batchSize', default=16, type=int, help="Size of the training batch")
parser.add_argument('--learningRate', default=2e-5, type=float, help="Learning Rate")
parser.add_argument('--no-weightedTraining', dest='weightedTraining', action='store_false')
parser.add_argument('--weightedTraining', default=True, action='store_true', help="Set to --no-weightedTraining if you want not to weight the classes")
parser.add_argument('--weightedSampler', default=False, type=bool, help="Set to true for having a balanced sampler")

"------------------------ Data Creation Options --------------------"

parser.add_argument('--taskType', type=str, default='multi', help='multi | single | binary')
parser.add_argument('--labelsType', type=str, default='propaganda', help='propaganda | persuasion')
parser.add_argument('--articlesDir', type=str, 
                    default='/content/drive/MyDrive/Colab Notebooks/semester-project-propaganda/semester-project-propaganda/code/data/train',
                    help="Specify the articles directory to create a csv format")
parser.add_argument('--fileOut', type=str, 
                    default="/content/drive/MyDrive/Colab Notebooks/semester-project-propaganda/semester-project-propaganda/code/data/multilabels_train.csv",
                    help="Specify the name of the csv in out")
parser.add_argument('--title', type=bool, default=False, help="Set to true to add the title-related features")
parser.add_argument('--length', type=bool, default=False, help="Set to true to add the length of the sentence as feature")
parser.add_argument('--position', type=bool, default=False, help="Set to true to specify the position related features")
parser.add_argument('--sentiment', type=bool, default=False, help="Set to true to specify the sentiment score")
parser.add_argument('--emotion', type=bool, default=False, help="Set to true to specify the emotion features")


par = parser.parse_args()

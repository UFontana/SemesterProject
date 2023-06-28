# Propaganda classification using SentenceBERT and interpretable features

This folder contains 2 jupyter notebook files:

1. Features.ipynb performs several tasks:
	i. prepracessing of the data and collecting sentence labels from fragment annotations from articles
	ii. creating dataframes
	iii. creating interpretable features: 
		a. title and context embedding
		b. emotion vector containing probabilities for 59 emotions associated with propaganda 
		c. labels for our task of multilabel classification obtained from ChatGPT API

2. TraditionalModels.ipynb performs several tasks:
	i. preprocessing created dataframes - type conversion, creating tensors and dataloaders for training
	ii. training and testing data with four traditional machine learning algorithms:
		a. Fully connected neural network with 1 hidden layer
		b. Random Forrest
		c. Decision Trees
		d. SVM
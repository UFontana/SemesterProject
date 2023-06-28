# Propaganda classification using SentenceBERT and interpretable features

This folder contains 2 jupyter notebook files.


1. **Features.ipynb** performs several tasks:


	- preprocessing of the data and collecting sentence labels from fragment annotations from articles
	- creating dataframes
	- creating interpretable features: 
		a. title and context embedding
		b. emotion vector containing probabilities for 59 emotions associated with propaganda 
		c. labels for our task of multilabel classification obtained from ChatGPT API

3. **TraditionalModels.ipynb** performs other tasks:
	- preprocessing created dataframes - type conversion, creating tensors and dataloaders for training
	- training and testing data with four traditional machine learning algorithms:
	  	- Fully connected neural network with 1 hidden layer
		- Random Forrest
		- Decision Trees
		- SVM

# spam_classification 

## Introduction
This is a classifier for spam based on the naive Bayes. The average accuracy is larger than 97%, which performs quite well. The datasets contain of more 60 thousands Chinese emails. We train a spam_classifier based on this dataset using ***Naive Bayes*** algorithm.

## Requirements
python 3  
packages: numpy, tqdm(a package which can show the progress bar), codes     

## Running
The source file is 
Put the dataset and the python source code into the same directory   
There are four parameters that can be configured in the bash.    

* ***-p percent*** #input a number between 0 and 1 to configure the percent of the data used for train. The default value is 0, which means it will use 5-fold cross validation.If you input a number larger than 0, it will test the affect of the size of the training set 
* ***-r random seed*** #input an integer to configure the random seed.In the program, there is a shuffle operation. To recur the result, we can configure a fixed random seed. The default value is 2333.    
* ***-n dataset_name*** #configure the dataset name which have been put into the same directory with the source code. The default name is 'trec06c-utf8'
* ***-d add_feature*** #configure whether to add other features besides the cut word in the emails. Other features consists of email addresses.The default value is 1, which means the email addresses will be included.If you input 0, the added feature will be excluded.
  
Therefore, the program can be run like this:

`python3 Bayes.py` This means that all configurable parameters are used in default value, which means the model will use 5-fold cross validation to make a test and add the feature.  
`python3 Bayes.py -p 0.8` This means the program will train 80% of the dataset.
`python3 Bayes.py -r 233` This means the random seed will be changed into 2333

## Function Instruction
Basically, there are two classes in the program:   
`class Dataset`   
`class Trainer`   
The `Dataset` provides interfaces to load the email from the files and organize the data into the desired format    
The `Trainer` is used for training the model and make a test to obtain the accuracy
More details about the algorithm are written in the report, which can be found here:

[**report.pdf**](https://github.com/JamesHujy/spam_classfication/blob/master/report.pdf)


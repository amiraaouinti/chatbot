import nltk
from nltk.stem.lancaster import LancasterStemmer
import random
import numpy as np
import json
import tflearn
import tensorflow as tf
import pickle
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words =[]
    labels =[]
    docs_x =[]
    docs_y =[]


    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = tknzr.tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))  #set makes words unique then convert it to  a list
    labels = sorted(labels)
    # Now we only have strings but Neural Network only understand numbers
    # we're using one hot encoded to transform all the strings into numbers so ce can train our model
    # one hot encoded usually represents if a word exist or not
    # create bag of words

    training =[]
    output =[]

    out_empty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    # our model

    training = np.array(training)
    output = np.array(output)
    
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
        
tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")

net = tflearn.regression(net)
# DNN is a type of neural network
model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    
# transform the user's sentence into bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = tknzr.tokenize(s)
    s_words= [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i]=1
                
    return np.array(bag)

## web scraping



# the code that will ask the user for a sentence and answer him

def chat():
    print("Start talking with ilboursa Bot (type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower()=="quit":
            break
        
        results = model.predict([bag_of_words(inp, words)])
        results_index = np.argmax(results)
        tag = labels[results_index] 
       # if results[results_index] > 0.7
      #  else:
       #     print("J'arrive pas à vous comprendre, essayez de nouveau")
        for tg in data['intents']:
            if tg["tag"] == tag:
                responses = tg["responses"]
                
        print(random.choice(responses))
        
        
        
chat()
            
    
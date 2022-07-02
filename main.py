from operator import indexOf
import nltk 
# run the following only once
# nltk.download('punkt') 
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()

import numpy as np
import tflearn 
import tensorflow
import random
import json

import pickle


# data pre-processing
with open("intents.json") as file:
    data=json.load(file)
    # print(data["intents"])
try:
    x
    #delete data.pickle if intents.json has been changed.
    with open("data.pickle", "rb") as f:
        words,labels,training, output =pickle.load(f)
except:

    words=[]
    labels=[]
    #all the patterns
    docs_x=[]
    #tag for each pattern in patterns
    docs_y=[]  
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds=nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(pattern) #all the patterns  rename: all_pattern later
            docs_y.append(intent["tag"]) # same length as docs_x, matching each pattern with its tag
        if intent["tag"] not in labels:
            labels.append(intent["tag"]) #all the tags
    # print(words) 
    # print("docs_x:", docs_x)
    # print("docs_y:",docs_y)
    # print("labels:",labels)

    # stem all the words in words list
    # remove any duplicate elements
    words=[stemmer.stem(w.lower()) for w in words if w not in "?" ]
    words=sorted(list(set(words)))
    labels=sorted(labels) 
    # print(words)
    # print(labels)


    #bag of words  check out One hot encoding: frequency 
    training=[]
    output=[]
    out_empty=[0 for _ in range(len(labels))]

    for idx, pattern in enumerate(docs_x):
        # print(idx,doc)
        # 0 Hi
        # 1 How are you
        # 2 Is anyone there?
        # 3 Hello
        # 4 Good day
        # 5 Whats up
        # 6 cya
        # ...
        #each pattern get a new empty bag
        bag=[]
        tokenized_p=nltk.word_tokenize(pattern)
        stemmed_p=[stemmer.stem(w.lower()) for w in tokenized_p]
        # print("tokens",tokens)
        # print("stemmed",stemmed)
        # wrds=[stemmer.stem(w.lower()) for w in doc]
        # print("all words:",words)
        # print("wrds:",wrds)
        for w in words:
            # the word is in the pattern we are currently looping through 
            if w in stemmed_p:
                bag.append(1)
            else:
                bag.append(0)
        output_row=out_empty[:]
        output_row[labels.index(docs_y[idx])] =1
        training.append(bag)
        output.append(output_row)


    training=np.array(training)
    output=np.array(output)
    with open("data.pickle", "wb") as f:
        pickle.dump((words,labels,training, output),f)

# building the model with tflearn
tensorflow.compat.v1.reset_default_graph()
net=tflearn.input_data(shape=[None, len(training[0])])
net=tflearn.fully_connected(net,8)
# hidden layer with 8 neurons 
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,len(output[0]),activation="softmax")
net=tflearn.regression(net)
model = tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:
    model=tflearn.DNN(net)
    model.fit(training,output, n_epoch=1000, batch_size=8,show_metric=True)
    model.save("model.tflearn")




#accept user input to classify user input 
#first turn the input to bag of words

def bag_of_words(s,words):
    bag=[0 for _ in range(len(words))]
    s_words=nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower()) for word in s_words]
    for s_word in s_words:
        for i,w in enumerate(words):
            if w==s_word:
                bag[i]=1
    return np.array(bag)

def chat():
    print("Start talking with the bot (type quit to stop)")
    while True:
        inp=input("You:  ")
        if inp.lower()== "quit":
            break
        #this will return a list of different probabilities 
        results=model.predict([bag_of_words(inp,words)])[0]
        # print(results)
         #give us the index of the greatest value in the list
        results_index=np.argmax(results) 
        tag=labels[results_index]
        if results[results_index] >0.7:
            # print(tag)
            for intent in data["intents"]:
                if intent["tag"]==tag:
                    responses=intent["responses"]
            print(random.choice(responses)) 
        else:
            print("Hmm,I didn't get that. Ask another question")   

        
chat()
        



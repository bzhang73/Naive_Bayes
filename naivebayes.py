#I have done this assignment completely on my own. I have not copied it, nor have I given my solution to anyone else. I understand that if I am involved in plagiarism or cheating I will have to sign an official form that I have cheated and that this form will be stored in my official university record. I also understand that I will receive a grade of 0 for the involved assignment for my first offense and that I will receive a grade of “F” for the course for any additional offense.
#author: Bo Zhang
import re
import time
import math
import glob
import string
import numpy
import sys

from os import listdir
from os.path import isfile, join

#stopwords dir
stopwords_dir="stopwords.txt"
#train spam dir
spam_train_dir = "train/spam/"
#train ham dir
ham_train_dir = "train/ham/"
#test spam dir
spam_test_dir = "test/spam/"
#test ham dir
ham_test_dir = "test/ham/"

#ham_train = sys.argv[1]
#spam_train = sys.argv[2]
#ham_test = sys.argv[3]
#spam_test = sys.argv[4]
#Lamda = sys.argv[5]
#iteration = sys.argv[6]
#learning_rate = sys.argv[7]


#number of wrong classification
wrong_counter = 0
#number of spam
spam_counter = 0
#number of ham
ham_counter = 0
#number of spam wrong
wrong_spam_counter = 0
#number of ham wrong
wrong_ham_counter = 0

#ham false number spam correct
spam_correct = 0
#ham false number spam wrong
spam_wrong = 0
#classified as ham wrong
ham_wrong = 0
#classified as ham correct
ham_correct = 0
#total number
total_number=0

ham_true=1

#read dictionary file
def read_dictionary_file(filename):
    with open(filename,'r',encoding='gb18030', errors='ignore') as text_file:
    
        lines = text_file.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace("\n", "")
        return lines


#read file
#remove digits
#remove special characters
#covert to lowercase
#remove duplicate
#return list
def read_file(filename):
    with open(filename,'r',encoding='gb18030', errors='ignore') as text_file:
        text = text_file.read()
#        print(text)
#        exit()
        return text

#stopwords from stopwords.txt
def getTokens(text):
    tokens = re.findall(r"[\w']+", text)
    for k in range(len(tokens)):
        tokens[k]=tokens[k].lower()
        tokens[k]=tokens[k].replace("_","")
        tokens[k]=re.sub("[0-9]+","",tokens[k])
        tokens[k]=re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "",tokens[k])
    tokens=set(tokens)
    return tokens

def training(train_files,train_labels,stopwords_tokens):
    labels=dict()
    for i in range(len(train_files)):
#        print(train_files[i]+" train")
        train_res=''
        if train_labels[i]==1:
            train_res=read_file(spam_train_dir + train_files[i])
        elif train_labels[i] == 0:
            train_res=read_file(ham_train_dir + train_files[i])
        
        train_res_tokens=getTokens(train_res)

        vector=[0] * len(stopwords_tokens)
        for j in range(len(stopwords_tokens)):
            if train_res_tokens.__contains__(stopwords_tokens[j]):
                vector[j] = 1
        vector=tuple(vector)

        if train_labels[i]==1:
            labels[vector]="SPAM"
        elif train_labels[i] == 0:
            labels[vector]="HAM"
    return labels


def testing(spam,ham,test_files,size,total_file,tokens,spam_file,ham_file,labels):
    spam_true = 0
    #ham false number spam wrong
    spam_false = 0
    #classified as ham wrong
    ham_false = 0
    #classified as ham correct
    ham_true = 0
    #total number
    total=0
#    ham_false-=1
    for i in range(len(test_files)):
        test_res=''
        
        if labels[i] == 1:
            test_res = read_file(spam_test_dir + test_files[i])
        if labels[i] == 0:
            test_res = read_file(ham_test_dir + test_files[i])
        test_tokens=getTokens(test_res)
        
        vector=[0] * len(stopwords_tokens)
        for j in range(len(stopwords_tokens)):
            if tokens.__contains__(stopwords_tokens[j]):
                vector[j]=1
        vector=tuple(vector)

#        print(total_file)
#        exit()
        spam_test_probability=calculate_probability(vector,tokens,spam,spam_file,size,total_file)
        ham_test_probability=calculate_probability(vector,tokens,ham,ham_file,size,total_file)

    
        if spam_test_probability>=ham_test_probability and labels[i]==1:
#            print(test_files[i]+" classified: SPAM True")
            spam_true += 1
        elif spam_test_probability>=ham_test_probability and labels[i]==0:
#            print(test_files[i]+" classified: SPAM False")
            spam_false += 1
        elif spam_test_probability<ham_test_probability and labels[i]==1:
#            print(test_files[i]+" classified: HAM False")
            ham_false += 1
        elif spam_test_probability<ham_test_probability and labels[i]==0:
#            print(test_files[i]+" classified: HAM True")
            ham_true += 1
            total+=1
    return spam_true,spam_false,ham_true,ham_false,total


def testing_res_without(spam,ham,train_files,size,total_file,tokens,spam_file,ham_file,labels,spam_tokens,
                        ham_tokens,spam_length,ham_length):
    spam_true = 0
    #ham false number spam wrong
    spam_false = 0
    #classified as ham wrong
    ham_false = 0
    #classified as ham correct
    ham_true = 0
    #total number
    total=0

    for i in range(len(train_files)):
#        print(spam_train_dir)
#        print(train_files[i])
#        exit();
        test_res=''
        if i<spam_file :
            test_res = read_file(spam_train_dir + train_files[i])
        else:
            test_res = read_file(ham_train_dir + train_files[i])
        test_tokens=getTokens(test_res)
        
        vector=[0] * len(stopwords_tokens)
        for j in range(len(stopwords_tokens)):
            if tokens.__contains__(stopwords_tokens[j]):
                vector[j]=1
        vector=tuple(vector)

#        print(total_file)
#        exit()
        spam_test_probability=calculate_probability(vector,ham_tokens,spam,spam_length,size,total_file)
        ham_test_probability=calculate_probability(vector,spam_tokens,ham,ham_length,size,total_file)
    #        print(spam_test_probability)
#        print(ham_test_probability)
#        exit()
        if spam_test_probability>=ham_test_probability and labels[i]==1:
    #            print(test_files[i]+" classified: SPAM True")
            spam_true += 1
        elif spam_test_probability>=ham_test_probability and labels[i]==0:
            #            print(test_files[i]+" classified: SPAM False")
            spam_false += 1
        elif spam_test_probability<ham_test_probability and labels[i]==1:
            #            print(test_files[i]+" classified: HAM False")
            ham_false += 1
        elif spam_test_probability<ham_test_probability and labels[i]==0:
            #            print(test_files[i]+" classified: HAM True")
            ham_true += 1
        total+=1
    return spam_true,spam_false,ham_true,ham_false,total


def testing_without(spam,ham,test_files,size,total_file,tokens,spam_file,ham_file,labels,spam_tokens,ham_tokens,time):
    spam_true = 0
    #ham false number spam wrong
    spam_false = 0
    #classified as ham wrong
    ham_false = 0
    #classified as ham correct
    ham_true = time
    #total number
    total=0
    for i in range(len(test_files)):
        test_res=''
        if labels[i] == 1:
            test_res = read_file(spam_test_dir + test_files[i])
        if labels[i] == 0:
            test_res = read_file(ham_test_dir + test_files[i])
        test_tokens=getTokens(test_res)

        vector=[0] * len(stopwords_tokens)
        for j in range(len(stopwords_tokens)):
            if tokens.__contains__(stopwords_tokens[j]):
                vector[j]=1
        vector=tuple(vector)

#        print(total_file)
#        exit()
        spam_test_probability=calculate_probability(vector,ham_tokens,spam,spam_file,size,total_file)
        ham_test_probability=calculate_probability(vector,spam_tokens,ham,ham_file,size,total_file)
#        print(spam_test_probability)
#        print(ham_test_probability)
#        exit()
        if spam_test_probability>=ham_test_probability and labels[i]==1:
#            print(test_files[i]+" classified: SPAM True")
            spam_true += 1
        elif spam_test_probability>=ham_test_probability and labels[i]==0:
#            print(test_files[i]+" classified: SPAM False")
            spam_false += 1
        elif spam_test_probability<ham_test_probability and labels[i]==1:
#            print(test_files[i]+" classified: HAM False")
            ham_false += 1
        elif spam_test_probability<ham_test_probability and labels[i]==0:
#            print(test_files[i]+" classified: HAM True")
            ham_true += 1
        total+=1
    return spam_true,spam_false,ham_true,ham_false,total


#count class tokens frequence
def frequencies_calculate(stopwords_tokens, feature_labels, class_label):
    frequencies = dict()
    for (i, vector) in enumerate(feature_labels):
        if feature_labels[vector] == class_label:
            for j in range(len(stopwords_tokens)):
                token = stopwords_tokens[j]
                if vector[j] == 1:
                    if frequencies.__contains__(token):
                        frequencies[token] = frequencies[token] + 1
                    else:
                        frequencies[token] = 1
    return frequencies




def calculate_probability(vector, tokens, frequencies, frequency, size, total_document):
#    print(total_document)
    probability = frequency / total_document
#    print(total_document)
#    print(frequency)
#    exit()

    laplace_estimate_log_probability = 0
    for (i, token) in enumerate(tokens):
        test_feature = vector[i]
        if test_feature == 1:
            if frequencies.__contains__(token):
                probOfTokenBelongingToClass = (frequencies[token] + 1)/ (frequency + size)
                laplace_estimate_log_probability += math.log(probOfTokenBelongingToClass+1.0, 2)
            else:
                probOfTokenBelongingToClass = (0 + 1) / (frequency + size)
                laplace_estimate_log_probability += math.log(probOfTokenBelongingToClass+1.0, 2)
#    print(laplace_estimate_log_probability)
#    exit()
    laplace_estimate_log_probability += math.log(probability, 2)
    return laplace_estimate_log_probability



#main

train_spam_files = sorted([f for f in listdir(spam_train_dir) if isfile(join(spam_train_dir, f))])
train_ham_files = sorted([f for f in listdir(ham_train_dir) if isfile(join(ham_train_dir, f))])
test_spam_files = sorted([f for f in listdir(spam_test_dir) if isfile(join(spam_test_dir, f))])
test_ham_files = sorted([f for f in listdir(ham_test_dir) if isfile(join(ham_test_dir, f))])

train_files = list(train_spam_files)
train_files.extend(train_ham_files)

test_files = list(test_spam_files)
test_files.extend(test_ham_files)

train_labels = [1] * len(train_spam_files)
train_labels.extend([0] * len(train_ham_files))

test_true_labels = [1] * len(test_spam_files)
test_true_labels.extend([0] * len(test_ham_files))

spam_file = len(train_spam_files)

ham_file = len(train_ham_files)

spam_probability = spam_file / (len(train_spam_files) + len(train_ham_files))

ham_probability = ham_file / (len(train_spam_files) + len(train_ham_files))

#print("Spam train file: " + str(spam_file))
#print("Ham train file: " + str(ham_file))
#print("Spam train probability: " + str(spam_probability))
#print("Ham train probability: " + str(ham_probability))
#
#print('')

stopwords_tokens = read_dictionary_file(stopwords_dir)
for k in range(len(stopwords_tokens)):
    stopwords_tokens[k]=stopwords_tokens[k].lower()
    stopwords_tokens[k]=stopwords_tokens[k].replace("_","")
#            stopwords_tokens[k]=re.sub("[0-9]+","",tokens[k])
    stopwords_tokens[k]=re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "",stopwords_tokens[k])
#stopwords_tokens=set(stopwords_tokens)

#print(stopwords_tokens)
#exit()

#training
#print("Training")
feature_labels = dict()
feature_labels=training(train_files,train_labels,stopwords_tokens)

spam_frequencies = frequencies_calculate(stopwords_tokens, feature_labels, "SPAM")
ham_frequencies = frequencies_calculate(stopwords_tokens, feature_labels, "HAM")
#print(spam_frequencies)

spam_tokens=[]
spam_length=0
for key in spam_frequencies:
    spam_tokens.append(key)
    spam_length+=spam_frequencies[key]

ham_tokens=[]
ham_length=0
for key in ham_frequencies:
    ham_tokens.append(key)
    ham_length+=ham_frequencies[key]
#print(spam_tokens)
#print(len(spam_frequencies))
#print(ham_frequencies)
#print(len(ham_frequencies))
#print(ham_tokens)
#print(ham_length)
#print(spam_length)
#exit()
news_ids=spam_tokens
news_ids.extend(ham_tokens)
fin_tokens = list(set(news_ids))
#print(news_ids)
#print(fin_tokens)
#exit()
spam_tokens=[]
for key in spam_frequencies:
    spam_tokens.append(key)

dictionary_size = len(stopwords_tokens)
total_train_file = len(train_spam_files) + len(train_ham_files)

#print(len(ham_tokens))
#print(len(spam_tokens))
#exit()

#spam_correct,spam_wrong,ham_correct,ham_wrong,total_number=testing_without(spam_frequencies,ham_frequencies,test_files,total_train_file,spam_tokens,ham_tokens,spam_file,ham_file,test_true_labels)

spam_correct,spam_wrong,ham_correct,ham_wrong,total_number=testing_res_without(spam_frequencies,ham_frequencies,train_files,len(fin_tokens),total_train_file,fin_tokens,spam_file,ham_file,test_true_labels,spam_tokens,ham_tokens,spam_length,ham_length)
#
#
#
#print()
#wrong_counter2=spam_probability+ham_wrong
wrong_counter2=abs(ham_probability*len(train_files)-ham_correct)+abs(spam_probability*len(train_files)-ham_wrong)
accuracy1 = ((len(test_files) - wrong_counter2) / len(test_files))* 100
#print("accuracy: " + str(accuracy1) + " %")
#print()

#exit()

spam_correct,spam_wrong,ham_correct,ham_wrong,total_number=testing_without(spam_frequencies,ham_frequencies,test_files,len(fin_tokens),total_train_file,fin_tokens,spam_file,ham_file,test_true_labels,spam_tokens,ham_tokens,0)

#wrong_counter=spam_wrong+ham_wrong
wrong_counter=abs(ham_probability*len(train_files)-ham_correct)+abs(spam_probability*len(train_files)-ham_wrong)
accuracy2 = ((len(test_files) - wrong_counter) / len(test_files)) * 100
#print("accuracy: " + str(accuracy2) + " %")
#print(ham_correct)
#
#
#
#print()#exit()


spam_correct,spam_wrong,ham_correct,ham_wrong,total_number=testing(spam_frequencies,ham_frequencies,test_files,len(stopwords_tokens),total_train_file,stopwords_tokens,spam_file,ham_file,test_true_labels)

wrong_counter1=abs(ham_probability*len(train_files)-ham_correct)+abs(spam_probability*len(train_files)-ham_wrong)
accuracy3 = ((len(test_files) - wrong_counter1) / len(test_files)) * 100
#print("accuracy: " + str(accuracy3) + " %")
#print(ham_correct)




#print('sopwords length: ' + str(dictionary_size))

#print()



print('Train                   Test_with_stopwords           Test_without_stopwords')
print(accuracy1,'       ',accuracy2,'           ',accuracy3)
print('')

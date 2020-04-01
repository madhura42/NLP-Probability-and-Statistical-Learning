#!/usr/local/bin/python3
# Naive Bayes Spam Classifier

import sys
import os
import pandas as pd
import numpy as np

train_dir = [sys.argv[1] + "/notspam", sys.argv[1] + "/spam"]
test_dir = sys.argv[2]
output = sys.argv[3]

# =============================================================================
# import data
# =============================================================================

#Training data
training_df = pd.DataFrame(columns=['filename', 'mail', 'label'])
label = 0

for path in train_dir:
    with os.scandir(path) as files:
        for file in files:
            filename = file.name
            with open(file, "r",  encoding = "CP437") as file:
                mail = file.read()
                mail = mail.lower().split()     #Preprocess data: convert to lowercase and tokenize

                training_df = training_df.append( { 'filename' : filename, 'mail' : mail, 'label' : label }, ignore_index = True )
                
    label += 1

#Testing data
testing_df = pd.DataFrame(columns=['filename', 'mail', 'pred_label'])

with os.scandir(test_dir) as files:
    for file in files:
        filename = file.name
        with open(file, "r",  encoding = "CP437") as file:
            mail = file.read()
            mail = mail.lower().split()     #Preprocess data: convert to lowercase and tokenize

            testing_df = testing_df.append( { 'filename' : filename, 'mail' : mail, 'pred_label' : 'lorem ipsum' }, ignore_index = True )

#Getting actual 'labels'
test_groundtruth = pd.read_csv(r'test-groundtruth.txt', header=None,  names=['filename', 'label'], sep=" ")

testing_df = pd.merge(testing_df, test_groundtruth, on='filename')

# =============================================================================

total_mails = training_df.shape[0]

notspam_mail_count = training_df['label'].value_counts()[0]
spam_mail_count = training_df['label'].value_counts()[1]

prob_spam_mail = spam_mail_count / total_mails
prob_notspam_mail = notspam_mail_count / total_mails

# =============================================================================
# Train
# =============================================================================

# Calculate Term Frequency
TF_notspam = {}
TF_spam = {}

for idx in range(total_mails):
    
    for word in training_df['mail'][idx]:  #error
        
        if training_df['label'][idx] == 0:      #notspam
            TF_notspam[word] = TF_notspam.get(word, 0) + 1
        
        elif training_df['label'][idx] == 1:    #spam
            TF_spam[word] = TF_spam.get(word, 0) + 1

notspam_word_count = sum( list(TF_notspam.values()) )
spam_word_count = sum( list(TF_spam.values()) )

#Calculate probablities
prob_spam = {}
prob_notspam = {}

for word in TF_spam:
    prob_spam[word] = TF_spam[word] / spam_word_count
    
for word in TF_notspam:
    prob_notspam[word] = TF_notspam[word] / notspam_word_count

# =============================================================================
# Predict
# =============================================================================

for idx in range(testing_df.shape[0]):
    
    p_spam = np.log(prob_spam_mail)
    p_notspam = np.log(prob_notspam_mail)
    
    for word in testing_df['mail'][idx]:   #message level
        
        if word in prob_spam:
            p_spam += np.log(prob_spam[word])
        
        else:
            p_spam += np.log( 10**-20 )
    
    for word in testing_df['mail'][idx]:
        
        if word in prob_notspam:
            p_notspam += np.log(prob_notspam[word])

        else:
            p_notspam += np.log( 10**-20 )

    if p_spam >= p_notspam:
        
        testing_df.loc[idx, 'pred_label'] = 'spam'
    
    else:
        testing_df.loc[idx, 'pred_label'] = 'notspam'

# =============================================================================
# Validate Model
# =============================================================================

true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0

for idx in range(testing_df.shape[0]):
    
    true_positives += int(testing_df['label'][idx] == 'spam' and testing_df['pred_label'][idx] == 'spam')
    true_negatives += int(testing_df['label'][idx] == 'notspam' and testing_df['pred_label'][idx] == 'notspam')
    false_positives += int(testing_df['label'][idx] == 'notspam' and testing_df['pred_label'][idx] == 'spam')
    false_negatives += int(testing_df['label'][idx] == 'spam' and testing_df['pred_label'][idx] == 'notspam')

confusion_matrix = pd.crosstab(testing_df['label'], testing_df['pred_label'], rownames=['Actual'], colnames=['Predicted'], margins=True)
print("Confusion Matrix:\n", confusion_matrix, "\n")

accuracy = ((true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)) * 100
precision = (true_positives / (true_positives + false_positives)) * 100
sensitivity = (true_positives / (true_positives + false_negatives)) * 100
specificity = (true_negatives / (true_negatives + false_positives)) * 100

print("Accuracy: " + str(round(accuracy, 2)) + "%")
print("Precision: " + str(round(precision, 2)) + "%")
print("Sensitivity: " + str(round(sensitivity, 2)) + "%")
print("Specificity: " + str(round(specificity, 2)) + "%")

# =============================================================================
# Write output to file
# =============================================================================

testing_df.to_csv(output, columns=['filename', 'pred_label'], header=None, index=None, sep=" ")

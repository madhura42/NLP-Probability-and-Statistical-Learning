# Part 1 : POS tagger 

Implementation of Part-of-speech tagger on a dataset based of brown corpus containing 12 part-of-speech tags namely ADJ (adjective), ADV (adverb), ADP (adposition), CONJ (conjunction), DET (determiner), NOUN, NUM (number), PRON (pronoun), PRT (particle), VERB, X (foreign word), and . (punctuation mark)

## Training: 
In the *train(self, data)*  method, data consist of a list of tuples, wherein the tuple consist of the sentence and the corresponding tags. We use this data to learn the following 
 
•	transition probability P(Si | Si-1) where Si is the tag for the current word and Si-1 is the tag of the previous word in the sentence. For the first word of the sentence, the transition is stored as start which accounts for the number of times the tag occurred as the starting tag of a sentence.

•	emission probability P(Wi | Si) is the probability of the word given a tag.

•	Prior probability P(Si), the probability of a tag. 
These probabilities are stored in the class Solver object.

## Challenges:
The words that occur in the testing data but are not present in the training set, the probability of the emission of such words is set to a small value of 1e-10 and they are assigned ‘noun’ tag. This decision was made because even with the skeleton code assigning noun to all the words, we were getting decent accuracy. Also, nouns like person name, places cannot all possibly occur in the training set.
The transition probability of Si-1 to Si not seen in the training data is set to 1/(total transitions to Si).

## Approach:

### Naïve Bayes:
Naive bayes is the simplest approach, we consider each word independent of the other and this approach being as naive as it is still gives a decent accuracy.
*simplified(self, sentence):*
The formula P(Si | Wi) = P(Wi | Si) * P(Si) / P(Wi) where we ignore the P(Wi) as it is a constant factor.
Therefore, P(Si | Wi) = P(Wi | Si) * P(Si)
The one with the highest probability is considered as tag for the word in the sentence.

### Viterbi: 
*hmm_viterbi(self, sentence):*
Viterbi is a dynamming program approach where we find the maximum a posteriori (MAP) probability to discover the optimal path.
vt(t+1) = max(P(Si | Wi) * vt(t) * P(Si | Si-1))
where P(Si | Wi) is emission and P(Si | Si-1) is the transnition probability.
The transition and emission probabilities calculated while training is used here. The words that do not exist are assigned a 'noun' tag and a small probability. 
 
### Gibbs Sampling: 
*complex_mcmc(self, sentence):*
Gibbs sampling is one of the techniques used under Markov chain Monte Carlo. We start from a random sample or some initial sample, in this case our initial sample consist of 'noun' tag for all the words in the sentence. We start with a burn-in period of 1000 iterations, consider the last sample as input to the next loop. All the samples from the burn-in period are discarded. We calculate the maximum of P(Si | Wi) * P(Si | Si-1), for all the words of the sentence, except for the last word of the sentence where we consider the last tag to have two parents, the previous tag as well as the first tag of the sentence which is given by P(Si | Wi) * P(Si | Si-1) * P(Si | S0).
 
## Accuracy:
The accuracy obtained over the testing set of 2000 sentences is as below. Highest accuracy is obtained through viterbi, as we consider the transition as well as the previous probability.

![alt text](https://github.iu.edu/cs-b551-fa2019/svbhakth-mabartak-knikharg-a3/blob/master/Accuracy.PNG)

# --------------------------------------------------------------------------------------------------------------------------------------

# Part 2: Code Breaking

We have an encrypted document which uses two techniques: Replacement and Rearrangement. In Replacement, each letter of the alphabet is replaced with another letter of the alphabet. (For example, all a's may have been replaced with f's, b's with z's, etc.). In rearrangement, the order of the words is scrambled. if n = 4 and the mapping function is f(0) = 2; f(1) = 0; f(2) = 1; f(3) = 3, then the string ‘test’ would be rearranged to be ‘estt’. Our goal is to decrypt the document such that the text appears as English like words. Since we don’t know the mapping that we used for encoding, we need to start with a random cipher.

## Approach:

For this problem, we first generate a cipher containing elements from ‘a’ to ‘z’ in random manner. We call this cipher as current cipher. Next, we train the corpus by calculating the probabilities. For calculating the initial probabilities, we create a simple dictionary of 26 keys (key ranges from ‘a’ to ‘z’). We calculate the initial by summing over all the occurrences of the letter at the start of the word. (for eg, consider a sentence “as oliver ate an apple”. In this sentence the initial probability of the letter ‘a’ will be 4). Similarly, we calculate the initial of all the letters.  The initial probability will be the initial for the letter over the total count of words in the corpus. For calculating the transition, we create dictionary of dictionaries where each in parent dictionary will be the prior and the child dictionary will have the corresponding conditional.  (for eg the transition for ‘aa’, ‘ab’, ‘ac’,  ‘ba’, ’bb’, ’bc’ will look something like this {‘a’:{‘a’ : 10 , ‘b’:12, ’c’:14},  ‘b’:{‘a’ : 20 , ‘b’:22, ’c’:24 } ). The transition probabilities from a first letter to second letter will be the transition over sum of all the transition from first letter to any other letter. 

Now we calculate another cipher, which is a modification of the current cipher. Let’s call this cipher as the proposed cipher. This modification is done by the function improve_cipher which swaps position of any one pair of letters randomly.

We have two another functions named create_dict and decrypt_text. The function create_dict takes as input a key (in our case cipher) and creates a dictionary for key where each letter in the cipher maps to an alphabet. (for eg. if the cipher is "dghik…." this function will create a dictionary like {d:a,g:b,h:c....}) The function decrypt_text takes a text and applies the cipher/key on the text and returns the decrypted text. 

To decrypt the text, we use Metropolis Hastings algorithm. In this algorithm, we use the current cipher and proposed cipher. We calculate the probabilities of the document (ie. text). Probabilities of the document is calculated by calculating the probabilities of each word in the document using the initial and transition probabilities that we calculated earlier. Let’s call the probability of document D decrypted by cipher T as P(D) and the probability of document D’ decrypted by cipher T’ as P(D’). To prevent floating point underflow, we sum the log of the probability of each word to get the probability of the document. We calculate the probability of the document using each cipher. Since the probabilities is in the log space, we take the exponential value of the difference between both document probabilities (ie. P(D’)-P(D)). Let’s call this as the acceptance probability. How do we calculate the acceptance criteria? For this we use binomial distribution of a random coin toss. If the acceptance probability is greater than the probability of the coin toss, then the proposed cipher becomes the current cipher else the current cipher is retained and we shuffle the rearrange table. 

## Other methods tried:

1.	To improve the proposed cipher, we tried to swap two pairs of letters instead of one. 
2.	To calculate the acceptance criteria, instead of taking the binomial distribution, we compared it with random.random() value.

# --------------------------------------------------------------------------------------------------------------------------------------

# Part 3: Naïve Bayes Spam Classifier

The goal of this problem is to build a Naïve Bayes text classifier to identify whether a document (in our case, an email) is spam or not spam. The words in our document are the features used for training the model. For a given document, we calculate the posterior probability that the document is spam given the words are present in the document. We use the Naïve Bayes assumption that given the document is spam (or not spam), no two words in the document are dependent. In other words, the posterior probability can be calculated as the product of the prior (marginal) probability of the document being spam (or not spam) and the likelihood (conditional probability) of a word being present in the document given it is spam (or not spam).

## Approach:

#### Preprocessing the data:

We have used the pandas library to read in the data as a dataframe and used Codeplex 437 encoding to retain all characters in the document and perform the text preprocessing. We have tried multiple approaches to preprocessing which included:

1.	Reading in the text as an email object (using the email standard library) and extracting the mail body while discarding the header and attachments
2.	Stripping the HTML tags (using the HTML standard library)
3.	Convert all text to lowercase and tokenizing using string methods
4.	Removing excess whitespace (using the regular expressions library)
5.	Removing stopwords and lemmatizing (using the nltk library)

However, the model performance was very poor (accuracy of ~48%) after preprocessing the text, so we decided to simplify our approach by only converting the text to lowercase and tokenizing (step 3) and discarding the other methods. This improved the model performance dramatically (accuracy of ~99%) and we decided to stick to this approach.

#### Training the model:

For training the model, the approach we have used is a bag-of-words model in which we represent the mail corpus as an unstructured bag of individual words and discarding the grammatical structure of the text. We have considered three different approaches to this:

1. Our initial approach was to list all the unique words (n) in the mail corpus and represent each mail as an n-dimensional vector where the entries corresponding to each word is a binary value – 1 if present, 0 if not present. The result is a rectangular m x n matrix, where m is the number of mails in the corpus and n is the number of unique words in the corpus. This was a computationally intensive process given the number of unique words (~129,000 unique words in the training corpus) and thus was discarded in favor of a different approach.

2. The second approach we tried was to compute Term Frequency (TF) or the number of times each word occurs in the corpus. We implemented this using python dictionary objects and created two separate dictionaries for ‘spam’ and ‘notspam’ categories. We then computed the probabilities of each word by dividing the frequencies with the total word count for each category. This approach was far less computationally intensive. 

3. The third approach we tried was to calculate Term Frequency-Inverse Document Frequency (TF-IDF). The Term Frequency approach has the drawback that highly frequent words have higher probabilities and might bias the model. The idea is to add a ‘weight’ to the TF by how frequently they appear in all mails. IDF is a score of how rare the word is in a mail corpus and higher the IDF, rarer the word and vice versa. Thus, frequent words like “to” or “from” which are commonly found in mails are penalized. However, this approach which is implemented by checking how many times each unique word in the corpus appears in documents is computationally intensive and was thus discarded in favor of the simpler and more efficient TF approach (approach 2).

#### Predicting

We trained the model (computing probabilities) using approach 2 and used the computed probabilities to compute posteriors on the testing mail corpus. To avoid floating point underflow when multiplying probabilities, we applied a log transformation and summed the transformed probabilities. The next challenge we faced was encountering words in testing data which were not present in training data. In this case the marginal probability of the word will be zero and thus the posterior probability of a mail being spam (or not spam) will also be zero. To avoid this, we set the probability of such newly encountered words to a very small value (1e-20).

For each mail in the testing corpus, we compute the posterior probability for it to be spam (or not spam). Whichever among the two probabilities is greater, we assign the corresponding ‘spam’ or ‘notspam’ label to the mail.

## Model Performance and Evaluation:

We have used the model to classify the testing mail corpus and calculated True Positives (TP), True Negatives (TN), False Positives (FP) and False Negatives (FN) by comparing our predictions with the provided results.
The confusion matrix is as follows:

![alt text](https://github.com/madhura42/NLP-Probability-and-Statistical-Learning/blob/master/part3/ConfusionMatrix.PNG)

We have used the following metrics to measure the performance of the model:

![alt text](https://github.com/madhura42/NLP-Probability-and-Statistical-Learning/blob/master/part3/Metrics.PNG)

1.	Accuracy: Accuracy is the ratio of the mails correctly identified as ‘spam’ or ‘notspam’ to the entire testing corpus. In other words, it measures how many mails we correctly labeled out of all mails. It is calculated as (TP+TN)/(TP+FP+FN+TN)

Our model is able to accurately classify 98.86% of the mails in the testing mail corpus.

2.	Precision: Precision is the ratio of mails correctly identified as ‘spam’ by the model to all mails identified as ‘spam’. It measures how many mails which our model identified as ‘spam’ are actually ‘spam’. Precision is calculated as TP/(TP+FP)

Of all the mails that our model has identified as ‘spam’, our model is correct in 99.07% of the cases.

3.	Sensitivity (True Positive Rate/Recall): Sensitivity is the ratio of mails correctly identified as ‘spam’ by the model to all mails which are actually ‘spam’. It measures how correctly our model has identified ‘spam’ mail. Sensitivity is calculated as TP/(TP+FN)

Of all the mails which are actually ‘spam’, our model has identified 98.48% of cases correctly.

4.	Specificity (True Negative Rate): Specificity is the ratio of mails correctly identified as ‘notspam’ by the model to all mails which are actually ‘notspam’. It measures how correctly our model has identified ‘notspam’ mail. Specificity is calculated as TN/(TN+FP)

Of all the mails which are actually ‘notspam’, our model has identified 99.2% of cases correctly.

#!/usr/local/bin/python3

import sys
import random
import numpy as np
#import time

def read_clean_file(filename):
        with open(filename, "r", encoding = 'utf8') as file:
            return "".join([ ("".join( [ i if i.islower() or i == ' ' else '' for i in line ] ) + " ") for line in file ] )

#corpus = read_clean_file('C:\\Users\\Madhura\\Desktop\\Fall Semester\\Elements of AI\\Assignment 3\\svbhakth-mabartak-knikharg-a3-master\\part2\\corpus.txt')
#text = read_clean_file('C:\\Users\\Madhura\\Desktop\\Fall Semester\\Elements of AI\\Assignment 3\\svbhakth-mabartak-knikharg-a3-master\\part2\\encrypted-text-1.txt')
   
def encode(input, replace_table, rearrange_table):
    # apply replace table
    str2 = input.translate({ ord(i):ord(replace_table[i]) for i in replace_table })

    # pad with spaces to even multiple of rearrange table
    str2 +=  ' ' * (len(rearrange_table)-(len(str2) %  len(rearrange_table)))

    # and apply rearrange table
    return ( "".join(["".join([str2[rearrange_table[j] + i] for j in range(0, len(rearrange_table))]) for i in range(0, len(input), len(rearrange_table))]))

#swapping any two random letters in cipher
def improve_cipher(cipher):

    pos1 = random.randint(0, len(list(cipher))-1)
    pos2 = random.randint(0, len(list(cipher))-1)
    
    if pos1==pos2:
        pos2 = random.randint(0, len(list(cipher))-1)
    else:
        cipher = list(cipher)
        temp_pos1 = cipher[pos1]
        temp_pos2 = cipher[pos2]
        cipher[pos1]=temp_pos2
        cipher[pos2]=temp_pos1
        
    return cipher

#Calculating initial and transition probabilities on text
def train_prob(corpus):
    word = corpus.lower().split()
    transition = {}
    initial = {}
    for i in range(len(word)):
        for j in range(len(word[i])):
            if j==0:
                initial[word[i][j]] = initial.get(word[i][j],0)+1
            if word[i][j] in transition.keys():
                if word[i][j-1] in transition[word[i][j]].keys():
                    transition[word[i][j]][word[i][j-1]] = transition[word[i][j]].get(word[i][j-1],0) + 1 
                else:
                    transition[word[i][j]][word[i][j-1]] = {} 
                    transition[word[i][j]][word[i][j-1]] = 1 
            else:
                transition[word[i][j]] = {}
                transition[word[i][j]][word[i][j-1]] = 1
    
    #Calculating initial probabilities            
    initial = { k: v/total for total in (sum(initial.values()),) for k, v in initial.items() }
    #Calculating transition probabilities
    for key, value in transition.items():
        transition[key] = { k : v/total for total in (sum(value.values()),) for k, v in value.items() }
    
#    for key, value in transition.items():
#        max_val = max(value.items(), key = lambda x : x[1])
#        transition[key] = { max_val[0] : max_val[1] }
        
    return initial, transition
#Calculating initial and transition probabilities on corpus text


#Creates dictionary which contains mapping of alphabets to cipher
def create_dict(cipher):
    cipher_dict = {}
    alphabet_list = [ chr(i) for i in range(97, 123) ]
    for i in range(len(cipher)):
        cipher_dict[alphabet_list[i]]=cipher[i]
    return cipher_dict

#Decrypts by applying cipher on text
def text_decrypt(text,cipher):
    cipher_dict = create_dict(cipher)
    text = list(text)
    text_new = ""
    for ele in text:
        if ele.lower() in cipher_dict:
            text_new += cipher_dict[ele.lower()]
        else:
            text_new+= " "
    return text_new

#Calculates the probabilities of the document
def get_cipher_prob(text,cipher):
    decrypted_text = text_decrypt(text,cipher)
    
    prob_doc = 0
    for w in decrypted_text.split():
        prob_word = 0
        for i in range(len(w)-1):
            if i == 0:
                prob_word = np.log( initial.get(w[i], 10**-15) )
            else:
                prob_word += np.log( transition.get(w[i], {}).get(w[i-1], 10**-15) )    
        prob_doc += prob_word
        
    return prob_doc

#def acceptance_criteria(log_proposal, log_current):
#	return (random.random() < np.exp(log_proposal - log_current))


def bino(probability):
    uniform = random.uniform(0,1)
    if uniform>=probability:
        return False
    else:
        return True

#Takes an input and applies metropolis hastings algorithm. Returns the best cipher.
def decrypt(text, corpus, iterations = 10000):
    
    initial, transition = train_prob(corpus)
    
    current_cipher = [ chr(i) for i in range(97, 123) ] #generate a random cipher of ASCII to start
    random.shuffle(current_cipher)
    
    rearrange_table = list(range(0,4))
   
    for i in range(iterations):
        
        proposed_cipher = improve_cipher(current_cipher)
        
        prob_current_cipher = get_cipher_prob(text, current_cipher)
        prob_proposed_cipher = get_cipher_prob(text, proposed_cipher)
        acceptance_probability = min(1,np.exp(prob_proposed_cipher-prob_current_cipher))
#        if prob_current_cipher > score:
#            best_state = current_cipher
        
        if bino(acceptance_probability):
            current_cipher = proposed_cipher
        
        if not bino(acceptance_probability):
            random.shuffle(rearrange_table)
            
        text = encode(text, create_dict(current_cipher), rearrange_table)
        
        if i%100 == 0:
            print("iter",i,":", text[:99])
            
    return text
  
if __name__== "__main__":
    if(len(sys.argv) != 4):
        raise Exception("usage: ./break_code.py coded-file corpus output-file")

    encoded = read_clean_file(sys.argv[1])
    corpus = read_clean_file(sys.argv[2])
    initial, transition = train_prob(corpus)
    decoded = decrypt(encoded, corpus, iterations = 100000)

    with open(sys.argv[3], "w") as file:
        print(decoded, file=file)

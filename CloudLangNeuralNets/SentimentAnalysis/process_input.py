# -*- coding: utf-8 -*-

import imdb_preprocess
import os
import six.moves.cPickle as pickle
import dill


dataset_path='/Tmp/bastienf/aclImdb/'



   


def build_vector(review):
    dill.settings['recurse'] = True
    f = open('imdb.dict.pkl', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    dictionary = u.load()
    
    #path = dataset_path
    #dictionary = imdb_preprocess.build_dict(os.path.join(path, 'train'))
    train_x = grab_data(review, dictionary)   
    train_y = [1] * len(train_x)
    
    #print (train_x)
    
    
    f = open('imdb.review.pkl', 'wb')
    pickle.dump((train_x, train_y), f, -1)
    f.close()
    return train_x
    



def grab_data(review, dictionary):
    sentences = []
    currdir = os.getcwd()
   
    sentences.append(review)
    print (sentences)
    os.chdir(currdir)
    #sentences = imdb_preprocess.tokenize(sentences)
    print (sentences)
    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs
    
def main():
     build_vector('this movie is not good')

    
if __name__ == '__main__':
    main()
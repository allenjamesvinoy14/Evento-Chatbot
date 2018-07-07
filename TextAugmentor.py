
# coding: utf-8

# In[1]:


import requests
import bs4 as BeautifulSoup
import nltk


def get_synonym(word):
        url = "http://www.thesaurus.com/browse/" + word +"?s=t"
        r = requests.get(url)
        soup = BeautifulSoup.BeautifulSoup(r.content,"lxml")
        word_data = soup.find_all("a",{"class":"css-1hn7aky e1s2bo4t1"})
        syns = [w.text for w in word_data]
        return syns[:4]

class TextAugmentor:

    def __init__(self,sentence):

        self.sentence = sentence
        self.target_words = nltk.word_tokenize(sentence)
        self.target_words = [i for i in self.target_words if i.isalpha() and len(i)>5]
        self.target_synonyms = {}

        for i in self.target_words:
            self.target_synonyms[i] = get_synonym(i)


        self.generated_sentences = []

    def GenerateSentences(self):
        for i in self.target_words:
            split_sentence = nltk.word_tokenize(self.sentence)
            target_index = split_sentence.index(i)

            for j in self.target_synonyms[i]:
                new_sentence = split_sentence
                new_sentence[target_index] = j

                if j[0] in ['a','e','i','o','u'] and target_index!=0 and new_sentence[target_index-1]=="a":
                    new_sentence[target_index-1] = "an"

                self.generated_sentences.append(' '.join(new_sentence))


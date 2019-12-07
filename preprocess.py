import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class Preprocess:
    def sentence_tokenizer(self, doc):
        return sent_tokenize(doc)
    
    def word_tokenizer(self, sentence):
        return word_tokenize(sentence) 
    
    def remove_stop_wrods(self, tokenized_sentences):
        stop_words = set(stopwords.words('english'))
        return [t for t in tokenized_sentences if not t in stop_words] 
    
    def clean_text_form(self, doc):
        doc = doc.lower()
        
        doc = re.sub(r"i'm", "i am", doc)
        doc = re.sub(r"aren't", "are not", doc)
        doc = re.sub(r"couldn't", "counld not", doc)
        doc = re.sub(r"didn't", "did not", doc)
        doc = re.sub(r"doesn't", "does not", doc)
        doc = re.sub(r"don't", "do not", doc)
        doc = re.sub(r"hadn't", "had not", doc)
        doc = re.sub(r"hasn't", "has not", doc)
        doc = re.sub(r"haven't", "have not", doc)
        doc = re.sub(r"isn't", "is not", doc)
        doc = re.sub(r"it't", "had not", doc)
        doc = re.sub(r"hadn't", "had not", doc)
        doc = re.sub(r"won't", "will not", doc)
        doc = re.sub(r"can't", "cannot", doc)
        doc = re.sub(r"mightn't", "might not", doc)
        doc = re.sub(r"mustn't", "must not", doc)
        doc = re.sub(r"needn't", "need not", doc)
        doc = re.sub(r"shouldn't", "should not", doc)
        doc = re.sub(r"wasn't", "was not", doc)
        doc = re.sub(r"weren't", "were not", doc)
        doc = re.sub(r"won't", "will not", doc)
        doc = re.sub(r"wouldn't", "would not", doc)
        
        doc = re.sub(r"\'s", " is", doc)
        doc = re.sub(r"\'ll", " will", doc)
        doc = re.sub(r"\'ve", " have", doc)
        doc = re.sub(r"\'re", " are", doc)
        doc = re.sub(r"\'d", " would", doc)
        
        return doc
    
    def remove_unchars(self, doc):
        doc = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', doc, flags=re.MULTILINE)
        doc = re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', '', doc)
        doc = re.sub(r'\b[0-9]+\b\s*', '', doc)
        
        return doc
    
    
    def get_pos(self, sentence):
        """
            This method is used for POS tagging
        """
        pos = []
        for word in sentence:
            w, p = nltk.pos_tag([word])[0]
            if p.startswith('J'):
                pos.append((w, wordnet.ADJ))
            elif p.startswith('V'):
                pos.append((w, wordnet.VERB))
            elif p.startswith('N'):
                pos.append((w, wordnet.NOUN))
            elif p.startswith('R'):
                pos.append((w, wordnet.ADV))
            else:
                pos.append(('',''))
    
        return pos
    
    def lemmatizer(self, words):
        lemmatizer = WordNetLemmatizer() 
        lemmatized_sentence = []
    
        for word in words:
            w,p = self.get_pos([word])[0]
            if p != '':
                w = lemmatizer.lemmatize(word, pos=p)
            else:
                w = lemmatizer.lemmatize(word)
            lemmatized_sentence.append(w)
        
        return lemmatized_sentence
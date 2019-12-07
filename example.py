from preprocess import Preprocess

preprocess = Preprocess()

string = "Hello, World! This is a sample text that I'm using for tesing"
print("Sample Text: ", string)

sentences = preprocess.sentence_tokenizer(string)
print("Sentences: ", sentences)

clean_text = preprocess.clean_text_form(sentences[1])
print("Clean Text 1: ", clean_text)

clean_text2 = preprocess.remove_unchars(clean_text)
print("Clean Text 2: ", clean_text2)

words = preprocess.word_tokenizer(clean_text2)
print("Words: ", words)

no_stop_words = preprocess.remove_stop_wrods(words)
print("Stop Words Removal: ", no_stop_words)

pos = preprocess.get_pos(words)
print("POS Tags: ", pos)

lemmas = preprocess.lemmatizer(words)
print("Lemmatization: ", lemmas)
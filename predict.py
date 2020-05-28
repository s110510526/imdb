from keras.models import load_model
import numpy as np
import nltk
from keras.datasets import imdb
from nltk import word_tokenize
import string
from keras.preprocessing import sequence

#載入模型
model = load_model('imdb_modelb.h5')

#轉換字串
def Preparing_string(text_string, dimension = 10000):
        text_string = text_string.lower()
        table = str.maketrans(dict.fromkeys(string.punctuation))
        text_string = text_string.translate(table)
        word2index = imdb.get_word_index()
        test=[]
        for word in word_tokenize(text_string):
            test.append(word2index[word])

        results = np.zeros(dimension)
        for _ , sequence in enumerate(test):
            if sequence < dimension:
                results[sequence] = 1
        print("\nOriginal string:", text_string,"\n")
        print("\nIndex conversion:", test,"\n")
        results = np.reshape(results,(1, 10000))
        return results


#印出預測結果
text = Preparing_string("First off, this is NOT a war film. It is a movie about the bond of men in war. It is by far the best movie I've seen in a very, very long time. I had high expectations and was not disappointed. At first I was eager to see the one shot idea Sam Mendes went into this with but, after awhile, I stopped paying attention to that. While everything about the movie was well done I was so caught up in the two central characters that nothing else mattered. I will watch this again and again.")
print(model.predict(text))
print ('預測結果', model.predict_classes(text))
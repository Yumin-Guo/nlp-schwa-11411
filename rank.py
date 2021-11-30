

import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pre_process import *
import spacy
nlp = spacy.load("en_core_web_sm")


# need to adjust the score with respect to the question length 
# need to test the scale of the correctness score and the importance of the question

def compare_correct(q_list):
	for i,(q,a,score) in enumerate(q_list):
		attempt_ans=generate_answer(q)
		doc1=nlp(a)
		doc2=nlp(attempt_ans)
		similarity=doc1.similarity(doc2)

		q_list[i][2]+=similarity

	q_list.sort(key=lambda x: x[2])
	return q_list

def rank(questions,X,word_list):
	out=[]
	for (q,a) in questions:
		words=q.split()
		acc=0
		for word in words:
			index=np.where(word_list==item)
			if len(index[0])!=0:
				acc+=word_list[index][0]
		out.append((q,a,acc))
	out.sort(key=lambda x: x[2])
	return out

def significant_words(path):

	sentences=make_sentence(path1)
	corpus=["".join(sentences)]
	print(corpus)
	vectorizer = TfidfVectorizer(stop_words="english")
	X = vectorizer.fit_transform(corpus)
	name=vectorizer.get_feature_names_out()
	print(type(name))
	return name,X

#significant_words("../nlp_proj/hp.txt")


import spacy
import stanza
import re 
import nltk

from nlpp import *


path1="../nlp_proj/hp.txt"

def make_sentence(path):
	out1=[]
	with open(path) as f:
		line = f.readline()
		while line:
			line = f.readline()
			out1.append(line)
	pop_list=[]
	for i,sentence in enumerate(out1):
		if sentence!="":
			if sentence=="\n":
				pop_list.append(i)
			else:
				if sentence[-1]=="\n":
					out1[i]=sentence[:-1]
		else:
			pop_list.append(i)
	offset=0
	for x in pop_list:
		out1.pop(x-offset)
		offset+=1
	real_out=[]
	for sentence in out1:
		temp=sentence.split(".")
		real_out=real_out+temp


	for (i,sent) in enumerate(real_out):
		temp=sent.split(" ")
		
		if len(temp)==2:
			real_out[i]=".".join(real_out[i-1:i+1])
			real_out[i-1]=""
	return real_out


def make_wh_questions(sentences,acc):
	out=[]
	
	for i,sentence in enumerate(sentences): 
		print(i)
		print(acc)
		out=out+make_wh_question(sentence)
		acc-=1
		if(acc==0):
			return out

	

	

if __name__ == '__main__':
	
	path1="../nlp_proj/hp.txt"
	sentences=make_sentence(path1)
	out1=make_wh_questions(sentences,10)
	
	path1="../nlp_proj/chinese.txt"
	sentences=make_sentence(path1)
	out1=out1+make_wh_questions(sentences[30:90],14)
	print(out1)














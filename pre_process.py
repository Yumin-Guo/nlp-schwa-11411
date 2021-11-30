import spacy
import stanza
import re 
import stanfordnlp

from nlpp import *


not_used_deprel=['root','acl',"appos", "advcl", "cc", "ccomp", "conj", "dep", "mark","parataxis","ref"]
path1="../nlp_proj/hp.txt"

def make_sentence(path):
	np1=spacy.load("en_core_web_sm")
	chunks=[]
	res=[]
	with open(path) as f:
		line = f.readline()
		while line:
			line = f.readline()
			chunks.append(line)
	for chunk in chunks:
		doc=np1(chunk)
		for sent in doc.sents:
			text=sent.text


			if (text.isspace()==False):

				#adjust the format of "()"
				if ("(" in text):

					text.replace("(",", ")
				if (")" in text):
					right_i=text.index(")")
					if right_i==len(text)-1:
						text.replace(")",".")
					else:
						text.replace(")",", ")
				res.append(text)

	return res


def make_wh_questions(sentences,acc):
	out=[]
	nlp = stanza.Pipeline('en', processors = "tokenize,mwt,pos,lemma,depparse,ner")
	nlp1 = spacy.load("en_core_web_sm")
	nlp2 = stanza.Pipeline(lang='en', processors='tokenize,ner')
	for i,sentence in enumerate(sentences):
		temp=make_wh_question(sentence,nlp,nlp1,nlp2)
		if temp !=[]:
			out=out+temp
			acc-=1
		if(acc==0):
			print(out)
			return out
	print(out)
	return out

	



def find_common(target,words,saved):
	out=[]
	#print("_______________________________________")
	#print("TARGET TEXT")
	#print(target["text"])
	for word in words:
		head_text=str(words[word['head']-1]['text'])
		head_id=words[word['head']-1]["id"]
		if head_text!=word["text"]:
			if head_text==target['text'] and head_id==target["id"]  and word["deprel"] not in not_used_deprel and "acl" not in word["deprel"] :
				out.append((word,word["id"]))

	#base case
	if len(out)==0:
		return saved
	#for i in out:
		#print(i[0]["text"])
	#recursive case
	else:
		saved=saved+out
		for (word,i) in out:

			saved=find_common(word,words,saved)

		return saved
	
	
def simplify_sentences(sentence,nlp):
	#print("olala")
	sentence=sentence.lower()
	doc = nlp(sentence)
	sent_dict = doc.sentences[0].to_dict()

	sent_root=[]
	processed={}
	output=[]
	#print(sent_dict)
	for word in sent_dict:
		#print(word)
		print ("{:<15} | {:<10} | {:<15} ".format(str(word['text']),str(word['deprel']), str(sent_dict[word['head']-1]['text'] if word['head'] > 0 else 'ROOT')))
		deprel=word['deprel']
		if "nsubj" in deprel:
			head=sent_dict[word['head']-1]
			sent_root.append(head)

	for start in sent_root:

		saved=find_common(start,sent_dict,[])
		saved.append((start,start["id"]))
		saved.sort(key=lambda x: x[1])
		simple_sentence=[val[0]["text"] for val in saved]
		#print(simple_sentence)
		output.append(simple_sentence)
	output=adjust_repetitions(output)
	#print(output)
	return output


#punctuation has not adjusted yet
#need to think out of a way to deal with it 
def adjust_repetitions(sents):
	out=[]
	for sent in sents:
		for (i,word) in enumerate(sent):
			if sent[i-1]==word:
				sent[i-1]=""
		val=" ".join(sent)

		out.append(val)

	return out

if __name__ == '__main__':
	
	path1="../nlp_proj/a1.txt"
	sentences=make_sentence(path1)
	nlp = stanza.Pipeline('en', processors = "tokenize,mwt,pos,lemma,depparse,ner") 
	out=make_wh_questions(sentences,10)


	#print(out)
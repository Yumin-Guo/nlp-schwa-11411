from pre_process import *


match={"Past":"did",
       "3":"does",
       "2": "do"}

def delete_clause(sentence,nlp):
	print(sentence)
	clause_words=["because", "caused by", "owing to","due to"]
	clauses=sentence.split(",")
	split_temp=[]
	ever_reason=False
	print(clauses)
	for i,clause in enumerate(clauses):
		is_reason=False
		for word in clause_words:
			if word in clause:
				is_reason=True
				ever_reason=True
				
		if is_reason==False or len(clauses)==1:
			split_temp.append(clause)
	if(ever_reason):
		drop_clause=",".join(split_temp)

		out=simplify_sentences(drop_clause,nlp)
		print(drop_clause)
		print("hello this is out")
		print(out)
		return out[0]
	else:
		return None

def make_why_question(candidate,nlp):
	out=[]
	doc=nlp(candidate)
	words = doc.sentences[0].to_dict()
	root=None
	for word in words:
		if word["deprel"]=="root":
			root=word
			print(root)
			break
	aux=None
	form_words=[]
	for word in words:
		head_text=str(words[word['head']-1]['text'])
		if head_text==root["text"] and word["deprel"]=="aux":
			aux=word
		else:
			form_words.append((word,word["id"]))
	form_words.sort(key=lambda x: x[1])
	
	if aux!=None:
		phrase=[val[0]["text"] for val in form_words]
		val=" ".join(phrase)
		val=change_sequence(val,root)
		q="Why "+aux["text"]+val +"?"
	else:
		phrase=[]
		do_verb=None
		for val in form_words:
			if val[0]==root:
				phrase.append(root["lemma"])
				val=root["feats"].split("|")
				if "Tense=Pres" in val:
					if 'Person=3' in val:
						do_verb=match["3"]
					else:
						do_verb=match["2"]
				else:
					do_verb=match["Past"]

			else:
				phrase.append(val[0]["text"])
		val=" ".join(phrase)
		val=change_sequence(val,root)
		q="Why "+ do_verb+" "+val+"?"
	print(q)
	return q






import spacy
import stanza
import re 
import nltk
from pre_process import * 

match={"Past":"did",
       "3":"does",
       "2": "do"}


'''
Issues:
1. coreference
2. lemmatize do, be, verbs etc
3. Delete adverbs before time phrase 
4. bug for who question


'''



l_geo=["GPE","LOC"]
l_cats=["PERSON","NORP","FAC","ORG","GPE","LOC","PRODUCT","EVENT","WORK_OF_ART","LAW","LANGUAGE","DATE","TIME","PERCENT","MONEY","QUANTITY","ORDINAL","CARDINAL"]
l_time=["DATE","TIME"]
l_number=["PERCENT","MONEY","QUANTITY","ORDINAL","CARDINAL"]
l_person=["PERSON"]
l_objects=["NORP","FAC","ORG","PRODUCT","EVENT","WORK_OF_ART","LAW","LANGUAGE"]


WHO=0
WHEN=1





def make_when_question(found_date,sent,nlp,unit):
    print("this is when question")
    print(sent)
    #print(found_date)
    out=[]
    doc = nlp(sent)
    root=None
    
    words = doc.sentences[0].to_dict()
    for word in words:
        if word["deprel"]=="root":
            root=word
            print(root)
            break
    aux=None
    form_words=[]
    for word in words:
        head_text=str(words[word['head']-1]['text'])
        print("aux" in word["deprel"])
        print(word["deprel"])
        if head_text==root["text"] and "aux" in word["deprel"] and aux==None:
            print("hello")
            aux=word

        else:
            if(word["text"] not in found_date):
                form_words.append((word,word["id"]))
    form_words.sort(key=lambda x: x[1])
    print(aux)
    if aux!=None:
        phrase=[val[0]["text"] for val in form_words]
        q="When "+aux["text"]+ " ".join(phrase)+"?"
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
        q="When "+ do_verb+" "+" ".join(phrase)+"?"
    
     




    return q
def make_how_much_question(found_num, sent,nlp,unit):

    out=[]
    doc = nlp(sent)
    words = doc.sentences[0].to_dict()
    for word in words:
            if word["text"] not in found_num:
                out.append((word,word["id"]))

    out.sort(key=lambda x: x[1])
    phrase=[val[0]["text"] for val in out]
    out=" ".join(phrase)
    if unit=="MONEY":
        out="How much did "+" "+out+"?"
    else:
        out="How many did "+" "+out+"?"
    return out



def make_who_question(found_person,sent,nlp,unit):
    
    print("this is when question")
    print(sent)
    out=[]
    doc = nlp(sent)
    root=None
    aux=None
    verb=None
   
   
    words = doc.sentences[0].to_dict()
    for word in words:
        if word["deprel"]=="root":
            root=word
            #print(root)
            break
    
    form_words=[]
    for word in words:
        head_text=str(words[word['head']-1]['text'])
        if head_text==root["text"] and "aux" in word["deprel"]:
            aux=word
            break
    
    if aux!=None:
        verb=aux["text"]
        for word in words:
            if word !=aux and word["text"] not in found_person:
                form_words.append((word,word["id"]))
    else:
        verb=root["text"]
        for word in words:
            if word!=root and word["text"] not in found_person:
                form_words.append((word,word["id"]))
    form_words.sort(key=lambda x: x[1])
    phrase=[val[0]["text"] for val in form_words]
    q="Who "+verb+" "+" ".join(phrase)+"?"
    

    return q








def find_part(entities,Tree,sentence,nlp,status):
    if status ==WHEN:
        cats=l_time
        func=make_when_question

    elif status==WHO:
        cats=l_person
        func=make_who_question
    elif status==WHERE:

        cats=l_geo
        func=make_where_qeustion
    elif status==WHAT:
        cats=l_objects
        func=make_what_qeustion
    else:
        cats=l_number
        func=make_when_qeustion
    found_entities=-1
    entity_type=None
    for sent in entities.sentences:
        for ent in sent.ents:
            if ent.type in cats :

                found_entities=ent.text
                entity_type=ent.type
    if found_entities !=-1:
        found_entities=found_entities.split()
        Q=func(found_entities,sentence,nlp,entity_type)
        return Q



def check_entity_exist(entities):
    for sent in entities.sentences:
        for ent in sent.ents:
            if ent.type in l_cats:
                return True
    return False

def make_wh_question(sentence,nlp,nlp_model_1,nlp_model_2):
    q_list=[]
    q_types=["who","when"]

    doc_ent= nlp(sentence)
    if check_entity_exist(doc_ent):
        simple_sents=simplify_sentences(sentence,nlp)
        for simple_sent in simple_sents:
            doc_tree=nlp_model_1(simple_sent)
            doc_ent=nlp_model_2(simple_sent)
            for i,q_type in enumerate(q_types):
                Q=find_part(doc_ent,doc_tree,simple_sent,nlp,i)
                if Q!=None:
                    q_list.append(Q)
     
    print(q_list) 

    return q_list

#sent="Unless otherwise specified, Chinese texts in this article are written in (Simplified Chinese/Traditional Chinese; Pinyin) format."
#sent="Harry Potter and the Prisoner of Azkaban is a fantasy film directed by Alfonso Cuarón and distributed by Warner Bros in 2004."
#sent="The film was released on 31 May 2004 in the United Kingdom and on 4 June 2004 in North America, as the first Harry Potter film released into IMAX theatres and to be using IMAX Technology."
#nlp_model_2 = stanza.Pipeline(lang='en', processors='tokenize,ner')
#nlp = stanza.Pipeline('en', processors = "tokenize,mwt,pos,lemma,depparse,ner") 
#path1="../nlp_proj/chinese.txt"
#entences=make_sentence(path1)
#for sent in sentences:
    #doc_ent= nlp_model_2(sent)
    #if check_entity_exist(doc_ent,"DATE"):
        #print(sent)
        #simplify_sentences(sent,nlp)




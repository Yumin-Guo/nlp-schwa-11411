import spacy
import stanza
import re 
import nltk


# repetitive codes a lot, but easy for make adjustment according
# to the question type 
def closest_n(text,word):
    if word in text:
        return text.index(word)
    else:
        champdiff=100
        champi=-1
        for i,x in enumerate(text):
            if word in x:
                if len(x)-len(word)<champdiff:
                    champdiff=len(x)-len(word)
                    champi=i
        return champi


def find_belong_date(doc,found_date):
    past_tense=False
    is_be=False
    
    for token in doc:
        childrens=[str(child) for child in token.children]
        for i ,date in enumerate(found_date):
            if date in childrens and token.text not in found_date:
                upper=token
                break
    print(upper)
    if upper==None:
        return None,None,None,None
    
    out=found_date+[upper.text]
    current_token=upper
    root=None
    for token in doc:
        if token.dep_=="ROOT":
            root=token
 
    if root==None:
        return None,None,None,None

    if (root.lemma_=="be"):
        is_be=True
    out=out+[root.text]
    print(out)

   
    if (current_token.lemma_=="be"):
        is_be=True

    return is_be,root.tag_,root.lemma_,out

def find_belong_person(doc,found_person):
    is_be=False
    root=None 
    for token in doc:
        if token.dep_=="ROOT":
            root=token

    if root==None:
        return None,None,None 
    if (root.lemma_=="be"):
        is_be=True
    out=found_person+[root.text]
    return is_be,root.tag_,out


def make_when_question(is_be,is_past,out,lemma,doc,sentence):
    text=re.findall(r"[\w']+|[.,!?;]", sentence)
    length=len(text)
    
    start_i=min(closest_n(text,out[0]),closest_n(text,out[-2]))
    end_i=max(closest_n(text,out[0]),closest_n(text,out[-2]))
    text=text[0:start_i]+text[end_i+1:]
    root_i=closest_n(text,out[-1])

    if is_be:
        be=text[root_i]+" "
        text[root_i]=""
        ques="When "+be+ " ".join(text)+" ?"
        print(ques)
        return ques
    else:
        if is_past=="VBD":
            do_verb="did "
        elif is_past=="VBZ":
            do_verb="does "
        else:
            do_verb="do "
        text[root_i]=lemma
        ques="When "+do_verb+ " ".join(text)+" ?"
        print(ques)
        return ques



def make_who_question(is_be,is_past,out,doc,sentence):
    print(out)
    text=re.findall(r"[\w']+|[.,!?;]", sentence)
    
    length=len(text)
    start_i=min(closest_n(text,out[0]),closest_n(text,out[-2]))
    end_i=max(closest_n(text,out[0]),closest_n(text,out[-2]))
    print(start_i)
    print(end_i)
    text=text[0:start_i]+text[end_i+1:]
    root_i=closest_n(text,out[-1])

    if is_be:
        be=text[root_i]+" "
        text[root_i]=""
        ques="Who "+be+ " ".join(text)+" ?"
        print(ques)
        return ques
    else:
        if is_past=="VBD":
            do_verb="did "
        elif is_past=="VBZ":
            do_verb="does "
        else:
            do_verb="do "
        verb=text[root_i]
        text[root_i]=""
        ques="Who "+verb+ " ".join(text)+" ?"
        print(ques)
        return ques







def when_question(entities,Tree):
    found_date=-1
    for sent in entities.sentences:
        for ent in sent.ents:
            if ent.type=="DATE":
                found_date=ent.text
    if found_date !=-1:
        found_date=found_date.split()
        print(found_date)
        
        return find_belong_date(Tree,found_date)
    else:
        return None,None,None,None


def who_question(entities,Tree):
    found_person=-1
    for sent in entities.sentences:
        for ent in sent.ents:
          
            if ent.type=="PERSON":
                found_person=ent.text
    if found_person !=-1:
        found_person=found_person.split()
        print(found_person)
        return find_belong_person(Tree,found_person)
    else:
        return None,None,None

def where_question(entities,Tree):
    found_org=-1
    for sent in entities.sentences:
        for ent in sent.ents:
            if ent.type=="ORG":
                found_person=ent.text

    found_org=found_org.split()

    return find_belong_person(Tree,found_person)




def make_wh_question(sentence):
    q_list=[]
    nlp_model_1 = spacy.load("en_core_web_sm")
    nlp_model_2 = stanza.Pipeline(lang='en', processors='tokenize,ner')
    doc_ent = nlp_model_2(sentence)
    doc_tree= nlp_model_1(sentence)

    is_be,is_past,out=who_question(doc_ent,doc_tree)
    if is_be!=None:
        q_list.append(make_who_question(is_be,is_past,out,doc_tree,sentence))
    is_be,is_past,lemma,out=when_question(doc_ent,doc_tree)
    if is_be!=None:
        q_list.append(make_when_question(is_be,is_past,out,lemma,doc_tree,sentence))

    return q_list





    
    
#!/usr/bin/python3 -W ignore::DeprecationWarning
# -*- coding:utf8 -*-

# all necessary packages
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words -= {'not'}
from nltk.corpus import wordnet as wn
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()

import spacy
nlp = spacy.load('en_core_web_sm') 

import re
import heapq
import io
import sys

# setup for finding most similar sentence
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('nq-distilbert-base-v1')

# setup for question answering system
from transformers import BertForQuestionAnswering, AutoTokenizer
modelname = 'deepset/bert-base-cased-squad2'
model_b = BertForQuestionAnswering.from_pretrained(modelname)
tokenizer = AutoTokenizer.from_pretrained(modelname)
from transformers import pipeline
nlp_b = pipeline('question-answering', model=model_b, tokenizer=tokenizer)

#################################################
# The following section is about binary answers #
#################################################
def textToComponents(text):
    doc = nlp(text)
    subject_found, aux_found = False, False
    subject, rest = None, None
    prior = []
  
    for i, token in enumerate(doc):

        if subject_found and aux_found: 
            return [t.text for t in prior], [t.text for t in subject], [t.text for t in rest]
        if not aux_found and (token.dep_ in {'aux', 'auxpass'} or token.pos_ == 'AUX'):
            aux_found = True
            rest = [t for t in doc[i+1:-1]]
        elif not subject_found and token.dep_ in {'nsubj', 'nsubjpass'}:
            subject_found = True
            subject = [t for t in token.subtree]
            prior = [t for t in prior if t not in token.subtree]
        elif not subject_found and token.pos_ != 'PUNCT':
            prior.append(token)
  
    # not found: no auxilary verb
    if subject_found:
        for i, token in enumerate(doc):
            if token not in prior and token not in subject:
                rest = doc[i:-1]
                return [t.text for t in prior], [t.text for t in subject], [t.text for t in rest]

def qToSentense(q, docT):
    # Given a question and the processed declarative sentence, return the context, subject, rest of the sentence in the question
    doc = nlp(q)
    subject_found, aux_found = False, False
    prior_sentence = []
    subject = []
    rest_sentence = []
    aux = None

    skipTo = len(doc)
    for i, token in enumerate(doc):
        if not aux_found and (token.dep_ in {'aux', 'auxpass'} or token.pos_ == 'AUX'):
            aux_found = True
            aux = token
        if not subject_found and token.dep_ in {'nsubj', 'nsubjpass'}:
            subject_found = True
            subject = [t for t in token.subtree]
            prior_sentence = [w for w in prior_sentence if w not in subject]

            skipTo = i + len(subject) - subject.index(token)
            if skipTo >= len(doc)-1:
                break # parsing is wrong
            rest_sentence = doc[skipTo:-1]
            return [t.text for t in prior_sentence], [t.text for t in subject], [t.text for t in rest_sentence]
        if not aux_found and token.pos_ != 'PUNCT':
            prior_sentence.append(token)
  
    # The case where nothing is found
    possibleSubj = " ".join(docT[1])
    matching_subj = re.findall(r'\b{phrase}\b'.format(phrase=possibleSubj), q, re.IGNORECASE)
    if len(matching_subj) > 0:
        fixedSubj = nltk.word_tokenize(matching_subj[0])
    else:
        return None, None, None
    for i, token in enumerate(doc):
        if token not in prior_sentence and token != aux and token.pos_ != 'PUNCT' and token.text not in fixedSubj:
            return [t.text for t in prior_sentence], fixedSubj, [t.text for t in doc[i:-1]]
    return None, None, None

def isHypernym(largeW, smallW):
    large = set(wn.synsets(largeW))
    small = set(wn.synsets(smallW))
    if large and small:
        hypers = set()
        for subsmall in small:
            hypers = hypers.union(set([i for i in subsmall.closure(lambda s:s.hypernyms())]))
        for sublarge in large:
            if sublarge in hypers:
                return True
    return False

def isSynonym(w1, w2):
    first = set(wn.synsets(w1))
    second = set(wn.synsets(w2))
    return len(first.intersection(second)) > 0

def removeStopWords(wordL):
    return [word.lower() for word in wordL if word.lower() not in stop_words]

def subjectMatch(subjQ, subjT):
    subjQ, subjT = removeStopWords(subjQ), removeStopWords(subjT)
    notQ, notT = ("not" in subjQ), ("not" in subjT)
    if notQ != notT:
        return False
    subjQ, subjT = " ".join(subjQ), " ".join(subjT)
    return re.search(r'\b{phrase}\b'.format(phrase=subjQ), subjT) != None

def contextMatch(contextQ, contextT):
    contextQ, contextT = removeStopWords(contextQ), removeStopWords(contextT)
    contextQ, contextT = " ".join(contextQ), " ".join(contextT)
    return not contextQ or re.search(
        r'\b{phrase}\b'.format(phrase=contextQ), contextT) != None

def restMatch(restQ, restT):
    restQ, restT = removeStopWords(restQ), removeStopWords(restT)
    notQ, notT = ("not" in restQ), ("not" in restT)
    if notQ != notT:
        return False
    for qw in restQ:
        if qw == "?":
            continue
        found = False
        if qw in restT:
            found = True
        else:
            for tw in restT:
                if isHypernym(qw, tw) or isSynonym(qw, tw):
                    found = True
                    break
        if not found: 
            return False
    return True

def splitOr(q):
    words = nltk.word_tokenize(q)
    if "or" in words: # must be true to call this function
        index = words.index("or")
        q1 = words[:index] + words[index+2:-1]
        q2 = words[:index-1] + words[index+1:-1]
        return " ".join(q1)+"?", " ".join(q2)+"?", words[index-1], words[index+1]
    else:
        raise Exception("Question should contain 'or' to make this function valid")

def answerOr(q, sentence):
    q1, q2, keyword1, keyword2 = splitOr(q)
    if answerBinary(q1, sentence): 
        return keyword1[0].upper() + keyword1[1:] + "."
    return keyword2[0].upper() + keyword2[1:] + "."

def fineTuneS(q, sentence):
    # find the part of the sentence most relevant to the question only
    q_words = nltk.word_tokenize(q.lower())
    useful_q_words = set(q_words) - stop_words
    match_l, all_l = 0, 0
    sent_all, sent_curr = [], []
    for (i, token) in enumerate(nltk.word_tokenize(sentence)):
        if token in ',:";.?\n':
            if all_l == 0 or match_l / all_l >= 0.6 or match_l / len(useful_q_words) >= 0.75:
                sent_all.append(" ".join(sent_curr))
            sent_curr, match_l, all_l = [], 0, 0
        else:
            sent_curr.append(token)
            if token not in stop_words:
                # print(token)
                all_l += 1
                if token.lower() in q_words:
                    match_l += 1
    if sent_curr and (all_l == 0 or match_l / all_l >= 0.6):
        sent_all.append(" ".join(sent_curr) + token)
    new_sent = " ".join(sent_all)
    if not new_sent:
        return sentence
    return new_sent + "."

def answerBinary(q, sentence):
    tuned_sentence = fineTuneS(q, sentence)
    print("\t",tuned_sentence)
    words = nltk.word_tokenize(q)
    if "or" in words and ("or" not in nltk.word_tokenize(tuned_sentence)):
        return answerOr(q, tuned_sentence)
    else:
        docT = textToComponents(tuned_sentence)
        if not docT:
            docT = textToComponents(sentence)
        print(docT)
        docQ = qToSentense(q, docT)
        if docQ[1] and docQ[2] and subjectMatch(docQ[1], docT[1]) and contextMatch(docQ[0], docT[0]) and restMatch(docQ[2], docT[2] + docT[0]):
            return "Yes."
        return "No."

#############################################
# The following section is about wh answers #
#############################################

# topic recognition 
def movie(text):
    # assume text given is only the first sentence
    return re.search("film", text, re.IGNORECASE) != None

def football(text):
    return re.search("soccer player|footballer", text, re.IGNORECASE) != None

def constellation(text):
    return re.search("constellation", text, re.IGNORECASE) != None

def language(text):
    return re.search("language", text, re.IGNORECASE) != None

def is_wh_question(question):
    doc = nlp(question)
    token = doc[0]
    if token.tag_ in {"WRB", "WP"}:
        return True, token.text
    return False, None

def handle_1st(context, question):
    # find in 1st sentence
    first_sent = context.split("\n\n\n")[1].split(".")[0]
    ans = nlp_b({
        "question": question,
        "context": first_sent
    })
    return ans["answer"]

def handle_intro(context, question):
    intro = context.split("\n\n\n")[1]
    ans = nlp_b({
        "question": question,
        "context": intro
    })
    return ans["answer"]

def findSubSent(keywords, sentence):
    subs = re.split('[,();:] ', sentence)
    for sub in subs:
        for k in keywords:
            if re.search(r'\b{pattern}\b'.format(pattern=k), sub, re.IGNORECASE) != None:
                return True
    return False

def ansWhy(sentence):
    # weigh the sentence based on it's relevance to question
    keywordS0 = {'because', 'due to', 'owing to', 'as a consequence of', 'resulted from'}
    keywordS1 = {'for', 'since', 'caused by', 'as'}
    if findSubSent(keywordS0, sentence):
        return 1.5
    if findSubSent(keywordS1, sentence):
        return 1
    return 0.8

def handle(context, question, sect_name):
    temp = context.split("\n\n\n")
    sect = None
    for i, p in enumerate(temp):
        if p.startswith(sect_name):
            sect = temp[i+1]
            break
    if not sect:
        temp = context.split("\n")
        for i, p in enumerate(temp):
            if p.startswith(sect_name):
                sect = temp[i+1]
                break
    if not sect:
        return None
    ans = nlp_b({
        "question": question,
        "context": sect
    })
    return ans["score"], ans["answer"]

film_keywords_1st_sent = set(["genre", "country", "time"])
film_1st_sent_handler = (film_keywords_1st_sent, handle_1st, None)
film_keywords_intro = set(["written", "directed", "edited", "produced", "box office", \
                            "review", "acclaimed", "praised", "criticized", "nominate", "award", "success"])
film_intro_handler = (film_keywords_intro, handle_intro, None)
film_handlers = [film_1st_sent_handler, film_intro_handler]

football_keywords_1st_sent = set(["who", "born", "where"])
football_1st_sent_handler = (football_keywords_1st_sent, handle_1st, None)
football_keywords_sec1 = set(["childhood", "family", "parents", "recruited", "accepted", \
                              "grow up"])
football_sec1_handler = (football_keywords_sec1, handle, "Early life")
football_keywords_sec2 = set(["honors", "awards"])
football_sec2_handler = (football_keywords_sec2, handle, "Honours")
football_handlers = [football_1st_sent_handler, football_sec1_handler, football_sec2_handler]

cons_keywords_intro = set(["located", "where", "name"]) 
cons_intro_handler = (cons_keywords_intro, handle_intro, None)
cons_keywords_sec1 = set(["stars"])
cons_sec1_handler = (cons_keywords_sec1, handle, "Stars")
cons_keywords_sec2 = set(["galaxy", "cluster", "nebula"])
cons_sec2_handler = (cons_keywords_sec2, handle, "Deep-sky objects")
cons_handlers = [cons_intro_handler, cons_sec1_handler, cons_sec2_handler]

lang_keywords_sec1 = set(["earliest", "develop", "transform", "simplify", "rise", \
                          "spread", "evolve", "change", "retain", "rename", "acquire", \
                          "abtain"])
lang_sec1_handler = (cons_keywords_sec1, handle, "History")
lang_keywords_sec2 = set(["grammatical", "noun", "adjective", "pronouns", "determiner", \
                          "verb", "syllable", "morpheme", "word classes", "syntax"])
lang_sec2_handler = (cons_keywords_sec2, handle, "Grammar")
lang_keywords_sec3 = set(["phonetic", "dialect", "pronunciation", "phoneme", "consonant", \
                          "tone", "vowel", "sound", "nasal", "rhythm", "intonation"])
lang_sec3_handler = (lang_keywords_sec3, handle, "Phonology")
lang_handlers = [lang_sec1_handler, lang_sec2_handler, lang_sec3_handler]

def hardcode(context, question, handlers):
    for keywords, handler, param in handlers:
        for keyword in keywords:
            if st.stem(keyword) in question:
                ans = handler(context, question, param)
                return ans
    return None # answer can't be found through hardcode

def check1(context, question):
    ans = None
    if not is_wh_question(question)[0]:
        return None
    if movie(context):
        ans = hardcode(context, question, film_handlers)
    elif football(context):
        ans = hardcode(context, question, football_handlers)
    elif constellation(context):
        ans = hardcode(context, question, cons_handlers)
    elif language(context):
        ans = hardcode(context, question, lang_handlers)
    
    if ans:
        return ans
    return None

def weigh(raw, keyword, sentence):
    weight = 1
    if keyword.lower() == "why":
        weight = ansWhy(sentence)
    return raw * weight

def settleWhy(answer):
    keywords = {'because', 'due to', 'owing to', 'as a consequence of', 'resulted from', 'so', 'for', 'since', 'caused by', 'as'}
    for keyword in keywords:
        if re.search(r'^{pattern}\b'.format(pattern=keyword), answer, re.IGNORECASE) != None:
            return answer
    doc = nlp(answer)
    for token in doc:
        if token.dep_ in {'nsubj', 'nsubjpass'}:
            return "because " + answer
    return "because of " + answer

def settleAns(answer, keyword):
    if keyword == "why":
        answer = settleWhy(answer)
    return answer[0].upper() + answer[1:] + "."

def check2(text, questions, answers):
    q_embeddings = model.encode(questions)
    sentences = []
    for sent in nltk.sent_tokenize(text):
        if sent != "\n":
            sentences.append(sent)
    txt_embeddings = model.encode(sentences)

    similarities = util.pytorch_cos_sim(q_embeddings, txt_embeddings)

    # from all candidate sentences, find the ones closer to the query
    savings = [] # keep the top sentences
    for idx in range(len(questions)):
        question = questions[idx]

        check_wh = is_wh_question(question)

        if check_wh[0]:
            keyword = check_wh[1].lower()
            msize = 8
        else:
            print("\tNOT WH :", question)
            keyword = ""
            msize = 1 # assume that binary question should be very similar

        size = 0
        pq = []
        heapq.heapify(pq)

        for (i, res) in enumerate(similarities[idx]):
            score = weigh(res.item(), keyword, sentences[i])
            heapq.heappush(pq, (score, sentences[i]))
            size += 1
            if size > msize:
                heapq.heappop(pq)
                size -= 1
          
        savings = pq.copy()

        if not is_wh_question(question): # binary question
            similar_sent = savings[0][1]
            ans = answerBinary(question, similar_sent)
            answers[idx] = ans
        else:
            if max(savings)[0] > 1:
                # we only care about this sentence
                ans = nlp_b({
                    "question": question,
                    "context": max(savings)[1]
                })
                answers[idx] = settleAns(ans['answer'], keyword)
                print(answers[idx])
                continue
            similar_sents = []
            for i in range(len(savings)):
                similar_sents.append(savings[i][1])
            
            highest_score = 0
            best_ans = ""
            for i, similar_s in enumerate(similar_sents):
                ans = nlp_b({
                    "question": question,
                    "context": similar_s
                })
                # print(ans)
                weightedScore = ans['score'] * float(savings[i][0])**2
                if weightedScore > highest_score:
                #   print(question, ans)
                    highest_score = weightedScore
                    best_ans = ans['answer']

            # print()
            if answers[idx]:
                print("\n\tOLD", answers[idx])
                old_score, old_ans = answers[idx]
                chosen = old_ans if old_score * 1.2 > highest_score else best_ans
                answers[idx] = settleAns(chosen, keyword)
            else:
                answers[idx] = settleAns(best_ans, keyword)
            print(best_ans)
    return answers

def main():
    output = open("../output.txt", 'w')
    old_target, sys.stdout, sys.stderr = sys.stdout, output, output

    if len(sys.argv) != 3:
        print("Usage: ./answer article.txt questions.txt")
        sys.exit(1)

    txt_file = sys.argv[1]
    q_file = sys.argv[2]

    with io.open(txt_file, 'r', encoding='utf8') as f:
        txt_text = f.read()

    with io.open(q_file, 'r', encoding='utf8') as f:
        questions = f.readlines()

    answers = [None for _ in range(len(questions))]

    for (i, question) in enumerate(questions):
        # print(question)
        ans = check1(txt_text, question)
        if ans:
            answers[i] = ans
    
    answers = check2(txt_text, questions, answers)

    sys.stdout = old_target
    for answer in answers:
        print(answer)


if __name__ == '__main__':
    main()
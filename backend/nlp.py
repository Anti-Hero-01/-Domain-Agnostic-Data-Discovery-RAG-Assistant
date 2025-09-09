# backend/nlp.py
import spacy
# Ensure you've installed en_core_web_sm: python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')

def extract_entities(text):
    doc = nlp(text)
    ents = []
    for e in doc.ents:
        ents.append({'text': e.text, 'label': e.label_})
    return ents

def extract_svo(doc):
    # doc is a spaCy Doc
    svos = []
    for sent in doc.sents:
        root = sent.root
        subj = None
        obj = None
        for tok in sent:
            if 'subj' in tok.dep_:
                subj = tok
            if 'obj' in tok.dep_:
                obj = tok
        if subj and obj:
            svos.append((subj.text, root.lemma_, obj.text))
    return svos

def extract_entities_and_triples(text):
    doc = nlp(text)
    entities = extract_entities(text)
    triples = extract_svo(doc)
    # normalize to consistent triple dicts
    triple_dicts = [{'subj': s, 'pred': p, 'obj': o} for s,p,o in triples]
    return {'entities': entities, 'triples': triple_dicts}

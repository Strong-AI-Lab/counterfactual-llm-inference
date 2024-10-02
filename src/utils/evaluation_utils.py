
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
import re
import openai

from typing import Tuple, List

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

def sem_equals(a : str, b :str) -> bool:
    positives = ['yes', 'true', '1', 'y', 't']
    negatives = ['no', 'false', '0', 'n', 'f']

    a_pos = a.lower() in positives
    b_pos = b.lower() in positives

    a_neg = a.lower() in negatives
    b_neg = b.lower() in negatives

    if not (a_pos or a_neg) or not (b_pos or b_neg):
        return a == b
    else:
        return (a_pos and b_pos) or (a_neg and b_neg)



def validate_with_llm(sentence : str, value : str) -> bool:
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
                {"role": "system", "content": f"Here is a sentence: '{sentence}'. Does the following value/statement '{value}' states the sentence as true, false or neutral/off-topic? Answer with a single word: 'yes', 'no', or 'neutral'."}
            ],
        max_tokens=5
    )
    t = re.search(r'yes', response.choices[0].message.content, re.IGNORECASE) is not None
    f = re.search(r'no', response.choices[0].message.content, re.IGNORECASE) is not None

    if t:
        return True
    elif f:
        return False
    else:
        raise ValueError(f"Could not validate sentence: '{sentence}' with value: '{value}'. LLM response: '{response.choices[0].message.content}'")
    

COMMON_SYNONYMS = {
    'dead' : ['die'],
    'high' : ['increas'],
}

COMMON_ANTONYMS = {
    'surviv' : ['die'],
    'aliv' : ['die'],
    'black' : ['blonde'],
    'dark' : ['bright'],
    'bright' : ['dark'],
    'low' : ['increas'],
    'contract' : ['avoid'],
}

def get_syn_anto(word : str) -> Tuple[List[str], List[str]]:
    synonyms = []
    antonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
            if lemma.antonyms():
                for antonym in lemma.antonyms():
                    antonyms.append(antonym.name())

    return synonyms, antonyms

def validate_sentence(sentence : str, value : str, allow_llm_parsing : bool = False) -> bool:
    if len(value) == 0:
        raise ValueError(f"Value is empty for sentence: '{sentence}'")

    stemmer = PorterStemmer()

    # Tokenize sentence and keep only object of sentence
    tokens = word_tokenize(sentence)
    tags = pos_tag(tokens)
    stem_tokens = [stemmer.stem(token) for token in tokens]
    verb_list = list(filter(lambda x: x[1][1].startswith('VB'), enumerate(tags)))
    if len(verb_list) >= 1:
        verb_idx = verb_list[0][0]
        post_verb_tokens = stem_tokens[verb_idx:]
    else:
        post_verb_tokens = stem_tokens
    
    # Assess if sentence is negative
    negation = False
    for i, token in enumerate(post_verb_tokens):
        if token.lower() in ['not', 'no', 'non', 'n\'t']:
            negation = True
            post_verb_tokens = post_verb_tokens[:i] + post_verb_tokens[i+1:]
            break

    # Tokenize value and assess negation
    value = value.replace('-', ' ').replace('_', ' ')
    value_tokens = word_tokenize(value)
    for i, token in enumerate(value_tokens):
        if token.lower() in ['not', 'no', 'non', 'n\'t']:
            negation = not negation # flip the negation
            value_tokens = value_tokens[:i] + value_tokens[i+1:]
            break
    # stem_value = stemmer.stem(" ".join(value_tokens))
    stem_value_tokens = [stemmer.stem(token) for token in value_tokens]

    if " ".join(stem_value_tokens) in " ".join(post_verb_tokens):
        return not negation
    
    elif " ".join(stem_value_tokens) in ["present"]: # Handle generic positive answer
        return not negation
    
    elif " ".join(stem_value_tokens) in ["absent"]: # Handle generic negative answer
        return negation
    
    elif len(value_tokens) == 1: # Hanlde synonyms and antonyms
        synonyms, antonyms = get_syn_anto(value_tokens[0])

        # Add common synonyms and antonyms not hamdled by nltk
        if stem_value_tokens[0] in COMMON_SYNONYMS:
            synonyms += COMMON_SYNONYMS[stem_value_tokens[0]]

        if stem_value_tokens[0] in COMMON_ANTONYMS:
            antonyms += COMMON_ANTONYMS[stem_value_tokens[0]]

        for syn in synonyms:
            if stemmer.stem(syn) in post_verb_tokens:
                return not negation
            
        for ant in antonyms:
            if stemmer.stem(ant) in post_verb_tokens:
                return negation
        
    if allow_llm_parsing:
        return validate_with_llm(sentence, value)

    # If no match, raise error
    raise ValueError(f"Could not validate sentence: '{sentence}' with value: '{value}'")

    
from __future__ import print_function
from datetime import datetime
from lxml import etree
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import MWETokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from pprint import pprint
import codecs
import collections
import json
import matplotlib.pyplot as plt
import nltk
import nltk.util
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import random
import os
import pickle

tokenizer = MWETokenizer()



nltk.download('words')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('averaged_perceptron_tagger')


def _hasclass(context, *cls):
    """ Checks if the context node has all the classes passed as arguments
    """
    node_classes = set(context.context_node.attrib.get('class', '').split())
    return node_classes.issuperset(cls)

xpath_utils = etree.FunctionNamespace(None)
xpath_utils['hasclass'] = _hasclass

#Regex to remove html tags and excess whitespace from professor descriptions
RE_MARKUP = re.compile(r'<[\w\W]*?>', flags=re.IGNORECASE | re.MULTILINE)
RE_EXCESS_WHITESPACE = re.compile(r'\s+')

parser = etree.HTMLParser()

def parse_person_data(persons):
    """Parse person research description and keywords
    into dictionary for further use"""
    with open(persons) as fobj:
        root = etree.parse(fobj, parser).getroot()

    divs = root.xpath("//div[@class='rendering rendering_person rendering_long rendering_person_long']")
    #Create a data dictionary to add name, research interest description, keywords
    person_data = []
    for div in divs:
        textblock_list = div.xpath("div[@class='textblock']")
        if not textblock_list:
            desc = None
        else:
            desc = etree.tostring(textblock_list[0], encoding="unicode")
        if desc is not None:
            desc = RE_MARKUP.sub(" ", desc)
            desc = RE_EXCESS_WHITESPACE.sub(" ", desc).strip()
        keyword_spans = div.xpath("div/span[text() = 'Keywords: ']")
        keywords = []

        if keyword_spans:
            for span in keyword_spans:
                parent = span.getparent()
                for child in parent.getchildren():
                    for s in child.xpath(".//span"):
                        if s.text:
                            keywords.append(s.text)
                            tokenizer.add_mwe((s.text.split()))
        else:
            keywords = None

        person_data.append({
            "name" : div.xpath("h2/span[@class='person']")[0].text,
            "description" : desc,
            "keywords" : keywords,
        })

    return person_data

elec_persons = parse_person_data("ELEC_persons.html")
sci_persons = parse_person_data("SCI_persons.html")

pattern = r'^[A-Ö]'

# Parse names into sets with (lastname, first initial) since some data identifies person that way
def parse_persons(elec_persons):
    initials_parsed = set()
    name_parsed = set()
    for person in elec_persons:
        initials_parsed.add(str(person["name"].split()[1] + ", " + re.match(pattern, person["name"].split()[0]).group(0) + "."))
        name_parsed.add(person["name"].split()[1] + ", " + person["name"].split()[0])

    return elec_persons, initials_parsed, name_parsed

elec_persons, initials_parsed, name_parsed = parse_persons(elec_persons)
sci_persons, sci_initials, sci_name_parsed = parse_persons(sci_persons)


# Parse research outputs to extract: title, description, associated professor
def parse_research_outputs(research_output, initials_parsed):
    with open(research_output) as fobj:
        root = etree.parse(fobj, parser).getroot()

    research_divs = root.xpath("//div[hasclass('rendering', 'rendering_researchoutput')]")

    research_data = []
    for div in research_divs:
        date_span = div.xpath("div[@class='properties']/div/span[text() = 'Publication date: ']")
        date = 0
        for span in date_span:
            parent = span.getparent()
            for child in parent.getchildren():
                for d_date in child.xpath(".//span"):
                    if re.search(r'\d{4}', d_date.text):
                        date = int(re.search(r'\d{4}', d_date.text).group())


        textblock_list = div.xpath("div[@class='textblock']")
        if not textblock_list:
            desc = None
        else:
            desc = etree.tostring(textblock_list[0], encoding="unicode")

        if desc is not None:
            desc = RE_MARKUP.sub(" ", desc)
            desc = RE_EXCESS_WHITESPACE.sub(" ", desc).strip()

            keyword_spans = div.xpath("div/div/div/span[text() = 'Keywords: ']")
            keywords = ""
            if keyword_spans:
                for span in keyword_spans:
                    parent = span.getparent()
                    for child in parent.getchildren():
                        for s in child.xpath(".//span"):
                            if s.text:
                                keywords += (s.text + ", ")
                                tokenizer.add_mwe((s.text.split()))
            else:
                keywords = None

            contributor_spans = div.xpath("div[@class='properties']/div/span[text() = 'Contributors: ']")
            contributors = []
            for span in contributor_spans:
                parent = span.getparent()
                for child in parent.getchildren():
                    for s in child.xpath(".//span"):
                        if s.text in initials_parsed:
                            contributors.append(s.text)

            for span in div.xpath("div[@class='properties']/div/span[text() = 'Corresponding author: ']"):
                parent = span.getparent()
                for child in parent.getchildren():
                    for s in child.xpath(".//div"):
                        if s.text in initials_parsed:
                            contributors.append(s.text)

            title_text = div.xpath("h2[@class='title']/span")
            if not title_text:
                title = None
            else:
                title = etree.tostring(title_text[0], encoding="unicode")

            research_data.append({
                "title" : title,
                "description" : desc,
                "contributor" : contributors,
                "output_keywords" : keywords
            })
        else:
            continue
    return research_data

research_outputs_data = parse_research_outputs("ELEC_outputs.html", initials_parsed)

sci_outputs = parse_research_outputs("SCI_outputs.html", sci_initials)

#Parse prizes and get: title, description, keywords
def parse_prizes_data(prizes, name_parsed):
    with open(prizes) as fobj:
        root = etree.parse(fobj, parser).getroot()

    prizes_divs = root.xpath("//div[hasclass('rendering', 'rendering_prize', 'rendering_long', 'rendering_prize_long')]")

    prizes_data = []
    for div in prizes_divs:
        textblock_list = div.xpath("div[@class='textblock']")
        if not textblock_list:
            desc = None
        else:
            desc = etree.tostring(textblock_list[0], encoding="unicode")

        prizes_spans = div.xpath("p/span[@class='person']")
        prizes_recipients = set()
        for span in prizes_spans:
            parent = span.getparent()
            for child in parent.getchildren():
                if child.text in name_parsed:
                    prizes_recipients.add(child.text)


        prizes_data.append({
            "title" : div.xpath("h2[@class='title']/span")[0].text,
            "description" : desc,
            "recipient" : prizes_recipients
        })
    return prizes_data

prizes_data = parse_prizes_data("ELEC_prizes.html", name_parsed)
sci_prizes = parse_prizes_data("SCI_prizes.html", sci_name_parsed)

# Add research output title + abstract to elec and sci persons data

for person in elec_persons:
    person.update({"outputs" : []})
    for output in research_outputs_data:
          if str(person["name"].split()[1] + ", " + re.match(pattern, person["name"].split()[0]).group(0) + ".") in output["contributor"]:
            if output["output_keywords"] is not None and output["title"] is not None and output["description"] is not None:
                person["outputs"].append(output["title"] + ", " + output["description"] + ", " + output["output_keywords"])
            elif output["output_keywords"] is not None and output["title"] is None and output["description"] is not None:
                person["outputs"].append(output["description"] + ", " + output["output_keywords"])
            elif output["output_keywords"] is not None and output["title"] is not None and output["description"] is None:
                person["outputs"].append(output["title"] + ", " + output["output_keywords"])
            elif output["output_keywords"] is None and output["title"] is not None and output["description"] is not None:
                person["outputs"].append(output["title"] + ", " + output["description"])
            elif output["output_keywords"] is None and output["title"] is None and output["description"] is None:
                person["outputs"].append(None)
            elif output["description"] == []:
                person["outputs"].append(None)

for person in sci_persons:
    person.update({"outputs" : []})
    for output in sci_outputs:
        if str(person["name"].split()[1] + ", " + re.match(pattern, person["name"].split()[0]).group(0) + ".") in output["contributor"]:
            if output["output_keywords"] is not None and output["title"] is not None and output["description"] is not None:
                person["outputs"].append(output["title"] + ", " + output["description"] + ", " + output["output_keywords"])
            elif output["output_keywords"] is not None and output["title"] is None and output["description"] is not None:
                person["outputs"].append(output["description"] + ", " + output["output_keywords"])
            elif output["output_keywords"] is not None and output["title"] is not None and output["description"] is None:
                person["outputs"].append(output["title"] + ", " + output["output_keywords"])
            elif output["output_keywords"] is None and output["title"] is not None and output["description"] is not None:
                person["outputs"].append(output["title"] + ", " + output["description"])
            elif output["output_keywords"] is None and output["title"] is None and output["description"] is None:
                person["outputs"].append(None)
            elif output["description"] == []:
                person["outputs"].append(None)


#Add prizes
for person in elec_persons:
    person.update({"prizes" : []})
    for prize in prizes_data:
        if str(person["name"].split()[1] + ", " + re.match(pattern, person["name"].split()[0]).group(0) + ".") in prize["recipient"] and prize["title"] and prize["description"]:
            person["prizes"].append(prize["title"] + ", " + prize["description"])

#Add prizes
for person in sci_persons:
    person.update({"prizes" : []})
    for prize in sci_prizes:
        if str(person["name"].split()[1] + ", " + re.match(pattern, person["name"].split()[0]).group(0) + ".") in prize["recipient"] and prize["title"] and prize["description"]:
            person["prizes"].append(prize["title"] + ", " + prize["description"])

#process person data to contain [[name, [desc + keyw], [title + abst]], [name, [desc + keyw], [title+abst]], ... ]
def process_person_data(person_data):
    persons_text_data = []
    persons_list_dict = []

    for person in person_data:
        person_list = []
        person_list.append(person["name"])
        keyw = ""
        joint_string = ""
        if person["outputs"] is not None and person["description"] is not None:
            for word in person["outputs"]:
                joint_string += word
            if joint_string != "":
                person_list.append(joint_string)
            joint_string = ""
            for word in person["description"]:
                joint_string += word
            if joint_string != "":
                person_list.append(joint_string)
        elif person["description"] is None and person["outputs"] is not None:
            for word in person["outputs"]:
                joint_string += word
            if joint_string != "":
                person_list.append(joint_string)
            joint_string = ""
            if person["keywords"] is not None:
                for word in person["keywords"]:
                    joint_string += "," + word
                if joint_string != "":
                    person_list.append(joint_string)
            if person["prizes"] is not None and person["prizes"] != []:
                for word in person["prizes"]:
                    joint_string += word
                if joint_string != "":
                    person_list.append(joint_string)
        elif person["outputs"] is None and person["description"] is not None:
            for word in person["description"]:
                joint_string += word
            if joint_string != "":
                person_list.append(joint_string)
            joint_string = ""
            if person["prizes"] is not None and person["prizes"] != []:
                for word in person["prizes"]:
                    joint_string += word
                if joint_string != "":
                    person_list.append(joint_string)
            joint_string = ""
            if person["keywords"] is not None:
                for word in person["keywords"]:
                    joint_string += "," + word
                if joint_string != "":
                    person_list.append(joint_string)

        if len(person_list) != 1:
            persons_text_data.append(person_list)

    return persons_text_data

text_data = process_person_data(elec_persons)
sci_text_data = process_person_data(sci_persons)

def process_text_data(text_data):
    print("Process text data of %s professors" % len(text_data))
    processed_text_data = []
    stop_words = stopwords.words('english')
    stop_words.extend(['also', 'however', 'could', 'like', 'therefore', 'may', 'quite', '\'s', 'able', 'must',
                    'often', 'since', 'whether', 'unless', 'upon', 'even', 'thus', 'in', 'on', 'when', 'under',
                    'using', 'without', "'s", 'abl', 'abov', 'ain', 'ani', 'aren', "aren't", 'becaus', 'befor',
                    'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'doe', 'doesn', "doesn't", 'don', "don't",
                    'dure', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'howev', 'isn', "isn't", 'just',
                    'like', 'll', 'm', 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'o', 'onc',
                    'onli', 'ourselv', 'quit', 's', 'shan', "shan't", "should'v", 'shouldn', "shouldn't", 'sinc', 't',
                    "that'll", 'themselv', 'therefor', 'unless', 'use', 've', 'veri', 'wasn', "wasn't", 'weren', "weren't",
                    'whi', 'won', "won't", 'wouldn', "wouldn't", 'y', "you'd", "you'll", "you'r", "you'v", 'yourselv', 'self'])

    for person_text_list in text_data:
        remove_spans = [re.sub(r'</?span>', ' ', word) for word in person_text_list]

        remove_non_alphanumeric = [re.sub(r'\W', ' ', word) for word in remove_spans]

        tokens = [tokenizer.tokenize(text.split()) for text in remove_non_alphanumeric]

        remove_underscores = [token.replace("_", " ") for item in tokens for token in item]


        pos = nltk.pos_tag(remove_underscores)
        remove_verbs = [pos_tuple for pos_tuple in pos if pos_tuple[1] != 'VBZ' and pos_tuple[1] != 'RB' and pos_tuple[1] != 'JJ' and pos_tuple[1] != 'VBN']
        ne = nltk.ne_chunk(remove_verbs)

        removed_ne = [item[0] for item in ne if not isinstance(item, nltk.tree.Tree)]

        stopwords_removed = [word.lower() for word in removed_ne if word.lower() not in stop_words]

        filtered_tokens = []

        for token in stopwords_removed:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token.replace(".", ""))


        stemmer = SnowballStemmer("english")
        stems = [stemmer.stem(t) for t in filtered_tokens]


        name = person_text_list[0]
        processed_text_data.append(
            (name, stems)
        )

    return processed_text_data

professors_text_data = process_text_data(text_data)

sci_professors_text_data = process_text_data(sci_text_data)

def parse_nasa(vocabulary):
    physics_voc = []
    physics_voc_non_stems = {}
    stemmer = SnowballStemmer("english")
    with open(vocabulary) as n:
        lines = n.readlines()
        synonyms = []
        terms = ()
        label = ""
        for line in lines:
            split_line = line.split(",")
            term = re.sub(r'(\s"|"|\W\s)', '', split_line[1])
            synonym = re.sub(r'(\s"|"|\W\s)', '', split_line[5])
            if label != term:
                if terms:
                    physics_voc.append(terms)
                label = term
                synonyms = []
                terms = (term, synonyms)
            terms[1].append(stemmer.stem(synonym.lower()))
            physics_voc_non_stems[stemmer.stem(synonym.lower())] = synonym.lower()

    return physics_voc, physics_voc_non_stems

# physics_voc datastructure: [(Concept 1: [subterm1, subterm2, subterm3]), (Concept 2, [...])]

physics_voc, physics_voc_non_stems = parse_nasa("nasa_thesaurus.txt")


def parse_elec_voc(vocabulary):
    elec_voc = []
    elec_voc_non_stems = {}
    stemmer = SnowballStemmer("english")
    with open(vocabulary) as n:
        lines = n.readlines()
        synonyms = []
        terms = ()
        label = ""
        for line in lines:
            split_line = line.split(",")
            term = re.sub(r'(\s"|"|\W\s)', '', split_line[0])
            synonym = re.sub(r'(\s"|"|\W\s)', '', split_line[1])
            if label != term:
                if terms:
                    elec_voc.append(terms)
                label = term
                synonyms = []
                terms = (term, synonyms)
            terms[1].append(stemmer.stem(synonym.lower().replace("\n", "")))
            elec_voc_non_stems[stemmer.stem(synonym.lower().replace("\n", ""))] = synonym.lower().replace("\n", "")

    return elec_voc, elec_voc_non_stems

elec_voc, elec_voc_non_stems = parse_elec_voc("elec_voc.txt")

def parse_sci_voc(vocabulary):
    sci_voc = []
    sci_voc_non_stems = {}
    stemmer = SnowballStemmer("english")
    with open(vocabulary) as n:
        lines = n.readlines()
        synonyms = []
        terms = ()
        label = ""
        for line in lines:
            split_line = line.split(",")
            term = re.sub(r'(\s"|"|\W\s)', '', split_line[0])
            synonym = re.sub(r'(\s"|"|\W\s)', '', split_line[1])
            if label != term:
                if terms:
                    sci_voc.append(terms)
                label = term
                synonyms = []
                terms = (term, synonyms)
            terms[1].append(stemmer.stem(synonym.lower().replace("\n", "")))
            sci_voc_non_stems[stemmer.stem(synonym.lower().replace("\n", ""))] = synonym.lower().replace("\n", "")

    return sci_voc, sci_voc_non_stems

sci_voc, sci_voc_non_stems = parse_sci_voc("sci_voc.txt")

def save(fname, dictionary):
    from threading import Thread

    def noInterrupt(path, obj):
        print("save to", path)
        with open(path, 'w') as f_obj:
            json.dump(obj, f_obj, sort_keys=True, indent=4)

    a = Thread(target=noInterrupt, args=(fname,dictionary))
    a.start()
    a.join()

# compare words cosine distance with word2vec to match text data to vocabulary
def word2vec(word):
    from collections import Counter
    from math import sqrt

    cw = Counter(word)
    sw = set(cw)
    lw = sqrt(sum(c*c for c in cw.values()))

    return cw, sw, lw

def cosdis(v1, v2):
    common = v1[1].intersection(v2[1])
    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]

def compare_person_voc(professors_text_data):

    most_freq = []
    most_freq2 = []
    for professor in professors_text_data:
        professor_list = []
        professor_list.append(professor[0])
        counter = collections.Counter(professor[1])
        most_common = [(word[0], word[1]) for word in counter.items()]
        most_common.sort(key=lambda tuple : tuple[1], reverse=True)
        professor_list.append(most_common)
        most_freq.append(professor_list)
        most_freq2.append(professor[0])
        most_freq2.append(counter.most_common(12))


    print("gathering umbrella terms....")
    elec_concepts = collections.defaultdict(set)
    physics_concepts = collections.defaultdict(set)
    sci_concepts = collections.defaultdict(set)

    all_elec_umbrellas = set()
    all_physics_umbrellas = set()
    all_sci_umbrellas = set()

    for umbrella_term, subterm_list in elec_voc:
        for subterm in subterm_list:
            elec_concepts[subterm].add(umbrella_term)

    for umbrella_term, subterm_list in physics_voc:
        for subterm in subterm_list:
            physics_concepts[subterm].add(umbrella_term)

    for umbrella_term, subterm_list in sci_voc:
        for subterm in subterm_list:
            sci_concepts[subterm].add(umbrella_term)

    elec_concepts = {k:v for k, v in elec_concepts.items()}

    physics_concepts = {k:v for k, v in physics_concepts.items() if len(v) > 18}
    physics_concepts = {k:{i for i in v if i in physics_concepts} for k, v in physics_concepts.items()}

    sci_concepts = {k:v for k, v in sci_concepts.items()}

    for subterm, umbrella_set in physics_concepts.items():
        all_physics_umbrellas |= umbrella_set

    for subterm, umbrella_set in elec_concepts.items():
        all_elec_umbrellas |= umbrella_set

    for subterm, umbrella_set in sci_concepts.items():
        all_sci_umbrellas |= umbrella_set

    words = {}
    all_words = {f[0] for p in most_freq for f in p[1]}
    N = len(all_words)
    print(f"calculating cosdis between {N} most frequent words and all subterms found in elec, physics and sci vocabularies")
    for i, word in enumerate(all_words):
        for subterm in elec_concepts.keys():
            if word not in words:
                words[word] = {}
            words[word][subterm] = cosdis(word2vec(word), word2vec(subterm))

        for subterm in physics_concepts.keys():
            if word not in words:
                words[word] = {}
            words[word][subterm] = cosdis(word2vec(word), word2vec(subterm))

        for subterm in sci_concepts.keys():
            if word not in words:
                words[word] = {}
            words[word][subterm] = cosdis(word2vec(word), word2vec(subterm))

        if i % 1000 == 0:
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), i, '/', N)

    print("processing professors")

    all_elec_subterms = []
    all_physics_subterms = []
    all_sci_subterms = []
    person_fingerprints = []
    p_fingerprints = []
    word_list = []
    persons_umbrella_subterm_list = []
    for professor in most_freq:
        person_fingerprint = []
        p_fingerprint = []
        person_umbrellas_subterms = []
        person_umbrellas_subterms.append(professor[0])
        p_fingerprint.append(professor[0])
        person_fingerprint.append(professor[0])
        person_concepts_elec = []
        person_concepts_physics = []
        person_concepts_sci = []
        person_subterms_elec = []
        person_subterms_physics= []
        person_subterms_sci = []
        person_umbrella_subterm = {}
        for freq_tuple in professor[1]:
            word, freq = freq_tuple
            all_words = []
            for i in range(freq):
                all_words.append(word)
            for subterm in elec_concepts:
                for word in all_words:
                    if words[word][subterm] > 0.95:
                        non_stemmed = elec_voc_non_stems[subterm]
                        person_subterms_elec.append(non_stemmed)
                        person_concepts_elec.extend(elec_concepts[subterm])
                        if non_stemmed not in person_umbrella_subterm.keys() and elec_concepts[subterm] != set():
                            person_umbrella_subterm[non_stemmed] = elec_concepts[subterm]


            for subterm in physics_concepts:
                for word in all_words:
                    if words[word][subterm] > 0.95:
                        non_stemmed = physics_voc_non_stems[subterm]
                        person_subterms_physics.append(non_stemmed)
                        person_concepts_physics.extend(physics_concepts[subterm])
                        if non_stemmed not in person_umbrella_subterm.keys() and physics_concepts[subterm] != set():
                            person_umbrella_subterm[non_stemmed] = physics_concepts[subterm]

            for subterm in sci_concepts:
                for word in all_words:
                    if words[word][subterm] > 0.95:
                        non_stemmed = sci_voc_non_stems[subterm]
                        person_subterms_sci.append(non_stemmed)
                        person_concepts_sci.extend(sci_concepts[subterm])
                        if non_stemmed not in person_umbrella_subterm.keys() and sci_concepts[subterm] != set():
                            person_umbrella_subterm[non_stemmed] = sci_concepts[subterm]

        person_fingerprint.append(person_concepts_elec)
        person_fingerprint.append(person_concepts_physics)
        person_fingerprint.append(person_concepts_sci)
        person_fingerprints.append(person_fingerprint)

        p_fingerprint.append(person_subterms_elec)
        p_fingerprint.append(person_subterms_physics)
        p_fingerprint.append(person_subterms_sci)
        p_fingerprints.append(p_fingerprint)

        person_umbrellas_subterms.append(person_umbrella_subterm)
        persons_umbrella_subterm_list.append(person_umbrellas_subterms)

    word_counter = collections.Counter(word_list)
    most_commons = word_counter.most_common(15)

    return person_fingerprints, all_elec_umbrellas, all_physics_umbrellas, all_sci_umbrellas, most_freq, most_freq2, persons_umbrella_subterm_list


person_concepts, all_elec_umbrellas, all_physics_umbrellas, all_sci_umbrellas, most_freq, most_freq2, person_umbrellas_subterms = compare_person_voc(professors_text_data + sci_professors_text_data)

def pick_most_freq_concepts(person_concepts):
    person_fingerprints = []
    person_and_freqs = []
    for professor in person_concepts:
        person_fingerprint = []
        person_freqs = []
        person_fingerprint.append(professor[0])
        person_freqs.append(professor[0])

        counter_elec = collections.Counter(professor[1])
        most_common_elec = counter_elec.most_common(20)
        most_common_elec_set = []
        for freq_tuple in most_common_elec:
            freq_list = list(freq_tuple)
            freq_list[0] += ", " + "Elec voc"
            most_common_elec_set.append(freq_list)
        person_fingerprint.append("Electrical vocabulary")
        person_fingerprint.append(most_common_elec)


        counter_physics = collections.Counter(professor[2])
        most_common_physics = counter_physics.most_common(20)
        most_common_physics_set = []
        for freq_tuple in most_common_physics:
            freq_list = list(freq_tuple)
            freq_list[0] += ", " + "Physics voc"
            most_common_physics_set.append(freq_list)
        person_fingerprint.append("Physics vocabulary")
        person_fingerprint.append(most_common_physics)

        counter_sci = collections.Counter(professor[3])
        most_common_sci = counter_sci.most_common(20)
        most_common_sci_set = []
        for freq_tuple in most_common_sci:
            freq_list = list(freq_tuple)
            freq_list[0] += ", " + "Sci voc"
            most_common_sci_set.append(freq_list)
        person_fingerprint.append("Sci vocabulary")
        person_fingerprint.append(most_common_sci)

        most_common_united = most_common_elec_set + most_common_physics_set + most_common_sci_set

        person_freqs.append(most_common_united)

        person_fingerprints.append(person_fingerprint)
        person_and_freqs.append(person_freqs)

    return person_fingerprints, person_and_freqs

training_data_fp, person_freqs = pick_most_freq_concepts(person_concepts)

tagged_elec_umbrellas = [word + ", " + "Elec voc" for word in all_elec_umbrellas]
tagged_physics_umbrellas = [word + ", " + "Physics voc" for word in all_physics_umbrellas]
tagged_sci_umbrellas = [word + ", " + "Sci voc" for word in all_sci_umbrellas]

columns = tagged_elec_umbrellas + tagged_physics_umbrellas + tagged_sci_umbrellas

d = { 'name' : [name[0] for name in text_data], 'description' : [ string_item for string_item in text_data if string_item != "" ] }
for item in sci_text_data:
    if item != "":
        d["name"].append(item[0])
        d["description"].append(item)


df = pd.DataFrame(data=d)


for c in columns:
    df.insert(2, c, 0, True)


row = 0

df_columns = list(df)
df_columns.remove("description")


for professor in person_freqs:
    for term, freqv in professor[1]:
        if freqv > 2:
            df.at[row, term] = 1
        else:
            df.at[row, term] = 0
    row += 1


csv = df.to_csv('dataframe.csv', index=False)

with open( "tokenizer.pickle", "wb" ) as fobj:
    pickle.dump( tokenizer, fobj, protocol=pickle.HIGHEST_PROTOCOL )

with open( "person_umbrellas_subterms.pickle", "wb" ) as fobj:
    pickle.dump( person_umbrellas_subterms, fobj )
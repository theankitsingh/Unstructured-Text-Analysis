#import numpy
#from sklearn.metrics import accuracy_score
import random 
from collections import Counter
import glob
import re
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from tokenize import tokenize



tokenizer=RegexpTokenizer("[^\W\d_]+")
en_stop = get_stop_words('en')
invalid = re.compile('[^a-z]')

def clean(file_data):
    #with open(file_data,'r',buffering=-1) as file:
        tokens = tokenizer.tokenize(file_data)
        cleaned = [i for i in tokens if not invalid.search(i)]
        raw_tokens = [i for i in cleaned if not i in en_stop]
        return raw_tokens

doc = glob.glob("/media/one/My Stuff1/STUDY/Project/Untitled Folder/*.txt")
documents=[]
for _file in doc:
    lines = []
    with open(_file) as myfile:
        file=myfile.read()
        lines=clean(file)
        documents.append(lines)
            #for line in file:
            #   line = line.strip()
        #   line=line.lower()
            #   lines.append(line)
        #li_all_doc.append(lines)

def sample_from(weights):
    total = sum(weights)
    rnd = total * random.random()       # uniform between 0 and total
    for i, w in enumerate(weights):
        rnd -= w                        # return the smallest i such that
        if rnd <= 0: return i 
# documents = [
#     ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
#     ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
#     ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
#     ["R", "Python", "statistics", "regression", "probability"],
#     ["machine learning", "regression", "decision trees", "libsvm"],
#     ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
#     ["statistics", "probability", "mathematics", "theory"],
#     ["machine learning", "scikit-learn", "Mahout", "neural networks"],
#     ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
#     ["Hadoop", "Java", "MapReduce", "Big Data"],
#     ["statistics", "R", "statsmodels"],
#     ["C++", "deep learning", "artificial intelligence", "probability"],
#     ["pandas", "R", "Python"],
#     ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
#     ["libsvm", "regression", "support vector machines"]
# ]
K = 4

document_topic_counts = [Counter()
                         for _ in documents]

topic_word_counts = [Counter() for _ in range(K)]

topic_counts = [0 for _ in range(K)]

document_lengths = [len(d) for d in documents]

distinct_words = set(word for document in documents for word in document)
W = len(distinct_words)

D = len(documents)

def p_topic_given_document(topic, d, alpha=0.1):
    """the fraction of words in document _d_
    that are assigned to _topic_ (plus some smoothing)"""

    return ((document_topic_counts[d][topic] + alpha) /
            (document_lengths[d] + K * alpha))

def p_word_given_topic(word, topic, beta=0.1):
    """the fraction of words assigned to _topic_
    that equal _word_ (plus some smoothing)"""

    return ((topic_word_counts[topic][word] + beta) /
            (topic_counts[topic] + W * beta))

def topic_weight(d, word, k):
    """given a document and a word in that document,
    return the weight for the k-th topic"""

    return p_word_given_topic(word, k) * p_topic_given_document(k, d)

def choose_new_topic(d, word):
    return sample_from([topic_weight(d, word, k)
                        for k in range(K)])


random.seed(0)
document_topics = [[random.randrange(K) for word in document]
                   for document in documents]

for d in range(D):
    for word, topic in zip(documents[d], document_topics[d]):
        document_topic_counts[d][topic] += 1
        topic_word_counts[topic][word] += 1
        topic_counts[topic] += 1

for iter in range(1000):
    for d in range(D):
        for i, (word, topic) in enumerate(zip(documents[d],
                                              document_topics[d])):

            # remove this word / topic from the counts
            # so that it doesn't influence the weights
            document_topic_counts[d][topic] -= 1
            topic_word_counts[topic][word] -= 1
            topic_counts[topic] -= 1
            document_lengths[d] -= 1

            # choose a new topic based on the weights
            new_topic = choose_new_topic(d, word)
            document_topics[d][i] = new_topic

            # and now add it back to the counts
            document_topic_counts[d][new_topic] += 1
            topic_word_counts[new_topic][word] += 1
            topic_counts[new_topic] += 1
            document_lengths[d] += 1

if __name__ == "__main__":
    for k, word_counts in enumerate(topic_word_counts):
        for word, count in word_counts.most_common():
            if count > 0: 
                    if((k+1)==1):
                         f= open("1.txt","a")
                         f.writelines('{} {} {} \n'.format(k+1, word, count))                
                    if((k+1)==2):
                         f= open("2.txt","a")
                         f.writelines('{} {} {} \n'.format(k+1, word, count))                
                    if((k+1)==3):
                         f= open("3.txt","a")
                         f.writelines('{} {} {} \n'.format(k+1, word, count))                
                    if((k+1)==4):
                         f= open("4.txt","a")
                         f.writelines('{} {} {} \n'.format(k+1, word, count))                

    topic_names = ["Topic:1-Big Data and programming languages",
                   "Topic:2-databases",
                   "Topic:3-machine learning",
                   "Topic:4-statistics"]

    for document, topic_counts in zip(documents, document_topic_counts):
	def fmtpairs(document):
    		pairs = zip(document[::2],document[1::2])
    		return '\n'.join('\t'.join(i) for i in pairs)        
	print fmtpairs(document)
	print(document)
        for topic, count in topic_counts.most_common():
            if count > 0:
                f= open("doc1.txt","a")
                f.writelines('{} {} \n'.format(topic_names[topic],count))
        print()
#accuracy_score(topic_names[topics],count)

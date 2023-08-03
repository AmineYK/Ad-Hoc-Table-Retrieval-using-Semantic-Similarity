
import math
import pyterrier as pt
import re
import numpy as np
import collections
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
from SPARQLWrapper import SPARQLWrapper, JSON
import warnings
warnings.filterwarnings("ignore")
import wikipedia
import gensim.downloader as api
from pyrdf2vec.graphs import KG
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker

if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

wv = api.load('word2vec-google-news-300')


#########################################################
#########################################################
#FONCTIONS DE RECUPERATION DES DONNEES DEPUIS LES TABLES#
#########################################################
#########################################################

# renvoie les data des headers d'une table
def get_text_from_headers(table):
    text = ""
    for line in table['tableHeaders'][0]:
        text += line['text']+" "
    return text

# renvoie le text des data d'une table
def get_text_from_data(table):
    text = ""
    for line in table['tableData']:
        for i in range(len(table['tableHeaders'][0])):
            text += line[i]['text']+" "
    return text

# renvoie les columns d'une table (pour chaque header)
def get_columns_data_table(table):
    cols = []
    for i in range(len(table['tableHeaders'][0])):
        l = []
        for line in table['tableData']:
            entity = line[i]['text']
            l.append(entity+" ")
        cols.append(l)

    return cols

# tokensize un texte
def tokenize(text):
    tokens = re.findall(r'\w+', text.lower())
    return set(tokens)



#########################################################
#########################################################
###########FONCTIONS DE REQUETE VERS DBPEDIA#############
#########################################################
#########################################################


# Verifie si une entité est presente sur DBedia 
def query_dbpedia(entity):
    # Set up SPARQL endpoint
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    # Formulate the SPARQL query
    query = f"""
        ASK WHERE {{
            <http://dbpedia.org/resource/{entity}> ?p ?o .
        }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # Execute the query and check the presence of the entity
    results = sparql.query().convert()

    return results["boolean"]


# Recupere toutes les entités de DBedia 
def entities_in_DBedia(document):
    # Example document
    # document = "La Tour Eiffel est située à Paris. Barack Obama a été président des États-Unis."

    # Tokenization and POS tagging
    tokens = document.split()  # Split the document into tokens (words)
    pos_tags = nltk.pos_tag(tokens)  # Perform POS tagging on the tokens

    # Extract named entities and filter based on presence in DBpedia
    entities = [entity[0] for entity in pos_tags if query_dbpedia(entity[0])]

    return entities

# Verifie si deux entités sont liés sur DBedia
def are_entities_linked(entity1, entity2):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    query = """
    SELECT DISTINCT ?property
    WHERE {{
        <http://dbpedia.org/resource/{}> ?property ?value1.
        <http://dbpedia.org/resource/{}> ?property ?value2.
    }}
    """.format(entity1, entity2)

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    properties = [result["property"]["value"] for result in results["results"]["bindings"]]

    return len(properties) > 0


#########################################################
#########################################################
###########FONCTIONS D'EXTRACTION WIKIPEDIA #############
#########################################################
#########################################################

# Retourne toutes les categories wikipedia d'une entité
def get_entity_categories(entity):
    try:
        page = wikipedia.page(entity)
        categories = page.categories
        return categories
    except wikipedia.exceptions.PageError:
        return []
    except wikipedia.exceptions.DisambiguationError:
        return []

# Renvoie toutes les categories sans doublons d'un contenu (ensemble d'entités)
def get_categories_content(content_entity):
    cats = []
    for entity in content_entity:
        cats.extend(get_entity_categories(entity))
        
    return set(cats)
    
    
#########################################################
#########################################################
################## FONCTIONS PRINCIPALES ################
#########################################################
#########################################################
    


# calcule la similarité cosinus entre deux vecteurs v1 et v2
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    # pour eviter une division par 0 dans le cas ou le vecteur contient que des composantes nulles
    eps = 1e-5
    norm_v1 = np.linalg.norm(v1) + eps
    norm_v2 = np.linalg.norm(v2) + eps
    return dot_product / (norm_v1 * norm_v2)
    
    
# Renvoie l'ensemble des entités de la colonne principale de la table
def CoreColumn(table):
    
    ratios = []

    # les columns de la table 
    columns  = get_columns_data_table(table)
    # garder uniquement les entités presentes sur DBedia pour chaque colonne 
    entities_columns = []
    for col in columns:
        entities_columns.append(entities_in_DBedia(col))

    # calcule le ratios des columns 
    for entities in entities_columns:
        if entities == []:
            ratios.append(0)
            continue
        unique_ent,values = np.unique(entities,return_counts=True)
        ratios.append(np.max(values) / len(unique_ent))
    # renvoie l'indice de la colonne principale
    i_CoreCol = np.argmax(ratios)

    # renvoyer l'ensemble des entités de colonne principale
    Ecc = np.unique(entities_columns[i_CoreCol])
    return set(Ecc)
    
# renvoie les 10 entités les mieux scorés 
def R_k(set_entities,k=10):
    return tokenize(set_entities)[:10]


# Renvoie un ensemble d'entités/mots en fonction de la base
def content_extraction(table,base='word',query=False):

    # Si c'est une requete
    if query and base == 'word':
        return set(tokenize(table))

    if query and base == 'entity':
        return R_k(table)

    pgtitle = table['pgTitle']   
    caption = table['tableCaption']
    text_headers = get_text_from_headers(table)
    
    # Word-based
    if base == 'word':
        table_tokens = set()
        table_tokens.update(tokenize(pgtitle))
        table_tokens.update(tokenize(caption))
        table_tokens.update(tokenize(text_headers))
        return table_tokens

    # Entity-based
    if base == 'entity':
        Ecc = CoreColumn(table)
        Tpt = R_k(pgtitle,10)
        Tpc = R_k(caption,10)
        return Ecc | Tpt | Tpc


# Renvoie une representation semantique du contenu en fonction de l'espace choisis
def semantic_representation(content,vocabulaire,space='boe'): 


  # boe : bag of entities    
  if space == 'boe':
    # vocabulaire ---> entities

    # Pour des raisons de temps de convergence : au lieu d'utiliser le vocabulaire de DBedia 
    # on utilise ce vocabulaire : un shuffle des valeurs de l'ensemble du contenu
    # on simplifie en creeant une matrice de dimension le nombre d'entités presentes dans le contenu
    
    boe = np.zeros((len(content),len(vocabulaire)))
    for i,entity in enumerate(content):
        for j,entity_voca in enumerate(vocabulaire):
            # si les deux entités sont liés selon DBedia 
            if are_entities_linked(entity_voca,entity):
                boe[i][j] = 1
    return boe

  # boc : bag of categories
  if space == 'boc':
    # vocabulaire ---> categories

    boc = np.zeros((len(content) , len(vocabulaire)))    
    for i,entity in enumerate(content):
        cats = get_entity_categories(entity)
        for cat_ent in cats:
            for j,cat in enumerate(vocabulaire):
                # si l'entité a comme categorie cat 
                if cat == cat_ent:
                    boc[i][j] = 1
    return boc

  # we : word embeddings
  if space == 'we':
    l = []
    for word in content:
        # si c'est un mot existant dans pour le modele de gensim
        if word in wv:
            l.append(wv.get_vector(word))
        # on creer un vecteur aleatoire pour le mot
        else: l.append(randomvec())
    return np.array(l)


  # ge : graph embeddings
  if space == 'ge':
        
    kg = KG(location="https://dbpedia.org/sparql", is_remote=True)

    transformer = RDF2VecTransformer(Word2Vec(),
        walkers=[RandomWalker(4, 10)])
    # Entities should be a list of URIs that can be found in the Knowledge Graph
    entities = []
    for ent in content:
        entities.append("http://dbpedia.org/resource/"+str(ent))

    embeddings = transformer.fit_transform(kg,entities)

    return np.array(embeddings)


# retourne le score de similarité cos selon la strategy
def similarity(table,query,strategy='early',aggr='max',isEmbedding=False):

    table = np.array(table)
    query = np.array(query)


    if strategy == 'early':
        # si on est dans une representation word embeddings
        if isEmbedding != False:
            table_words,query_words,words_corpus = isEmbedding
            # prendre en compte les ponderations tfidf des mots 
            table_tfidf_weights = calculate_tfidf(table_words,words_corpus)
            query_tfidf_weights = calculate_tfidf(query_words,words_corpus)
            
            centroid_query = []
            for i,tq in enumerate(query):
                centroid_query.append(tq * query_tfidf_weights[i])
                
            centroid_table = []
            for i,tt in enumerate(table):
                centroid_table.append(tt * table_tfidf_weights[i])
                
            # calcule des vecteurs centroids
            centroid_query = np.sum(centroid_query,axis=0)
            centroid_table = np.sum(centroid_table,axis=0)

            return cosine_similarity(centroid_table,centroid_query)
            
        # si on est pas dans une representation word embeddings
        # calcule des vecteurs centroids
        centroid_query = np.sum(query,axis=0) / query.shape[0]
        centroid_table = np.sum(table,axis=0) / table.shape[0]
        return cosine_similarity(centroid_table,centroid_query)

    if strategy == 'late':
        scores = []
        for t in range(table.shape[0]):
            for q in range(query.shape[0]):
                score = cosine_similarity(table[t],query[q])
                scores.append(score)
        if aggr == 'max':
            return np.max(scores)
        if aggr == 'avg':
            return np.mean(scores)
        if aggr == 'sum':
            return np.sum(scores)

# calcule le score BM25 d'une requete et un document
def score_bm25(query, document, k1=1.2, b=0.75):
    # Tokenize and stem the query and document
    query_tokens = word_tokenize(query, language='french')
    document_tokens = word_tokenize(document, language='french')
    stemmer = FrenchStemmer()
    query_tokens = [stemmer.stem(token) for token in query_tokens]
    document_tokens = [stemmer.stem(token) for token in document_tokens]

    # Calculate term frequencies in the document
    doc_term_freq = {}
    for term in document_tokens:
        doc_term_freq[term] = doc_term_freq.get(term, 0) + 1

    # Calculate BM25 score
    score = 0
    avg_doc_len = len(document_tokens)
    for term in query_tokens:
        if term in doc_term_freq:
            tf = doc_term_freq[term]
            idf = math.log((len(document_tokens) - doc_term_freq.get(term, 0) + 0.5) / (doc_term_freq.get(term, 0) + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (avg_doc_len / avg_doc_len)))

    return score
    
# fait un upsupervised ranking et calcule le score BM25 associé
def unsuprvised_ranking(table,q,single=True,weights=[1/5,1/5,1/5,1/5,1/5]):
    document = ""
    headers = table['tableHeaders']
    pgtitle = table['pgTitle']   
    caption = table['tableCaption']
    tabledata = table_data = table['tableData']
    sectiontitle = table['sectionTitle']
    text_data = get_text_from_data(table)
    text_headers = get_text_from_headers(table)
    f_i = [pgtitle,sectiontitle,caption,text_headers,text_data]
    
    if single:
        document += pgtitle +" "+caption+" "+sectiontitle+text_headers+" "+text_data
        return score_bm25(q,document)
    
    else:
        score = 0
        if np.sum(weights) != 1:
            raise Exception("Weights are not summing to 1 !!")
        for i,weight in enumerate(weights):
            score += score_bm25(q,f_i[i])*weight
        return score

# calcule un vecter de ponderations tfidf pour le contenu sur le corpus de documents 
def calculate_tfidf(content, documents):
    # Calculer la fréquence des termes dans le document (TF)
    term_freq = collections.Counter(content)

    # Calculer l'inverse de la fréquence des termes dans l'ensemble de documents (IDF)
    idf = {}
    total_documents = len(documents)

    for term in content:
        doc_count = sum(1 for doc in documents if term in doc)
        idf[term] = math.log(total_documents / (1 + doc_count))

    # Calculer les pondérations TF-IDF
    tfidf = {}

    for term, freq in term_freq.items():
        tfidf[term] = freq * idf[term]

    return np.array(list(tfidf.values()))

# calcule de ndcg sur une collection de documents
def ndcg(ranking, relevance, k):
    # Calcul du DCG
    dcg = 0.0
    for i in range(min(k, len(ranking))):
        dcg += (2**relevance[ranking[i]] - 1) / np.log2(i + 2)

    sorted_relevance = sorted(relevance, reverse=True)
    idcg = 0.0
    for i in range(min(k, len(sorted_relevance))):
        idcg += (2 ** sorted_relevance[i] - 1) / np.log2(i + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg

# renvoie la relevance des documents du corpus
def get_relevance(corpus):
    rel = []
    for table in corpus:
        rel.append(int(table['_id'].split('-')[1]))
    return rel

    
# calcule un vecteur random de dimension 300
def randomvec():
    default = np.random.randn(300)
    default = default  / np.linalg.norm(default)
    return default
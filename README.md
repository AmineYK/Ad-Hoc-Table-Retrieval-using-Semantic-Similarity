# Ad-Hoc-Table-Retrieval-using-Semantic-Similarity

This is a research project focused on Information Retrieval. It involves the implementation of various information retrieval models described in the research paper titled "Ad Hoc Table Retrieval using Semantic Similarity," written by Shuo Zhang and Krisztian Balog from the University of Stavanger.
The paper was presented at the WWW 2018: The 2018 Web Conference, which took place in Lyon, France, from April 23rd to 27th, 2018. It is published in the ACM format as follows:

Shuo Zhang and Krisztian Balog. 2018. Ad Hoc Table Retrieval using Semantic Similarity. In WWW 2018: The 2018 Web Conference, April 23–27, 2018, Lyon, France. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3178876.3186067.



In this project, we aim to explore and implement the techniques proposed in the paper, which focuses on ad hoc table retrieval through the application of semantic similarity. By reproducing and potentially extending the findings of the original research, we strive to contribute to the advancement of information retrieval methods and their practical applications.
The research paper titled "Ad Hoc Table Retrieval using Semantic Similarity" introduces several techniques for ad hoc table retrieval based on semantic similarity. Here is a summary of the main techniques presented in the paper:

### Semantic Similarity
The paper proposes to use semantic similarity as a key factor in table retrieval. Semantic similarity measures the resemblance between tables based on the meaning of their content rather than relying solely on exact matches or surface-level features.

### Table Representation
To compute semantic similarity, tables are represented in a structured format that captures both the tabular structure and the textual content within the cells. This representation allows for a more meaningful comparison between tables.

### Word Embeddings 
Word embeddings are utilized to capture the semantic relationships between words in the table cells. By mapping words to dense vectors in a continuous space, the model can capture the contextual meaning of the words and enhance the table representation.

### Semantic Matching
The paper proposes a semantic matching approach that considers both the structural alignment and the content similarity between tables. This allows for a comprehensive comparison of tables while accounting for variations in their structure and content.

### Ranking and Retrieval
The techniques are integrated into a retrieval system that ranks tables based on their semantic similarity to a given query. The system aims to retrieve tables that are most relevant to the query, taking into account both the table's structure and the meaning of its content.

### Evaluation: 
The proposed techniques are evaluated using standard benchmark datasets and compared with existing table retrieval methods. The evaluation metrics include precision, recall, and F1 score, which assess the effectiveness and performance of the approach.

#### PS
I did not replicate the experiments from the paper exactly, as they require a significant amount of data, and for the sake of convergence time, I attempted to minimize the data while maintaining the coherency of the work done.

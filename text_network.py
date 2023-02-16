from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from pyvis.network import Network
import networkx as nx


class TextNetwork:
    def __init__(self):
        self.g = nx.Graph()

    def create_network(self, data_reviews, column_name):
        for index, row in data_reviews.iterrows():
            text = row[column_name]
            tokens = word_tokenize(text)
            # Add edges between each pair of tokens
            for i in range(len(tokens)):
                for j in range(i+1, len(tokens)):
                    self.g.add_edge(tokens[i], tokens[j])
    def draw_network(self):
        net = Network(notebook=True, cdn_resources='in_line')
        net.width = "100%"
        net.height = "800px"
        net.from_nx(self.g)
        return net.show("graph.html")

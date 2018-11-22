from summa.keywords import keywords,get_graph
from summa.summarizer import summarize,get_graph
from summa.commons import Graph as TGraph
from networkx import Graph

def buld_braph(graph=TGraph()):
    g = Graph()
    for edge in graph.edges():
        g.add_edge(edge)

if __name__ == '__main__':
    text = """Automatic summarization is the process of reducing a text document with a \
computer program in order to create a summary that retains the most important points \
of the original document. As the problem of information overload has grown, and as \
the quantity of data has increased, so has interest in automatic summarization. \
Technologies that can make a coherent summary take into account variables such as \
length, writing style and syntax. An example of the use of summarization technology \
is search engines such as Google. Document summarization is another."""

    print(keywords(text))
    print(summarize(text))
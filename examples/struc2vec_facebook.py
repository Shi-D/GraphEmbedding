import numpy as np



from ge.classify import read_node_label,Classifier

from ge import Struc2Vec

from sklearn.linear_model import LogisticRegression



import matplotlib.pyplot as plt

import networkx as nx

from sklearn.manifold import TSNE




def plot_embeddings(embeddings, nodenum):

    emb_list = []

    for k in range(nodenum):

        emb_list.append(embeddings[str(k)])

    emb_list = np.array(emb_list)


    model = TSNE(n_components=2)

    node_pos = model.fit_transform(emb_list)


    plt.scatter(node_pos[0], node_pos[1], label=1)  # c=node_colors)

    plt.legend()

    plt.show()


if __name__ == "__main__":

    for i in range(50):
        print('i:', i)
        G = nx.read_edgelist('../data/DY-FB/facebook.'+str(i)+'.edges', create_using=nx.DiGraph(), nodetype=None,
                             data=[('weight', float)])
        a = np.array(range(100)).astype(str)
        G.add_nodes_from(a)

        model = Struc2Vec(G, 10, 80, workers=4, verbose=40, )
        model.train(embed_size=8)
        embeddings = model.get_embeddings()

        # print(embeddings)

        with open('../data/DY-FB/emb/facebook.'+str(i)+'.emb', 'w') as f:
            for i in embeddings:
                f.write(str(i) + ' ' + str(list(embeddings[i])) + '\n')



            # plot_embeddings(embeddings, 100)
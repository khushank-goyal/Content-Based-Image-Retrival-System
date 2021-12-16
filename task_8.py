import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from featureUtilities import writeToFile
### Function to get the subject-subject similarity  matrix  from the input file
### Input : name: name of the file
### Output: Subject - Subject Similarity Matrix
def get_matrix(name):

    #  Creates a path to the location of the file
    path = 'sim_matrix/'+name

    #Opens the file to read
    file = open(path, 'r')
    a = file.read()


    # Splits the input string by ',' and stores it into a list
    a = a.split(',')

    adj = list()

    # Converts every string into a float type and stores it a list

    for line in a:
        pp = line.split('\n')

        for i in pp:
            if i != '':
                adj.append(float(i))

    # Converts a list to numpy array and reshapes into of size(40,40) as there are 40 subjects
    adj = np.array(adj)
    adj = np.reshape(adj, (40, 40))


    # Returns the subject-subject similarity matrix
    return adj

### Function to calculate the n most similar subjects for a subject
### Input: l: list with similarity score for a subject with all the subjects
###        n : user specified integer value of n to find n similar subjects
### Output index : Returns the indexes for least similar subjects for given subjects

def n_significant(l, n):

    # List to store the index value for least significant subjects
    index = list()
    q = l.copy()

    # Sorts the input list based on the similarity score, with most similar objects first and least sililar last
    q.sort(reverse=True)

    # least significant subjects (k) = total numbers of subjects -  n most similar subjects
    k = len(l) - n

    # Retrieving the index of k least significant subjects and storing it into list 'index'
    for i in range(k):
        o = l.index(q[i])
        # print(l[i], o)
        index.append(o)

    # Returns the list with indexes of least similar objects for a particular subject
    return index


### Function to calculate the incoming edges of the nodes in the graph
### Input : adj -Adjacency matrix for the graph of size 40 * 40
### Output : N- List with number of incoming nodes for each subject i.e node
###          Nedges -  List with the incoming nodes for each subject
def incoming( adj):

    ## creates an np array of length of adjacency matrix i.e. 40 with all values of zeros
    N = np.zeros(len(adj))

    ## Creates a empty list for each node( subject) in the graph
    Nedges = [[] for x in range(len(adj))]

    ## Iteates through the adjacency matrix to calculate incoming edge to each node
    for i in range(len(adj)):
        for j in range(len(adj)):

            if adj[i][j] != 0 :
                N[j] = N[j] + 1
                Nedges[j].append(i + 1)

    # Returns the list with number of incoming edges and nodes of incoming edges for each node
    return N, Nedges
## Function to calculate adjacency matrix
## Input adj : adjacency matrix
## Output adj : updated adjacency matrix
def weights(adj):
    for i in range(len(adj)):
        for j in range(len(adj)):

            if adj[i][j]!=0:
                adj[i][j] = 1/adj[i][j]
    return adj

## Function to draw a graph from the given adjacency matrix
## Input : x - adjacency matrix of size (40 *40)
## Output : Returns a graph with nodes and edges


def drawGraph(x):

    # Initialize a empty directed graph
    G = nx.DiGraph()

    ## Iterate through through adjacency matrix to see if there is edge
    ## if adj[i][j] !=0 then there is edge from node i to node j
    for i in range(len(x)):
        for j in range(len(x)):
            if x[i][j] != 0:

                G.add_edge(i + 1, j + 1, weight=x[i][j])

    print(G)

    ## draw the graph and display it
    nx.draw(G, with_labels=True)
    #plt.show()

    # Return the Graph G with nodes and edges
    return G


## Function to create a graph for the given subject - subject similarity matrix
## Input : x - Subject- Subject Similarity Matrix of size 40 * 40
##         n - User Specified integer value for n most similar image
## Output: x - Adjacency matrix with edges to n most similar subjects
##         G - Graph with 40 nodes and 40 * n edges
##         N - List with number of incoming nodes for each subject i.e node
##         Nedges -  List with the incoming nodes for each subject

def createGraph(x, n):

    ## iterate through each subject to find n most similar subjects
    for i in x:

        index = n_significant(list(i), n)
        for j in index:
            i[j] = 0

    #print(x)

    ## Calculate the incoming edges for each node
    N, Nedges=incoming(x)
    #print(N)
    #print(Nedges)

    x = weights(x)

    ## Create a graph and display it
    G = drawGraph(x)

    # Return the graph, incoming edges and adjacency matrix for the graph
    return x,G,N, Nedges

## Function to check whether two matrices are equal or close to each other i.e if they converge
##  Input : s_initial - Matrix 1 to compare
##          s_final - Matrix 2 to compare
##  Output : Boolean value = True if it converges
##                             False if it doesnt
def s_converge(s_initial,s_final):

    ## iterate through each value in the 2-d matrix and check if they equal
    for i in range(len(s_initial)):
        for j in range(len(s_initial)):
            if (abs(s_initial[i][j] - s_final[i][j]) > 1e-5000):
                return False
    else:
        return True

## Function to implement ASCOS ++ algorithm
## Input : adj - Adjacency Matrix for the graph of size 40 *40
##         Nedges - Incoming edges for each nodes
##         s_final - ASCOS++ Similarity score matrix

def ascos(adj, Nedges):


    # Initialize the matrix of size 40 * 40 with all values as zeros
    s_initial = np.zeros((len(adj), len(adj)))

    # Initialize the matrix of size 40 * 40  with diagonal elements as 1
    s_final = np.eye(len(adj), len(adj))
    ## Considering the decay factor
    c = 0.9

    ## iterating until the matrix converge
    for iterations in range(10000):

        ## Check if matrix converge and stop if it does
        converge = s_converge(s_initial, s_final)
        if converge == True:
            break
        else:
        ##  Copying the value of similarity score matrix calculate in the previous iteration into temp matrix
            s_initial = s_final.copy()
            # print(s_initial)
        ## iterate to calculate similarity score of edge node with one another
            for i in range(len(adj)):
                for j in range(len(adj)):
                    # if both the nodes are same - Similarity Score is 1
                    if i == j:
                        s_final[i][j] = 1
                    else:
                    # Check if there are any Incoming edges
                        if len(Nedges[i]) != 0:
                            wi = 0

                            # take the incoming nodes into a list
                            neighbours = Nedges[i]
                            # print(i,neighbours)

                            # Iterate through each neighbour to caluclate w(p*)
                            # wp(*) = Summation of weights of p to incoming nodes r
                            for r in neighbours:
                                wi = wi + adj[i][r - 1]
                            count=0
                            if wi == 0:
                                wi = 1


                            s_final[i][j] = 0

                            sum = 0
                            # calculate the similarity score matrix S(ij)
                            # S(ij) =  w(ik) * e^(1- w(ik) * s(kj)( of previous iteration)
                            # where k = incoming neighbours of node i
                            for k in neighbours:
                                sum = sum + (adj[i][k - 1] * (1 - np.exp(-adj[i][k - 1])) * s_initial[k - 1][j])
                            s_final[i][j] = c * sum / wi
                            if np.isnan(s_final[i][j]):
                                print(s_final[i][j],sum,wi)

    #print(iterations)
    # return the ASCOS ++ Similarity score matrix
    return s_final

def task8():
    name = input('Enter the text file name which contains subject similarity matrix\n ')
    
    adj = get_matrix(name)
    #print(adj)

    n = int(input("Enter the number of similar subjects to consider for each subject 'n' :"))
    m = int(input("Enter the number of significant subjects you want 'm' :"))
    adj,G,N, Nedges = createGraph(adj,n+1)
    #print(sum(N))
    #print(adj)
    #print(Nedges)
    in_edge = dict()
    for i in range(len(N)):
        in_edge[i+1]=N[i]

    #print(in_edge)
    #print(adj)

    sim_matrix = ascos(adj,Nedges)


    similarity_matrix_folder = 'sim_matrix/'
    fileName = 't8.txt'

    filepath = f'{similarity_matrix_folder}{fileName}'
    writeToFile(sim_matrix,filepath)


    print(" The ASCOS ++ Similarity Score Matrix is ")
    print(sim_matrix)

    ## Calculate the m most significant subjects



    #sim_matrix=np.transpose(sim_matrix)
    rank_subjects=dict()
    sum_subject=[]
    for i in range(len(sim_matrix)):
        sum = 0
        for j in range(len(sim_matrix)):
            sum=sum+sim_matrix[i][j]
        rank_subjects[i+1] = sum


    sort_subjects = sorted(rank_subjects.items(), key=lambda x: x[1], reverse=True)
    #print(sort_subjects)
    sort_edges = sorted(in_edge.items(), key=lambda x: x[1], reverse=True)
    
    print(" The number of in-edges for each node ranked in decreasing order\n")
    print(sort_edges)
    print(" \nThe {} most Significant Subjects are \n".format(m))

    # print the m most significant Subjects
    for i in range(m):
        print("Subject {}".format(sort_subjects[i][0],sort_subjects))
        a=sort_subjects[i][0]
        #print(a,in_edge[a])

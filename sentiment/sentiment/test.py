import random
import numpy as np
from collections import Counter
import collections
def kmeans2(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    centroids=[sample.copy() for sample in random.sample(examples,K)]
    bestmatch=[random.randint(0,K-1) for item in examples]
    distances=[0 for item in examples]
    pastmatches=None
    examples_squared=[]
    for item in examples:
        tempdict=collections.defaultdict(float)
        for k,v in item.items():
            tempdict[k]=v*v
        examples_squared.append(tempdict)


    for run_range in range(maxIters):
        centroids_squared=[]
        for item in centroids:
            tempdict = collections.defaultdict(float)
            for k, v in item.items():
                tempdict[k] = v * v
            centroids_squared.append(tempdict)


        for index,item in enumerate(examples):
            min_distance=999999
            for i,cluster in enumerate(centroids):
                distance=sum(examples_squared[index].values())+sum(centroids_squared[i].values())
                #for k in set(item.keys() & cluster.keys()):
                for k in (item.keys() & cluster.keys()):
                    distance+=-2*item[k]*cluster[k]
                if distance<min_distance:
                    min_distance=distance
                    bestmatch[index]=i
                    distances[index]=min_distance
        if pastmatches==bestmatch:
            break
        else:
            clustercounts=[0 for cluster in centroids]
            for i,cluster in enumerate(centroids):
                for k in cluster:
                    cluster[k]=0.0
            for index,item in enumerate(examples):
                clustercounts[bestmatch[index]]+=1
                cluster=centroids[bestmatch[index]]
                for k,v in item.items():
                    if k in cluster:
                        cluster[k]+=v
                    else:
                        cluster[k]=0.0+v
            for i, cluster in enumerate(centroids):
                for k in cluster:
                    cluster[k]/=clustercounts[i]
            pastmatches=bestmatch[:]
    return centroids,bestmatch,sum(distances)
    # END_YOUR_CODE


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.

    how do i evaluate the clusters? 
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    clusters=[]
    all_words = []
    for f_v in examples:
        for word in f_v:
            if word not in all_words:
                all_words.append(word)
    
    
    for i in range(K):
        cluster={}
        for word in all_words:
            cluster[word] = random.randint(0,K) #word count random
        clusters.append(cluster)
    
    '''
    assignments = []
    for i in range(len(examples)):
        assignments.append(random.randint(1,K))
    '''
  
    def find_assignment(vector, clusters):
        min = np.inf
        assignment = -1
        for i in range(len(clusters)):
            dst = 0
            for word in vector:
                if word in clusters[i]:
                    dst += (vector[word]-clusters[i][word])* (vector[word]-clusters[i][word])
                else:
                    dst += vector[word]*vector[word]
            
            if  dst < min:
                assignment = i
                min = dst
        return assignment

    
    

    assignments = np.zeros(len(examples))
    old_assignments = assignments
    old_clusters = clusters

    for t in range(maxIters):
        
        for i in range(len(examples)):
            f_v = examples[i]
            assignments[i] = find_assignment(f_v, clusters)
        # print(assignments)

        # for i in range(len(clusters)):
        #     for word in clusters[i]:
        #         clusters[i][word] =  avg_word_count_given_cluster(examples, i, word)#average of all the examples mapped to this cluster
        # #print(clusters)

        clusters = [{} for _ in range(K)]
       
        mapping_count = [0 for _ in range(K)]

        for i in range(len(examples)):
            j = int(assignments[i]) #assigned cluster index
            mapping_count[j]+=1
            for word in examples[i]:
                if word in clusters[j]:
                    clusters[j][word] += examples[i][word]
                else:
                    clusters[j][word] = 0.0 + examples[i][word]
            
        for i in range(len(clusters)):
            for word in clusters[i]:
                clusters[i][word] /= mapping_count[i]


        if clusters == old_clusters and np.array_equal(old_assignments, assignments):
            break
        old_clusters = clusters
        old_assignments = assignments

    
    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)


    print(examples)

    totalCost = 5
    return  (clusters, assignments, totalCost)
    # END_YOUR_CODE

def generateClusteringExamples(numExamples, numWordsPerTopic, numFillerWords):
    '''
    Generate artificial examples inspired by sentiment for clustering.
    Each review has a hidden sentiment (positive or negative) and a topic (plot, acting, or music).
    The actual review consists of 2 sentiment words, 4 topic words and 2 filler words, for example:

        good:1 great:1 plot1:2 plot7:1 plot9:1 filler0:1 filler10:1

    numExamples: Number of examples to generate
    numWordsPerTopic: Number of words per topic (e.g., plot0, plot1, ...)
    numFillerWords: Number of words per filler (e.g., filler0, filler1, ...)
    '''
    sentiments = [['bad', 'awful', 'worst', 'terrible'], ['good', 'great', 'fantastic', 'excellent']]
    topics = ['plot', 'acting', 'music']
    def generateExample():
        x = Counter()
        # Choose 2 sentiment words according to some sentiment
        sentimentWords = random.choice(sentiments)
        x[random.choice(sentimentWords)] += 1
        x[random.choice(sentimentWords)] += 1
        # Choose 4 topic words from a fixed topic
        topic = random.choice(topics)
        x[topic + str(random.randint(0, numWordsPerTopic-1))] += 1
        x[topic + str(random.randint(0, numWordsPerTopic-1))] += 1
        x[topic + str(random.randint(0, numWordsPerTopic-1))] += 1
        x[topic + str(random.randint(0, numWordsPerTopic-1))] += 1
        # Choose 2 filler words
        x['filler' + str(random.randint(0, numFillerWords-1))] += 1
        return x

    random.seed(42)
    examples = [generateExample() for _ in range(numExamples)]
    return examples

examples = generateClusteringExamples(10, 5, 4)

for v in examples:
    print(v)

clusters, assignments, totalCost = kmeans(examples, 3, 100)

print(assignments)
for v in clusters:
    print(v)

clusters, assignments, totalCost = kmeans2(examples, 3, 100)

print(assignments)
for v in clusters:
    print(v)




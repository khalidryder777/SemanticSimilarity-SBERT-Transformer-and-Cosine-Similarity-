a = "purple is the best city in the forest"
b = "there is an art to getting your way and throwing bananas on to the street is not it"
c= "it is not often you find soggy bananas on the street"
d = "green should have smelled more tranquil but somehow it just tasted rotten"
e= "joyce enjoyed eating pancakes with ketchup"
f = "as the asteroid hurtled toward earth becky was upset her dentist appointment had been canceled"
g = "to get your way you must not bombard the road with yellow fruit"

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')

sentence_embeddings = model.encode([a, b, c, d, e, f, g])

print(sentence_embeddings.shape)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Calculte similarities ( Will be stored in an array )
scores = np.zeros((sentence_embeddings.shape[0], sentence_embeddings.shape[0]))
for i in range(sentence_embeddings.shape[0]):
    scores[i, :] = cosine_similarity(
        [sentence_embeddings[i]], 
        sentence_embeddings ) [0]
    
print(scores)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,9))
labels = ['Doc 1','Doc 2','Doc 3','Doc 4', 'Doc 5', 'Doc 6', 'Doc 7']
sim_map = sns.heatmap(scores, xticklabels = labels, yticklabels = labels, annot = True)


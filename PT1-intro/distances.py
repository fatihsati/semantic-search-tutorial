import numpy as np


def dot_scratch(embed1, embed2):
    """dot product of two vectors is the sum of element wise multiplication of vectors.
    """
    dot = sum(i*j for i,j in zip(embed1, embed2))
    return dot


def norm_scratch(embed):
    """norm of magnitude. Square root of the sum of the square of each item.
    1. take square of each item in vector.
    2. take sum of the results..
    3. take square root of the sum."""
    return sum(i**2 for i in embed) ** 0.5


def calculate_cosine_scratch(embed1, embed2):
    """cosine similarity is calculating by:
    1. calculate dot product between embeddings
    2. calculate norm of each embed
    3. divine dot product to multiplication of norms.
    dot(embed1, embed2) / (norm(embed1) * norm(embed2))
    """
    cosine = dot_scratch(embed1, embed2) / (norm_scratch(embed1) * norm_scratch(embed2))
    return cosine


def calculate_cosine(embed1, embed2):
    """
    Calculate cosine using numpy functions
    """

    dot_product = np.dot(embed1, embed2)
    norm_embed1 = np.linalg.norm(embed1)
    norm_embed2 = np.linalg.norm(embed2)
    return dot_product / (norm_embed1 * norm_embed2)


def calculate_euclidean_scratch(embed1, embed2):
    """
    calculate the distance between two points:
    x: (2,3), y: (5,4)
    distance: square root of ( (2-5)**2 + (3-4)**2 )
    """
    return sum([(i-j)**2 for i,j in zip(embed1, embed2)]) ** 0.5
    
def calculate_euclidean(embed1, embed2):
    """calculate euclidean distance using numpy"""

    embed1 = np.array(embed1) # convert to np.array if not
    embed2 = np.array(embed2) # convert to np.array if not
    # numpy can handle element wise operation.
    return np.linalg.norm(embed1 - embed2)


if __name__ == "__main__":
    demo_embed1 = [1.9, 2.1, 0.5, 2.2, 1.0]
    demo_embed2 = [0.9, 2.9, 3.5, 2.0, 2.8]

    print()
    # cosine
    cos = calculate_cosine(demo_embed1, demo_embed2)
    print("Cosine sim.(numpy): ", cos)
    cos = calculate_cosine_scratch(demo_embed1, demo_embed2)
    print("Cosine sim.(Scratch): ", cos)

    print()
    # euclidean
    distance = calculate_euclidean(demo_embed1, demo_embed2)
    print("Euc. distance(numpy): ", distance)
    distance = calculate_euclidean_scratch(np.array(demo_embed1), demo_embed2)
    print("Euc. distance(Scratch): ", distance)


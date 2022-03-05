# Sam Berkson
# CPSC 322
# PA4

def compute_euclidean_distance(v1, v2):
    '''
    Returns the Euclidean distance between two vectors.
    '''
    distance = (sum([(float(v1[i]) - float(v2[i])) ** 2 for i in range(len(v1))])) ** 0.5
    return distance
    
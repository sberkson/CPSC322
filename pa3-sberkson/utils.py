# Sam Berkson
# CPSC 322
# PA3

from mypytable import MyPyTable

def get_frequencies(column):
    vals = []
    count = []

    for val in column:
        if val not in vals:
            vals.append(val)
            count.append(0)
        count[vals.index(val)] += 1
    
    return vals, count

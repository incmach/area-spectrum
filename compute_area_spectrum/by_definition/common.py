import numpy as np

def double_volume(*vs):
    v0 = vs[0]
    vs = vs[1:]
    #TODO this must be exact but looks non-exact, make it look exact e.g. via assertions
    return abs(int(np.round(np.linalg.det(
        [ 
            [ vij - v0j for vij, v0j in zip(vi, v0) ]
        for vi in vs ]))))


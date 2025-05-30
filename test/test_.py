from pyclassify.utils import distance, majority_vote
from pyclassify import kNN
from math import sqrt
import pytest

def test_distance():
    a = [0.,0.,0.]
    b = [1.,1.,1.]
    c = [2.,2.,2.]
    assert distance(a,b) == distance(b,a)
    assert distance(a,b) >= 0
    assert (sqrt(distance(a,c)) <= sqrt(distance(a,b)) + sqrt(distance(b,c)))

def test_majority_vote():
    x = [1,1,2,1,0,0]
    assert majority_vote(x) == 1

def test_kNN_constructor():
    with pytest.raises(ValueError):
        a = kNN(1.)
    with pytest.raises(ValueError):
        b = kNN(1,0)
    with pytest.raises(ValueError):
        c = kNN(1,'test')
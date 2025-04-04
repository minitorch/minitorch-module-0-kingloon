"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float):
    return x * y

def id(x: float):
    return x

def add(x: float, y: float):
    return x + y

def neg(x: float):
    return -x

def lt(x: float, y: float):
    return x < y

def eq(x: float, y: float):
    return x == y

def max(x: float, y: float):
    if x < y:
        return y
    return x

def is_close(x: float, y: float):
    return abs(x - y) < 1e-2

def sigmoid(x: float):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        # Does not blow up for negative x values
        return math.exp(x) / (1.0 + math.exp(x))
    
def relu(x: float):
    if x > 0:
        return x
    return 0.0

def log(x: float, tolerance=1e-10, max_iter=100):
    if x <= 0:
        raise ValueError("x must be positive")
    if x == 1:
        return 0.0
    y = 1.0
    for _ in range(max_iter):
        ey = exp(y)
        y_new = y - (ey - x) / ey
        if abs(y_new - y) < tolerance:
            return y_new
        y = y_new
    return y

def exp(x):
    # result = 0.0
    # factorial = 1
    # for n in range(n_terms):
    #     if n > 0:
    #         factorial *= n
    #     result += (x**n) / factorial
    # return result
    return math.exp(x)

def inv(x):
    return 1 / x

def log_back(x: float, d: float):
    return d / x

def inv_back(x: float, d: float):
    return -d / (x * x)

def relu_back(x: float, d: float):
    if x > 0:
        return d
    else:
        return 0.0

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn):
    """
    Higher-order map.
    
    Args:
        fn (one-arg function): Funtion from one value to one value
        
    Returns:
        function: A function that takes a list, applies 'func' to each element, 
        returns a list
    """
    def map_fn(ls):
        n = len(ls)
        res = [0 for i in range(n)]
        for i in range(n):
            res[i] = fn(ls[i])
        return res
    return map_fn

def zipWith(fn):
    """
    Higher-order zipWith (or map2)
    
    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_
    
    Args:
        fn (two-arg function): combine two values
        
    Returns:
        function: take two equally sized lists 'ls1' and 'ls2', produce a
        new list by applying fn(x, y) on each pair of elements.
    """
    def process(a, b):
        n = len(a)
        res = [0 for _ in range(n)]
        for i in range(n):
            res[i] = fn(a[i], b[i])
        return res
    return process


def reduce(fn, start):
    """
    Higher-order reduce.
    
    Args:
        fn(tow-arg function): combine two values
        start (float): start value :math: 'x_0'
        
    Returns:
        function: function that take a list 'ls' of elements
        :math:'x_1' \ldots 'x_n' and computes the reduction :math:'fn(x_3, fn(x_2,
        fn(x_1, x_0)))'
    """
    def reduce_fn(ls):
        n = len(ls)
        cur = start
        for i in range(n):
            cur = fn(cur, ls[i])
        return cur
    return reduce_fn

def negList(ls):
    "Use :func:'map' and func:'neg' to negate each element in ls"
    return map(neg)(ls)
    
def addLists(ls1, ls2):
    "Add the elements of 'ls1' and 'ls2' using :func: 'zipWith' and :func:'add'"
    return zipWith(add)(ls1, ls2)

def sum(ls):
    "Sum up a list using :func:'reduce' and func:'add'"
    return reduce(add, 0.0)(ls)

def prod(ls):
    "Product of a list using: func:'reduce' and cunc'mul'"
    return reduce(mul, 1.0)(ls)
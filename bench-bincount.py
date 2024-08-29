import platform
import timeit
import tracemalloc
from argparse import ArgumentParser
from itertools import product

import numpy
import pandas
import vigra
from numba import njit
from rich.progress import Progress


def vigra_bincount(labels):
    """
    A RAM-efficient implementation of numpy.bincount() when you're dealing with uint32 labels.
    If your data isn't int64, numpy.bincount() will copy it internally -- a huge RAM overhead.
    (This implementation may also need to make a copy, but it prefers uint32, not int64.)
    """
    labels = labels.astype(numpy.uint32, copy=False)
    labels = numpy.ravel(labels, order="K").reshape((-1, 1), order="A")
    # We don't care what the 'image' parameter is, but we have to give something
    image = labels.view(numpy.float32)
    counts = vigra.analysis.extractRegionFeatures(image, labels, ["Count"])["Count"]
    return counts.astype(numpy.int64)


def pandas_bincount(labels):
    labels = numpy.ravel(labels, order="K")
    labels = pandas.Series(labels, copy=False)
    vc = labels.value_counts()
    vc = vc.reindex(range(labels.max() + 1), fill_value=0)
    return vc.values


@njit
def numba_bincount(a):
    c = numpy.zeros(a.max() + 1, dtype=numpy.int64)
    for x in a.flat:
        c[x] += 1
    return c


def numpy_ufunc_bincount(a):
    a = numpy.ravel(a, order="K")
    counts = numpy.zeros(a.max() + 1, numpy.int64)
    numpy.add.at(counts, a, 1)
    return counts


SETUP = """
import numpy
import pandas
import vigra
from numba import njit

def vigra_bincount(labels):
    labels = labels.astype(numpy.uint32, copy=False)
    labels = numpy.ravel(labels, order="K").reshape((-1, 1), order="A")
    # We don't care what the 'image' parameter is, but we have to give something
    image = labels.view(numpy.float32)
    counts = vigra.analysis.extractRegionFeatures(image, labels, ["Count"])["Count"]
    return counts.astype(numpy.int64)

def pandas_bincount(labels):
    labels = numpy.ravel(labels, order="K")
    labels = pandas.Series(labels, copy=False)
    vc = labels.value_counts()
    vc = vc.reindex(range(labels.max()+1), fill_value=0)
    return vc.values

@njit
def numba_bincount(a):
    c = numpy.zeros(a.max()+1, dtype=numpy.int64)
    for x in a.flat:
        c[x] += 1
    return c

def numpy_ufunc_bincount(a):
    a = numpy.ravel(a, order='K')
    counts = numpy.zeros(a.max()+1, numpy.int64)
    numpy.add.at(counts, a, 1)
    return counts

data = numpy.random.randint(0, 256, {shape}, dtype="uint32")
"""


def check():
    data = numpy.random.randint(0, 256, (1000, 500, 250), dtype="uint32")

    # Sanity check
    b1 = numpy.bincount(data.reshape(-1))
    b2 = vigra_bincount(data)
    b3 = pandas_bincount(data)
    b4 = numpy_ufunc_bincount(data)
    b5 = numba_bincount(data)
    assert b1.tolist() == b2.tolist() == b3.tolist() == b4.tolist() == b5.tolist()


def parse_args():
    p = ArgumentParser()

    p.add_argument("--hostname", default=platform.node(), help="A way to identify your machine.")

    args = p.parse_args()
    return args


def main():
    args = parse_args()
    check()
    print("Check OK")

    py_version = platform.python_version()
    npy_version = numpy.__version__
    pd_version = pandas.__version__
    vigra_version = vigra.version
    host = args.hostname
    pf = platform.platform()

    results = {}  # {(method, shape): "tab\tseparated\tvalues"}

    number = 3

    shapes = [(512, 128, 1), (1024, 512, 1), (2048, 1024, 1), (512, 512, 32), (1024, 1024, 256)]

    methods = [
        "numpy.bincount(data.reshape(-1))",
        "vigra_bincount(data)",
        "pandas_bincount(data)",
        "numpy_ufunc_bincount(data)",
        "numba_bincount(data)",
    ]

    combinations = list(product(methods, shapes))

    with Progress() as p:
        for method, shape in p.track(combinations, description="Measuring runtime"):
            t = timeit.timeit(method, setup=SETUP.format(shape=shape), number=number) / number
            results[(method, shape)] = (
                f"{method}\t{shape!s}\t{t}\t{py_version}\t{npy_version}\t{pd_version}\t{vigra_version}\t{host}\t{pf}"
            )

        for method, shape in p.track(combinations, description="Measuring memory footprint"):
            tracemalloc.start()
            _t = timeit.timeit(method, setup=SETUP.format(shape=shape), number=number) / number

            mem_min, mem_max = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            results[(method, shape)] += f"\t{mem_min}\t{mem_max}"

    with open("results-bincount.csv", "a") as f:
        f.write("\n".join(results.values()))
        f.write("\n")


if __name__ == "__main__":
    main()

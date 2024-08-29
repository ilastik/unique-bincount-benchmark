import platform
import timeit
import tracemalloc
from argparse import ArgumentParser
from itertools import product

import numpy
import pandas
import vigra
from rich.progress import Progress

SETUP = """
import numpy
import vigra
import pandas


data = numpy.random.randint(0, 256, {shape}, dtype="uint32")

def bincount_unique(a):
    return numpy.bincount(a.reshape(-1)).nonzero()[0]


def pandas_unique(a):
    a = numpy.ravel(a, order="K")
    u = pandas.unique(a)
    u.sort()
    return u
"""


def bincount_unique(a):
    return numpy.bincount(a.reshape(-1)).nonzero()[0]


def pandas_unique(a):
    a = numpy.ravel(a, order="K")
    u = pandas.unique(a)
    u.sort()
    return u


def check():
    data = numpy.random.randint(0, 256, (1000, 500, 250), dtype="uint32")

    # Sanity check
    u1 = numpy.unique(data)
    u2 = vigra.analysis.unique(data)
    u3 = bincount_unique(data)
    u4 = pandas_unique(data)
    assert u1.tolist() == u2.tolist() == u3.tolist() == u4.tolist()


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
        "vigra.analysis.unique(data)",
        "numpy.unique(data)",
        "bincount_unique(data)",
        "pandas_unique(data)",
    ]

    combos = list(product(methods, shapes))

    with Progress() as p:
        for method, shape in p.track(combos, description="Measuring runtime"):
            t = timeit.timeit(method, setup=SETUP.format(shape=shape), number=number) / number
            results[(method, shape)] = (
                f"{method}\t{shape!s}\t{t}\t{py_version}\t{npy_version}\t{pd_version}\t{vigra_version}\t{host}\t{pf}"
            )

        for method, shape in p.track(combos, description="Measuring memory footprint"):
            tracemalloc.start()
            _t = timeit.timeit(method, setup=SETUP.format(shape=shape), number=number) / number

            mem_min, mem_max = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            results[(method, shape)] += f"\t{mem_min}\t{mem_max}"

    with open("results-unique.csv", "a") as f:
        f.write("\n".join(results.values()))
        f.write("\n")


if __name__ == "__main__":
    main()

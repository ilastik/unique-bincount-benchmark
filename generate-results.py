import datetime
from datetime import timedelta, timezone
from pathlib import Path

import numpy
import pandas
import pytz
import seaborn as sns
from jinja2 import Template
from pint import UnitRegistry

ureg = UnitRegistry()

x = ureg.Quantity(1, "bytes")


utctzinfo = timezone(timedelta(hours=0))
timeformat = "%Y-%m-%dT%H:%M:%S"


def gen_plots():
    shape_index_mapping = {
        "(512, 128, 1)": numpy.prod((512, 128, 1)) * 4 * ureg.bytes,
        "(1024, 512, 1)": numpy.prod((1024, 512, 1)) * 4 * ureg.bytes,
        "(2048, 1024, 1)": numpy.prod((2048, 1024, 1)) * 4 * ureg.bytes,
        "(512, 512, 32)": numpy.prod((512, 512, 32)) * 4 * ureg.bytes,
        "(1024, 1024, 256)": numpy.prod((1024, 1024, 256)) * 4 * ureg.bytes,
    }

    results_unique = pandas.read_csv("results-unique.csv", sep="\t")

    results_unique["shape_ind"] = results_unique["shape"].map(shape_index_mapping)
    results_unique["shape_ind_r"] = results_unique["shape_ind"].apply(lambda x: format(x.to_compact(), "~.0f"))
    results_unique["memory_ref"] = results_unique["shape_ind"].apply(lambda x: x.magnitude)
    results_unique["mem_max_norm"] = results_unique["mem_max"] / results_unique["memory_ref"]

    p = sns.lineplot(x="shape_ind_r", y="t", hue="method", data=results_unique)
    p.set(yscale="log")
    fig = p.get_figure()
    fig.savefig("unique-runningtime.png")

    p.clear()

    p = sns.lineplot(x="shape_ind_r", y="mem_max_norm", hue="method", data=results_unique)
    p.set(yscale="log")
    fig = p.get_figure()
    fig.savefig("unique-max-memory.png")

    p.clear()

    # normalize data
    # method    shape   t   py_version  npy_version pandas_version  vigra_version   host    platform    mem_min mem_max
    np_unique = results_unique[results_unique["method"] == "numpy.unique(data)"]

    normalization_factors = {}

    for _, row in np_unique.iterrows():
        key = (
            row["host"],
            row["platform"],
            row["shape"],
            row["py_version"],
            row["npy_version"],
            row["pandas_version"],
            row["vigra_version"],
        )
        normalization_factors[key] = row["t"]

    def normalize(row):
        key = (
            row["host"],
            row["platform"],
            row["shape"],
            row["py_version"],
            row["npy_version"],
            row["pandas_version"],
            row["vigra_version"],
        )
        return row["t"] / normalization_factors[key]

    results_unique["normalized_t"] = results_unique.apply(normalize, axis=1)

    p = sns.lineplot(x="shape_ind_r", y="normalized_t", hue="method", data=results_unique)
    p.set(yscale="log")
    fig = p.get_figure()
    fig.savefig("unique-runningtime-normalized.png")

    p.clear()

    results_bincount = pandas.read_csv("results-bincount.csv", sep="\t")

    results_bincount["shape_ind"] = results_bincount["shape"].map(shape_index_mapping)

    results_bincount["shape_ind_r"] = results_bincount["shape_ind"].apply(lambda x: format(x.to_compact(), "~.0f"))
    results_bincount["memory_ref"] = results_bincount["shape_ind"].apply(lambda x: x.magnitude)
    results_bincount["mem_max_norm"] = results_bincount["mem_max"] / results_bincount["memory_ref"]

    p = sns.lineplot(x="shape_ind_r", y="t", hue="method", data=results_bincount)
    p.set(yscale="log")
    fig = p.get_figure()
    fig.savefig("bincount-runningtime.png")

    p.clear()

    p = sns.lineplot(x="shape_ind_r", y="mem_max_norm", hue="method", data=results_bincount)
    p.set(yscale="log")
    fig = p.get_figure()
    fig.savefig("bincount-max-memory.png")

    p.clear()

    # normalize data
    # method    shape   t   py_version  npy_version pandas_version  vigra_version   host    platform    mem_min mem_max
    np_bincount = results_bincount[results_bincount["method"] == "numpy.bincount(data.reshape(-1))"]

    normalization_factors = {}

    for _, row in np_bincount.iterrows():
        key = (
            row["host"],
            row["platform"],
            row["shape"],
            row["py_version"],
            row["npy_version"],
            row["pandas_version"],
            row["vigra_version"],
        )
        normalization_factors[key] = row["t"]

    def normalize(row):
        key = (
            row["host"],
            row["platform"],
            row["shape"],
            row["py_version"],
            row["npy_version"],
            row["pandas_version"],
            row["vigra_version"],
        )
        return row["t"] / normalization_factors[key]

    results_bincount["normalized_t"] = results_bincount.apply(normalize, axis=1)

    p = sns.lineplot(x="shape_ind_r", y="normalized_t", hue="method", data=results_bincount)
    p.set(yscale="log")
    fig = p.get_figure()
    fig.savefig("bincount-runningtime-normalized.png")

    p.clear()


def main():
    template = Template(Path("Results.md.in").read_text())
    gen_plots()
    n_hosts = len(pandas.read_csv("results-unique.csv", sep="\t")["host"].unique())

    tz = pytz.timezone("Europe/Berlin")
    now = datetime.datetime.now(tz)

    # render
    rendered = template.render(now=now.strftime(timeformat), n_hosts=n_hosts)

    Path("Results.md").write_text(rendered)


if __name__ == "__main__":
    main()

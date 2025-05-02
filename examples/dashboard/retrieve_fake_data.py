import pandas as pd


def load_results() -> list[pd.DataFrame]:
    results = []

    # In[39]:

    metadata = dict(
        instance="p01",
        config=dict(
            solver="CpSat",
            params=dict(timeout=1000, parallel=True, free_search=False),
            name="cpsat-0",
        ),
        status="optimal",
    )
    timestamps = pd.DatetimeIndex(
        [
            "2018-01-01 00:00:00",
            "2018-01-01 00:00:02",
            "2018-01-01 00:00:01",
            "2018-01-01 00:00:03",
            "2018-01-01 00:00:04",
        ]
    )
    data = dict(obj=[10, 7, 8, 6.5, 6.2], bound=[0, 4, 2, 6, 6.1])
    df = pd.DataFrame(data, index=timestamps)
    df.attrs = metadata  # spectial attribute to store metadata
    results.append(df)

    # In[42]:

    metadata = dict(
        instance="p01",
        config=dict(
            solver="gurobi",
            params=dict(timeout=1000, parallel=True, free_search=True),
            name="gurobi-0",
        ),
        status="optimal",
    )
    timestamps = pd.to_timedelta(range(4), unit="s")
    data = dict(obj=[9.5, 8.8, 7.5, 6.5], bound=[0, 2, 4, 6])
    df = pd.DataFrame(data, index=timestamps)
    df.attrs = metadata  # spectial attribute to store metadata
    results.append(df)

    # In[43]:

    metadata = dict(
        instance="p01",
        config=dict(
            solver="CpSat",
            params=dict(timeout=1000, parallel=True, free_search=True),
            name="cpsat-1",
        ),
        status="unknown",
    )
    timestamps = pd.to_timedelta(range(0, 4, 2), unit="s")
    data = dict(obj=[10, 7.5], bound=[0, 3])
    df = pd.DataFrame(data, index=timestamps)
    df.attrs = metadata  # spectial attribute to store metadata
    results.append(df)

    # In[44]:

    metadata = dict(
        instance="p02",
        config=dict(
            solver="gurobi",
            params=dict(timeout=1000, parallel=True, free_search=True),
        ),
        status="optimal",
    )
    timestamps = pd.to_timedelta(range(0, 4, 2), unit="s")
    data = dict(obj=[12, 5], bound=[0, 3])
    df = pd.DataFrame(data, index=timestamps)
    df.attrs = metadata  # spectial attribute to store metadata
    results.append(df)

    return results

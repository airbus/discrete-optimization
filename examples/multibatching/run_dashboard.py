from discrete_optimization.generic_tools.dashboard import Dashboard
from discrete_optimization.generic_tools.study import Hdf5Database

study_name = "multibatching-study"

# retrieve data
with Hdf5Database(f"{study_name}.h5") as database:
    # results = database.load_results()
    results = database.load_results()
    l = []
    for r in results:
        if "fit" in r.columns and len(r.fit) > 0:
            l.append(r)
    results = l

# with Hdf5Database(f"sm-study-with-ls-and-greedy.h5") as database:
#    results_2 = database.load_results()
#    results.extend(results_2)


# launch dashboard with this data
app = Dashboard(results=results)
app.run()

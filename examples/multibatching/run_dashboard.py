from discrete_optimization.generic_tools.dashboard import Dashboard
from discrete_optimization.generic_tools.study import Hdf5Database


def run():
    study_name = "multibatching-study"

    # retrieve data
    with Hdf5Database(f"{study_name}.h5") as database:
        results = database.load_results()
        print(f"Loaded {len(results)} results from database")

        # Filter results that have fitness data
        l = []
        for i, r in enumerate(results):
            if "fit" in r.columns and len(r) > 0:
                l.append(r)
                print(
                    f"Result {i}: {len(r)} data points, best fit={r['fit'].min():.2f}"
                )
            else:
                print(f"Result {i}: No fitness data (columns={r.columns.tolist()})")
        results = l

        print(f"\nFiltered to {len(results)} results with fitness data")

    # launch dashboard with this data
    if results:
        app = Dashboard(results=results)
        app.run()
    else:
        print("No results with fitness data to display!")

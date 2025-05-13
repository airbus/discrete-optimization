from discrete_optimization.generic_tools.dashboard import Dashboard
from discrete_optimization.generic_tools.study import Hdf5Database

study_name = "Coloring-Study-0"

# retrieve data
with Hdf5Database(f"{study_name}.h5") as database:
    results = database.load_results()


# launch dashboard with this data
app = Dashboard(results=results)
app.run()

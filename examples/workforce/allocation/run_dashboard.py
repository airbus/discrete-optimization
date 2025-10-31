#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.dashboard import Dashboard
from discrete_optimization.generic_tools.study import Hdf5Database

study_name = "allocation-study-0-"

# retrieve data
with Hdf5Database(f"{study_name}.h5") as database:
    results = database.load_results()


# launch dashboard with this data
app = Dashboard(results=results)
app.run()

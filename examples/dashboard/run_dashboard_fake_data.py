#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from retrieve_fake_data import load_results

from discrete_optimization.generic_tools.dashboard.dashboard import Dashboard

if __name__ == "__main__":
    # retrieve data
    results = load_results()

    # launch dashboard with this data
    app = Dashboard(results=results)
    app.run()

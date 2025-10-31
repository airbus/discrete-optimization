from retrieve_fake_data import load_results

from discrete_optimization.generic_tools.dashboard.dashboard import Dashboard

if __name__ == "__main__":
    # retrieve data
    results = load_results()

    # launch dashboard with this data
    app = Dashboard(results=results)
    app.run()

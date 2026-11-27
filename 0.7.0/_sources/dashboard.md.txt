
# Dashboard

If you want to compare several solvers on a bunch of problem instances, we have you cover with the discrete-optimization dashboard!

This tool is used to summarize in a few graphs your experiments.
The following steps are necessary:
- Create a study by running different solver *configs* on several problem *instances*.
- Store time-indexed metrics for each experiment like d-o fitness, internal model objective, bound, gap, ...,
  with the necessary metadata (solver config, problem instance, solver final status).
- Retrieve this data as a list of pandas dataframes.
- Launch the dashboard to visualize

We enter into details in the next sections.

![dashboard in action](dashboard.gif)

## Prerequisites

For the dashboard to run you need to install "dash", "plotly", and "dash_bootstrap_components" libraries.
If you want to make use of the hdf5 database presented below to store the data, you will also need "pytables".
All these libraries can be installed by choosing the extra "dashboard" when installing discrete-optimization:
```shell
pip install discrete-optimization[dashboard]
```


## Launching a d-o study

We call here a study a set of experiments, where each experiment is the choice of
- a solver config (class and hyperparameters)
- a problem instance to solve

### Experiment
Each experiment will stores some metadata plus some timeseries.

The d-o libary introduces in that purpose several classes:
- [`SolverConfig`](api/discrete_optimization.generic_tools.study.rst#discrete_optimization.generic_tools.study.experiment.SolverConfig):
storing the solver class and the keyword arguments used by its constructor, `init_model()`, and `solve()`.
- [`Experiment`](api/discrete_optimization.generic_tools.study.rst#discrete_optimization.generic_tools.study.experiment.Experiment):
stores the solver config, the instance id, the solver status and the metrics dataframe. You can (and you should) associate a name to
each solver config (for an easier understanding of the dashboard). The easiest way to generate the experiment is by using its alternate constructor
`Experiment.from_solver_config()`.

```python
import pandas as pd

from discrete_optimization.generic_rcpsp_tools.solvers.ls import (
    LsGenericRcpspSolver,
    LsSolverType,
)
from discrete_optimization.generic_tools.study import (
    Experiment,
    SolverConfig,
)

xp_id = 0  # should be unique across a given study
instance = "j1201_10.sm"  # identifier for the problem instance
config_name = "hc"
solver_config = SolverConfig(
    cls=LsGenericRcpspSolver,
    kwargs=dict(ls_solver=LsSolverType.HC, nb_iteration_max=10000),
)

# solve + metrics retrieval as pandas.DataFrame
problem = ... # e.g. using a filepath based on instance identifier
solver = solver_config.cls(problem, **solver_config.kwargs)
solver.init_model(**solver_config.kwargs)
solver.solve(**solver_config.kwargs)
status = solver.status_solver
reason = ""  # string used to store the reason of an error during solve
metrics = pd.DataFrame(...)  # see next section how to retrieve metrics

xp = Experiment.from_solver_config(
    xp_id=xp_id,
    instance=instance,
    config_name=config_name,
    solver_config=solver_config,
    metrics=metrics,
    status=status,
    reason=reason,
)
```

### Stats retriever callbacks
You can make use of the stats retrievers callbacks to store the metrics timeseries, i.e. `BasicStatsCallback` or
`StatsWithBoundsCallback` (if you are using a solver implementing the mixin `BoundsProviderMixin` like cpsat solvers).
They both have the method `get_df_metrics()` to retrieve the stored metrics (fitness, best objective bound, ...) as a pandas
DataFrame.

```python
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
    StatsWithBoundsCallback,
)

stats_cb = BasicStatsCallback()  # or StatsWithBoundsCallback() if solver provides internal objective value and bound?
result_store = solver.solve(
    callbacks=[
        stats_cb,
    ],
    **solver_config.kwargs,
)
metrics = stats_cb.get_df_metrics()
```


### Database
In order to store each experiment, a dedicated database [`Hdf5Database`](api/discrete_optimization.generic_tools.study.rst#discrete_optimization.generic_tools.study.database.Hdf5Database)
has been implemented making use of hdf5 format.

It exposes several methods, in particular:
- `get_new_experiment_id()`: which allows to get a unique id for each experiment to be stored
- `store(xp)`: which stores a given experiment
- `load_results()`: which returns the list of experiments stored so far as a list of pandas dataframes whith their metadata stored
  in their `attrs` dictionary attribute. This is the format which will be needed later by the dashboard.

To avoid locking the database, it is best practice to
- use it inside a `with` statement (to be sure to close it even in case of errors)
- to open it just when needed (at the end of the experiment to store it)

```python
with Hdf5Database(database_filepath) as database:
    xp_id = database.get_new_experiment_id()
    xp = Experiment.from_solver_config(
        xp_id=xp_id,
        instance=instance,
        config_name=config_name,
        solver_config=solver_config,
        metrics=metrics,
        status=status,
        reason=reason,
    )
    database.store(xp)
```

or
```python
with Hdf5Database(database_filepath) as database:
    results = database.load_results()
```

### Full example

Putting all that together, we got an example of such a study on rcpsp problems
which makes use of the database, the stats callbacks and the experiments related classes:

```{literalinclude} ../../examples/dashboard/run_rcpsp_study.py
:caption: [examples/dashboard/run_rcpsp_study.py](https://github.com/airbus/discrete-optimization/blob/master/examples/dashboard/run_rcpsp_study.py)
```


## Launching the d-o dashboard

Once your study is done (or at least some results have been stored), you just have to
- retrieve your results
- initialize a `Dashboard` object with it
- run the dashboard

It gives:
```{literalinclude} ../../examples/dashboard/run_dashboard_rcpsp_study.py
:caption: [examples/dashboard/run_dashboard_rcpsp_study.py](https://github.com/airbus/discrete-optimization/blob/master/examples/dashboard/run_dashboard_rcpsp_study.py)
```

By default, the dashboard will then be available at http://127.0.0.1:8050/.

You will find on a left panel the different filters:
- solver configs
- instances, with some aliases:
  - @all: select all the instances
  - @withsol: select all instances for which each selected config has found at least one solution
- metric to show
- clip at: to avoid outliers, the data whose absolute value is above it are removed
- on-off buttons for
  - time in log-scale
  - transposing the graph (solved instances graph only)
  - spectifying if the metric is supposed to be minimized or maximized the metric (aggregated ranks table only)

On the main panel, you have several tabs corresponding to graph type, or tables:
- "Metric evolution": the curve for each experiment of the chosen metric
- "Metric aggregation along instances": the metric is aggregated on selected instances.
  A point is drawn for a given time on the curve of a given config,
  only if at least a point occurred before for each filtered instances.
  The table below gives the last point of each curve (so the aggregated metric at solve end of each experiment)
  + number of experiments by config + number of experiments without solution (usually because of timeout)
- "Nb of solved instances": the graph shows the time evolution of the number of solved instances (i.e. whose solver status is "optimal")
  On the graph the % is relative to the total number of experiments done for the solver config. On the table below, you get
  this total number of experiments versus the number of experiments finishing with solver status "optimal".
  A switch "w/o proof" allows to consider an experiment solved as soon as the fitness found is the same as another experiment
  with the same instance and for which the solver has a status "optimal".
- "Solvers competition": aggregated config rank along instances, aggregated distance to best metric, aggregated convergence time.
- "Config explorer": displays the solver class and hyperparameters corresponding to each solver config name.
- "Experiment data": displays the raw timeseries for each experiment
- "Empty experiments": list the experiments with no data and the potential reason (timeout or raised error)

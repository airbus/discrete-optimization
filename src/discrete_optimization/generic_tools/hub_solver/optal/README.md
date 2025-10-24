# Optal for discrete-optimization

We briefly explain how to make use of Optal wrappers coded in d-o.

## How to install Optal?

WARNING: You need to install it at the correct location, i.e. where the mts optal models are stored (under the discrete-optimization installation directory,
which may be different from your local repository clone).


- install npm + node >= 20.
  See https://docs.npmjs.com/downloading-and-installing-node-js-and-npm for installation process.
- go to the directory where the optal models are stored
  ```shell
  optal_dir=$(python -c "
  import discrete_optimization.generic_tools.hub_solver.optal as optal
  import os
  print(os.path.dirname(optal.__file__))
  ")
  cd ${optal_dir}
  ```
- [install OptalCP](https://optalcp.com/docs/installation#install-optalcp):
  - either the free preview version
    ```shell
    npm install 'scheduleopt/optalcp-js-bin-preview#latest'
    ```
  - or the licensed one
    ```shell
    npm install 'scheduleopt/optalcp-js-bin#latest'
    ```

**Note**:
    If using the preview version of optal, make sure to use the option `do_not_retrieve_solutions=True` in `optal_solver.solve()`,
    where `optal_solver` is one of the optal wrapper included in d-o.

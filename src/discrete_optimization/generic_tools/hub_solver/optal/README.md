# Optal for discrete-optimization

We briefly explain how to make use of Optal wrappers coded in d-o.

## How to install Optal?

See http://dev.vilim.eu/docs/Quick%20Start/editions/ for pip commands according to the edition you have a license for.

In a nutshell:

- preview edition (free, no retrievable solutions, but still objective and bound values):
  ```shell
  pip install git+https://github.com/ScheduleOpt/optalcp-py-bin-preview@latest
  ```

- academic edition
  ```shell
  pip install git+https://github.com/ScheduleOpt/optalcp-py-bin-academic@latest
  ```

- full edition
  ```shell
  pip install git+https://github.com/ScheduleOpt/optalcp-py-bin@latest
  ```

Notes:
- you can also use `uv pip` if already using uv.
- when on the d-o repo, the preview version is in the dependency group "optalcp-preview",
  so you can also type
  ```shell
  pip install --group optalcp-preview
  ```

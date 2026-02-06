# TODO 

- Finish new Longitudinal Dataset (get item needs to group all timepoints for given case, right now it does not do it)
- Configure logger levels at application level, not in library modules. Options: (1) `logging.basicConfig(level=logging.INFO)` in entry point, (2) `logging.getLogger("lesion_tracking").setLevel(logging.DEBUG)` for package-specific config, or (3) use `DEBUG=1` env var

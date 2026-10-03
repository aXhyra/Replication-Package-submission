# Testing Human-Centric Systems: Personality Matters

This package replicates the results of **Testing Human-Centric Systems: Personality Matters**.
It contains the healthiness of the ride-hailing Human-Centric System (HCS) of San Francisco over a grid of 15 events × 8 personality populations = 120 runs (**RQ1**), the effect of two prevention strategies on the HCS failures (**RQ2**), and the attribution of the HCS failures to the event, the personality and their interaction (**RQ3**).


## Structure

This replication package includes 3 notebooks corresponding to the three research questions (RQ1, RQ2, RQ3) addressed in the paper, a utility notebook, and the datasets, scripts, and outputs necessary to replicate the results:

- `datasets/raw/<event>/ocean/`: the 120 runs without prevention, one CSV per run: one row per simulated second, 56 runtime metrics.
- `datasets/prevention/<strategy>/<event>/ocean/`: the runs re-executed with a prevention strategy (`call_freelance_drivers`: 33 runs, `enforce_surge_limit`: 57 runs).
- `utils.ipynb`: contains utility functions for computing the results of the notebooks.
- `rq1-healthiness.ipynb`: **RQ1**  results.
- `rq2-prevention.ipynb`: **RQ2** results.
- `rq3-analysis.ipynb`: **RQ3** results.
- `validation/`: the validation of the model's accuracy, i.e., the fidelity of the `normal` event with the baseline reference US-population to the source traffic and ride-hailing data of San Francisco:
  - `validation.ipynb`: the validation notebook.
  - `datasets/simulation/normal/ocean/`: the dedicated validation run, with the per-second CSV (`sf_normal_ocean_dr_general_pass_general.csv`), the SUMO per-step summary (`summary.xml.gz`, for the vehicles inserted in each hour) and the ride-hailing fleet (`sf_tnc_fleet_25110921_25111101.rou.xml.gz`, to separate the traffic vehicles from the ride-hailing ones), both gzipped.
  - `datasets/source/`: the source San Francisco data (SFCTA ride-hailing pickups/dropoffs per TAZ, day of week and hour; SFMTA-based hourly traffic counts of the simulated window).
  - `fidelity_out/`: the outputs of the validation notebook (whole-period fidelity, hourly source and simulated counts, hourly profiles plot).
- `build_dashboard.py`, `dashboard_template.html`: the script and template to build `dashboard.html` from the notebooks' outputs.
- `dashboard.html`: the interactive dashboard of the results, to open in a browser.
- `ocean.py`: the OCEAN personality model of the simulator, which drives the decisions and ratings of the simulated drivers and passengers (standard library only; `python3.10 ocean.py` runs its self-test).
- `prompt.md`: the prompt used to generate `ocean.py`.
- `analysis_out/`: the raw outputs of the notebooks.
- `requirements.txt`: Python dependencies.


## Experimental Setup

- **Events**: `normal` (no event), five single events (*Underground Alarm* (UA), *Wildcat Strike* (WS), *Flash Mob* (FM), *Long Rides* (LR), *Boycott TNCs* (BT)), and nine compound events that inject two of them at the same time.
- **Personality populations**: a passenger personality (*general*, *students*, *managers*, *bank tellers*) combined with a driver personality (*freelance drivers*, *professional drivers*). The baseline general US-population (freelance drivers, general passengers) is called `baseline` in the files; the populations with professional drivers carry the `_drivers` suffix.
- In every run the event and the personality are injected at 19:00 of the second day (timestamp 79,200 s); all runs share the same random seeds and clock, so they can be compared at any timestamp.


## How-to-run instructions

1. Open project (`cd path/to/this/project`), create a virtual environment with Python >=3.10 (`python3.10 -m venv .venv`) and activate it (`source .venv/bin/activate`).
2. Install requirements from `requirements.txt` file (`pip install -r requirements.txt`).
3. Run the notebooks in order.
Each can be opened in Jupyter (`jupyter lab`) and run top to bottom, or executed headless:
```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 rq1-healthiness.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 rq2-prevention.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 rq3-analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 validation/validation.ipynb
```
4. Build the dashboard with `python3.10 build_dashboard.py`.
5. Open `dashboard.html` in a browser to interact with the results.

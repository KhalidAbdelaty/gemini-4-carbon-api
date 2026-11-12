# PayGuard: a Gemini 4 Carbon API deployment audit

Gemini 4 Carbon audits a fictional fraud classifier against three deployment rules through the
Interactions API and three read-only Python tools. A separate evaluator and a sealed holdout window
check the frozen verdict. All data is synthetic.

## Setup

Python 3.11 or newer (`google-genai` 3.1.0 requires it) and a Gemini API key on a paid-tier project.
Gemini 4 Carbon has no free tier.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a file named `.env` in this folder with one line: `GEMINI_API_KEY=your-key`. The code loads only
that variable and removes `GOOGLE_API_KEY` from the process, because the SDK prefers `GOOGLE_API_KEY`
when both are set.

## Run

```bash
python generate_data.py            # train, validation, and data/sealed/holdout.csv (fixed seed)
python prepare_model.py            # fits the classifier on train, scores the later windows
python evaluator.py --fixture      # exhaustive check of the validation window, no API call
python evaluator.py --self-test    # configurations the evaluator must reject
python first_request.py            # one Gemini 4 Carbon request (billed)
python agent.py                    # medium audit, evaluator, holdout once, clean high rerun (billed)
python agent.py --level high --attempt-only   # one more independent attempt (billed)
streamlit run app_streamlit.py
```

`agent.py` writes each experiment to `runs/`. In the app, "Show the latest recorded run" reads the newest
file there and makes no API call. Until you save a run of your own, it shows the tutorial's first attempt
from `recorded_runs/`.

## Recorded results

`recorded_runs/` holds the six attempts from the tutorial, run on November 11, 2026 with
`google-genai` 3.1.0: every tool call and result, token usage per turn, the frozen report, the evaluator's
score, and the holdout result for the first `medium` attempt. No API key is stored. To see the
medium-versus-high summary without an API call:

```bash
python compare_levels.py
```

## Reproducing the six-attempt comparison

Each command below starts from a clean state, so no attempt sees another one's history. Together they
make three attempts per thinking level (billed, about $2 at the introductory rate):

```bash
python agent.py                                 # medium attempt 1, evaluator, holdout, high attempt 1
python agent.py --level medium --attempt-only   # medium attempt 2
python agent.py --level medium --attempt-only   # medium attempt 3
python agent.py --level high --attempt-only     # high attempt 2
python agent.py --level high --attempt-only     # high attempt 3
python compare_levels.py --folder runs          # summary of your own attempts
```

Model output varies from run to run, so expect your tool calls, tokens, and costs to differ from the
recorded ones. The data, tools, limits, and evaluator stay the same.

## Files

- `config.py`: seeds, windows, the policy, the thinking levels, application limits, and prices
- `generate_data.py`, `prepare_model.py`: the synthetic data and the scikit-learn classifier
- `tools.py`: the three tools the agent may request; they read the validation window only
- `agent.py`: the Interactions API loop, limits, final structured report, and cost
- `evaluator.py`: the independent check, outside the agent's tool list
- `holdout_check.py`: applies a frozen recommendation to the holdout, once
- `app_streamlit.py`: the experiment UI
- `compare_levels.py`: summarizes saved attempts by thinking level
- `recorded_runs/`: the six attempts from the tutorial

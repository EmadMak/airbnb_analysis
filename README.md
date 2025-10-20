## Airbnb Reviews Analysis

This project performs data cleaning, preprocessing, and visualization on Airbnb app reviews (from Google Play, App Store, and Trustpilot>
It uses Python + uv for environment management and marimo for interactive notebook analysis.

## Getting Started

### 1. Install uv

If you haven’t installed uv yet:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or via pip:
```bash
pip install uv
```

### 2. Set up the environment
All dependencies (including marimo, pandas, matplotlib, and seaborn) are managed by uv.
```bash
uv sync
```
This will install everything declared in your pyproject.toml.

### 3. Run the interactive marimo notebook
To start the marimo interface:
```bash
uv run marimo run notebooks/analysis.py
```
This launches an interactive web app at http://localhost:2718

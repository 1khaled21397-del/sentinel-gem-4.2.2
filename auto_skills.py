name: Apply Fixes & Verify

on:
  push:
    branches: [main, develop]
    paths:
      - '**.py'
      - 'requirements.txt'
      - 'sentinel_config.json'
  pull_request:
    branches: [main]
  workflow_dispatch:

env:
  PYTHON_VERSION: '3.11'

jobs:
  verify:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Cache pip dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Verify all module imports
        run: |
          python -c "import data_engine; print('✅ data_engine')"
          python -c "import technical_analysis; print('✅ technical_analysis')"
          python -c "import sentiment_scraper; print('✅ sentiment_scraper')"
          python -c "import ml_forecast; print('✅ ml_forecast')"
          python -c "import gap_predictor; print('✅ gap_predictor')"
          python -c "import auto_skills; print('✅ auto_skills')"
          python -c "import overnight_alpha; print('✅ overnight_alpha')"
          python -c "import backtest_engine; print('✅ backtest_engine')"
          python -c "import regime_detector; print('✅ regime_detector')"
          python -c "import hedge_engine; print('✅ hedge_engine')"

      - name: Verify data_engine URL fix
        env:
          EODHD_API_KEY: ${{ secrets.EODHD_API_KEY }}
        run: |
          python -c "
          from data_engine import _build_url
          url = _build_url('COMI.EGX', 'EGX', 'd', 500)
          assert 'COMI.EGX.EGX' not in url, f'DOUBLE SUFFIX BUG: {url}'
          assert 'COMI.EGX' in url, f'MISSING SUFFIX: {url}'
          print('✅ URL fix verified:', url.replace(os.getenv('EODHD_API_KEY', ''), '***'))
          "

      - name: Smoke test with synthetic data
        run: |
          python -c "
          from data_engine import _generate_synthetic_data
          df = _generate_synthetic_data('TEST.EGX', 100)
          assert len(df) == 100, f'Expected 100 bars, got {len(df)}'
          assert all(d.weekday() in [0,1,2,3,6] for d in df['date']), 'Invalid trading days'
          print('✅ Synthetic data calendar correct (Sun-Thu)')
          "

      - name: Verify sentiment_scraper dual AI wiring
        run: |
          python -c "
          from sentiment_scraper import SentimentResult
          from dataclasses import fields
          field_names = [f.name for f in fields(SentimentResult)]
          assert 'ai_source' in field_names, 'Missing ai_source field'
          assert 'ai_score' in field_names, 'Missing ai_score field'
          assert 'ai_confidence' in field_names, 'Missing ai_confidence field'
          print('✅ sentiment_scraper v4.2.1+ verified')
          "

      - name: Test config JSON validity
        run: |
          python -c "
          import json
          with open('sentinel_config.json') as f:
              cfg = json.load(f)
          assert cfg.get('version') == '4.2', f"Expected version 4.2, got {cfg.get('version')}"
          assert 'dual_ai' in cfg.get('sentiment', {}), 'Missing dual_ai config'
          print('✅ sentinel_config.json valid')
          "

      - name: Report status
        if: always()
        run: |
          echo "All verification steps complete."

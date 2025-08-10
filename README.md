# AI‑Assisted Risk Screening

An end‑to‑end prototype that combines a modern Angular frontend, a Django API, and a PyTorch model served via Flask to predict health risks (stroke today, diabetes next). Built to demonstrate full‑stack engineering, ML integration, and pragmatic product delivery.

## Overview
- Frontend: Angular 15 SPA for patient/doctor flows
- Backend: Django 4 with custom user model for authentication
- ML service: Flask app hosting a lightweight PyTorch model for stroke‑risk inference
- Data: Public healthcare datasets included for experimentation/training

## Architecture
```
Angular (SPA)
  └── calls Django API (auth, user flows)
            └── MySQL (dev/prod) or SQLite (local)
  └── calls Flask + PyTorch service (/stroke inference)
```

## Tech Stack
- Frontend: Angular 15, TypeScript, RxJS
- Backend: Django 4.1, MySQL (configurable), custom auth model
- ML/Serving: Python 3.8+, Flask, PyTorch
- Tooling: pip/venv, npm, Angular CLI



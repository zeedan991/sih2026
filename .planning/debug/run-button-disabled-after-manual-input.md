# Debug session: run button disabled after manual input

**Status:** Resolved
**Observed:** 2026-09-07

## Symptoms

- Expected: after selecting a held-out record, or choosing manual entry and completing valid fields, **Run hybrid analysis** is enabled and predicts from the current values.
- Actual: after moving between data-source options, the selected record could be visible while the Run button remained disabled. Manual entry also depended on a separate **Use entered values** action, and edits made after that action could leave an older `state.patient` snapshot behind.
- Browser/API errors: none. `/health` reported the model service ready.

## Reproduction and evidence

1. Load the app with ready models: the initial catalog record can run successfully.
2. Change **Data source** to manual: `frontend/app.js` cleared `state.patient` and explicitly disabled the Run button.
3. Change back to the held-out catalog: `renderSelectedPatient(...)` restored `state.patient` and the visible record, but did not restore the button state.
4. In manual mode, values were copied into `state.patient` only when **Use entered values** was clicked. Subsequent edits did not update that copied array.

This eliminated model readiness and `/predict` as causes: the same live backend returned a complete prediction when the initial enabled catalog button was clicked.

## Root cause

Button availability was mutated independently in several event handlers instead of being derived from the authoritative state (`state.isBusy` and `state.patient`). The manual form also used a one-time copied snapshot rather than synchronizing edits or re-reading fields at prediction time.

## Fix

- Added `syncPredictButtonAvailability()` as the single rule for Run-button availability.
- Catalog rendering now synchronizes availability after restoring the active patient.
- Manual mode validates and applies the current form immediately and on every input event.
- `runPrediction()` re-reads and validates manual fields immediately before sending `/predict`, preventing stale values.
- Empty manual fields are invalid instead of being silently converted by `Number("")` to zero.
- CSV success/failure uses the same availability rule.

## Regression verification

- Static UI contract test asserts catalog selection re-synchronizes the button, manual values are re-read at prediction time, empty values are rejected, and edits update the active manual record.
- Browser verification covers catalog → manual → edit → Run and manual → catalog → Run against the live local service.
- Browser result: both paths completed prediction successfully; the edited gender value changed to `1.0000` in the manual preview before the `MANUAL-INPUT` result rendered.
- Full suite: `94 passed, 5 expected dependency warnings` in 19.89 seconds.

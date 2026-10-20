# Tello Battery, Flight, and Camera Examples

**What does the preserved script actually send to the drone?**

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md) · [Collection overview](../../README.md)

---

## Question and approach

The basic script connects, checks battery, and takes off only when the value is at least 20%. The camera example starts a stream, previews, takes off, and lands. Inspect the actual control flow before any hardware use.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [(1.1.1)first_flight.py](%281.1.1%29first_flight.py) | Top-level experiment; imports: djitellopy, time |
| [(1.1.2)Flight with camera.py](%281.1.2%29Flight%20with%20camera.py) | Top-level experiment; imports: cv2, djitellopy, time |
| [test_drone.py](test_drone.py) | Top-level experiment; imports: djitellopy |

## Inputs, tools, and outputs

Observed import roots: `cv2`, `djitellopy`, `time`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

Top-level imports/execution can trigger connection and flight. The existing battery branch is not a complete safety system. No hardware run was performed.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../../docs/REPAIRS.md).

---

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md)

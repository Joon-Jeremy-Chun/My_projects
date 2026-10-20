# KDCA Historical Data Collection

**How are historical COVID-19 API responses retrieved and organized?**

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md) · [Collection overview](../../README.md)

---

## Question and approach

The simple request specifies one date/region; the multi-date version loops over dates. Other scripts merge frames, correct selected points, and form age-group datasets.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [COIVD-19_Korea_age_groups.py](COIVD-19_Korea_age_groups.py) | Top-level experiment; imports: datetime, pandas, requests, xml.etree.ElementTree |
| [Multi date group by_(Korean).py](Multi%20date%20group%20by_%28Korean%29.py) | Top-level experiment; imports: datetime, pandas, requests, xml.etree.ElementTree |
| [code sample.py](code%20sample.py) | Top-level experiment; imports: pandas, requests, xml.etree.ElementTree |
| [correcting some data points(last year).py](correcting%20some%20data%20points%28last%20year%29.py) | Defined helpers: `process_and_merge_files_for_age_group` |
| [merged the dataFrames.py](merged%20the%20dataFrames.py) | Defined helpers: `process_and_merge_files_for_age_group` |
| [multiple API request.py](multiple%20API%20request.py) | Defined helpers: `fetch_data` |
| [simple API request.py](simple%20API%20request.py) | Top-level experiment; imports: requests |

## Inputs, tools, and outputs

Observed import roots: `datetime`, `os`, `pandas`, `requests`, `xml`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

Requests have external effects. No API credential was tested and no current data were collected. Historical corrections need their provenance checked before using snapshots for new analyses.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../../docs/REPAIRS.md).

---

[Portfolio](../../../README.md) · [Evidence & results](../../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../../docs/REPAIRS.md) · [Reproduction](../../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../../docs/CODE_CATALOG.md)

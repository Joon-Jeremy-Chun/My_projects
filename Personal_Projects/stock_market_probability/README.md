# Market Direction and Probability Studies in R

**How are returns, runs, and conditional market summaries explored?**

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md) · [Collection overview](../README.md)

---

## Question and approach

Inspect quantmod data acquisition, return-sign labeling, lagged run counts, and the selected denominator. Other files explore volatility, P/E bands, dividends, and coin-toss probability examples.



## File-by-file reading guide

| Source | What to inspect |
|---|---|
| [Drow the S&P Bollinger bands chart.R](Drow%20the%20S%26P%20Bollinger%20bands%20chart.R) | Top-level experiment; imports: dygraphs, quantmod |
| [Drow the S&P chart.R](Drow%20the%20S%26P%20chart.R) | Top-level experiment; imports: dygraphs, quantmod |
| [PE ratio bands.R](PE%20ratio%20bands.R) | Top-level experiment; imports: quantmod, rvest |
| [Several markets values.R](Several%20markets%20values.R) | Top-level experiment; imports: quantmod |
| [dividend history.R](dividend%20history.R) | Top-level experiment; imports: quantmod |
| [같은면이 나올 확률분포.R](%EA%B0%99%EC%9D%80%EB%A9%B4%EC%9D%B4%20%EB%82%98%EC%98%AC%20%ED%99%95%EB%A5%A0%EB%B6%84%ED%8F%AC.R) | Top-level experiment; imports: dplyr |
| [동전이 연속해서 같은면이 나올 확률.R](%EB%8F%99%EC%A0%84%EC%9D%B4%20%EC%97%B0%EC%86%8D%ED%95%B4%EC%84%9C%20%EA%B0%99%EC%9D%80%EB%A9%B4%EC%9D%B4%20%EB%82%98%EC%98%AC%20%ED%99%95%EB%A5%A0.R) | Top-level experiment; imports: dplyr |
| [상승과 하락 폭 평균 표준편차.R](%EC%83%81%EC%8A%B9%EA%B3%BC%20%ED%95%98%EB%9D%BD%20%ED%8F%AD%20%ED%8F%89%EA%B7%A0%20%ED%91%9C%EC%A4%80%ED%8E%B8%EC%B0%A8.R) | Top-level experiment; imports: none detected |
| [상승한날 하락한날 데이터셋 저장.R](%EC%83%81%EC%8A%B9%ED%95%9C%EB%82%A0%20%ED%95%98%EB%9D%BD%ED%95%9C%EB%82%A0%20%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B%20%EC%A0%80%EC%9E%A5.R) | Top-level experiment; imports: dplyr, quantmod |
| [연속상승 상승폭 평균 표준편차.R](%EC%97%B0%EC%86%8D%EC%83%81%EC%8A%B9%20%EC%83%81%EC%8A%B9%ED%8F%AD%20%ED%8F%89%EA%B7%A0%20%ED%91%9C%EC%A4%80%ED%8E%B8%EC%B0%A8.R) | Top-level experiment; imports: dplyr, quantmod |
| [연속상승 하락폭 평균 표준편차.R](%EC%97%B0%EC%86%8D%EC%83%81%EC%8A%B9%20%ED%95%98%EB%9D%BD%ED%8F%AD%20%ED%8F%89%EA%B7%A0%20%ED%91%9C%EC%A4%80%ED%8E%B8%EC%B0%A8.R) | Top-level experiment; imports: dplyr, quantmod |
| [주식시장에서 상승과 하락 확률.R](%EC%A3%BC%EC%8B%9D%EC%8B%9C%EC%9E%A5%EC%97%90%EC%84%9C%20%EC%83%81%EC%8A%B9%EA%B3%BC%20%ED%95%98%EB%9D%BD%20%ED%99%95%EB%A5%A0.R) | Top-level experiment; imports: dplyr, quantmod |
| [주식시장에서 연속 상승과 하락 확률.R](%EC%A3%BC%EC%8B%9D%EC%8B%9C%EC%9E%A5%EC%97%90%EC%84%9C%20%EC%97%B0%EC%86%8D%20%EC%83%81%EC%8A%B9%EA%B3%BC%20%ED%95%98%EB%9D%BD%20%ED%99%95%EB%A5%A0.R) | Top-level experiment; imports: dplyr, quantmod |

## Inputs, tools, and outputs

Observed import roots: `dplyr`, `dygraphs`, `quantmod`, `rvest`. These observations are not a tested dependency lockfile.

## Interpretation and reproduction limits

Overlapping run frequencies are not automatically conditional transition probabilities. Current finance-source compatibility and any predictive trading result are unverified.

Check the actual input path, state ordering, parameter units, working directory, and output destination before a new run. New code repairs and their tested scope are listed in [the repair record](../../docs/REPAIRS.md).

---

[Portfolio](../../README.md) · [Evidence & results](../../docs/PROJECT_EVIDENCE.md) · [2026 repairs](../../docs/REPAIRS.md) · [Reproduction](../../docs/REPRODUCIBILITY.md) · [Full code catalog](../../docs/CODE_CATALOG.md)

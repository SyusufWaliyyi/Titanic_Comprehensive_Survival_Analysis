# Titanic\_Comprehensive\_Survival\_Analysis



\# RMS Titanic -- Comprehensive Survival Analysis



> End-to-end data cleaning, validation, and exploratory survival analysis of 891 Titanic passengers using Python. Covers sentinel-string detection, multi-strategy missing-value imputation, feature engineering (title extraction, family size, age banding), and ten purpose-built survival visualisations.



---



\## Table of Contents



\- \[Project Overview](#project-overview)

\- \[Dataset](#dataset)

\- \[Project Structure](#project-structure)

\- \[Pipeline Summary](#pipeline-summary)

\- \[Key Findings](#key-findings)

\- \[Tech Stack](#tech-stack)

\- \[Getting Started](#getting-started)

\- \[Notebook Walkthrough](#notebook-walkthrough)

\- \[Design Decisions](#design-decisions)

\- \[Key Insights](#key-insights)

\- \[License](#license)



---



\## Project Overview



This project performs a comprehensive survival analysis on the RMS Titanic passenger dataset. The primary goals are:



1\. \*\*Detect and handle hidden missing values\*\* stored as `\\N` sentinel strings -- invisible to standard null checks

2\. \*\*Apply context-aware imputation strategies\*\* tailored to the volume and nature of missingness in each column

3\. \*\*Engineer meaningful features\*\* including passenger title (social status proxy), family size, age bands, and fare bands

4\. \*\*Quantify survival rates\*\* across every major passenger dimension: sex, class, age, family, embarkation port, and title

5\. \*\*Surface the interplay between variables\*\* -- particularly how class amplifies the sex-based survival disparity



The project targets intermediate Python practitioners and follows standard data-science project conventions.



---



\## Dataset



| Property | Detail |

|---|---|

| File | `titanic.csv` |

| Rows | 891 passengers |

| Columns | 12 (raw) |

| Period | April 1912 (single voyage) |

| Source | Classic Kaggle / public Titanic dataset |



\### Columns



| Column | Raw Type | Cleaned Type | Description |

|---|---|---|---|

| `PassengerId` | `int64` | `int64` | Unique passenger identifier |

| `Survived` | `int64` | `int64` | Target variable: 0 = perished, 1 = survived |

| `Pclass` | `int64` | `int64` | Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd) |

| `Name` | `str` | `str` | Full name (used for title extraction) |

| `Sex` | `str` | `str` | Passenger sex |

| `Age` | `str` | `float` | Age in years (sentinel strings + 19.87% missing) |

| `SibSp` | `int64` | `int64` | Siblings / spouses aboard |

| `Parch` | `int64` | `int64` | Parents / children aboard |

| `Ticket` | `str` | `str` | Ticket number (not used in analysis) |

| `Fare` | `float64` | `float64` | Passenger fare in GBP |

| `Cabin` | `str` | dropped | Cabin number (77.1% missing -- converted to flag) |

| `Embarked` | `str` | `str` | Port of embarkation: S / C / Q |



\### Data Quality Issues Found



| Column | Issue | Volume | Strategy |

|---|---|---|---|

| All columns | `\\N` used as sentinel string for NULL -- invisible to `.isnull()` | Widespread | Replace with `np.nan` before any type conversion |

| `Age` | 19.87% missing after sentinel removal; stored as `object` | 177 / 891 | Cast to float; impute with \*\*Pclass + Sex group median\*\* |

| `Cabin` | 77.1% missing -- imputation would fabricate 3 in 4 values | 687 / 891 | Drop column; create binary `Cabin Known` flag |

| `Embarked` | 2 rows missing (0.22%) | 2 / 891 | Fill with mode (Southampton) |



---



\## Project Structure



```

titanic-survival-eda/

|

+-- titanic.csv                    # Raw data (not committed -- see note below)

+-- titanic\_analysis.ipynb         # Main analysis notebook

+-- titanic\_analysis.png           # Composite output chart

+-- README.md                      # This file

+-- requirements.txt               # Python dependencies

```



> \*\*Note:\*\* Raw data files are not committed to version control. Place `titanic.csv` in the project root before running the notebook.



---



\## Pipeline Summary



```

Raw CSV

&nbsp; |

&nbsp; +-- Stage 1: Load \& inspect

&nbsp; |     Shape, dtypes, raw unique values (catches \\N sentinel),

&nbsp; |     null audit before AND after sentinel replacement

&nbsp; |

&nbsp; +-- Stage 2: Clean \& validate

&nbsp; |     Replace \\N sentinels -> NaN

&nbsp; |     Cast Age and Fare to float

&nbsp; |     Cabin: drop + create Cabin Known binary flag

&nbsp; |     Embarked: fill 2 NaN with mode

&nbsp; |     Age: fill 177 NaN with Pclass+Sex group median

&nbsp; |     Assert: zero nulls in all critical columns

&nbsp; |

&nbsp; +-- Stage 3: Feature engineering

&nbsp; |     Family Size = SibSp + Parch + 1

&nbsp; |     Is Alone = (Family Size == 1)

&nbsp; |     Age Band (Child / Teen / Young Adult / Adult / Senior)

&nbsp; |     Fare Band (Low / Medium / High / Very High)

&nbsp; |     Title extracted from Name (Mr / Mrs / Miss / Master / Rare)

&nbsp; |

&nbsp; +-- Stage 4: EDA aggregations

&nbsp; |     Survival rates by: sex, class, age band, title,

&nbsp; |     family size, embarkation port, fare band

&nbsp; |     Sex x Class cross-tabulation

&nbsp; |

&nbsp; +-- Stage 5: Visualisations (10 charts)

&nbsp; |

&nbsp; +-- Stage 6: KPI summary

&nbsp; |

&nbsp; +-- Stage 7: Key insights

```



---



\## Key Findings



| Metric | Value |

|---|---|

| Total Passengers | 891 |

| Survived | 342 (38.4%) |

| Perished | 549 (61.6%) |

| Female Survival Rate | 74.2% |

| Male Survival Rate | 18.9% |

| 1st Class Survival Rate | 63.0% |

| 2nd Class Survival Rate | 47.3% |

| 3rd Class Survival Rate | 24.2% |

| Highest Survival Age Band | Child (58.0%) |

| Solo Traveller Rate | 60.3% |

| Avg Fare | GBP 32.20 |

| Median Age (post-imputation) | 26.0 years |



\*\*The three strongest survival predictors in this dataset:\*\*



1\. \*\*Sex\*\* -- female passengers survived at 74.2% vs 18.9% for males. The largest single gap in the data.

2\. \*\*Class\*\* -- 1st class survival (63.0%) is 2.6x higher than 3rd class (24.2%).

3\. \*\*Sex x Class interaction\*\* -- 1st class females survived at 96.8%; 3rd class males at 13.5%. The combination of both disadvantages compounded dramatically.



---



\## Tech Stack



| Library | Version | Purpose |

|---|---|---|

| `pandas` | >= 2.0 | Data loading, cleaning, aggregation |

| `numpy` | >= 1.25 | Numeric operations and assertions |

| `matplotlib` | >= 3.7 | Base plotting, boxplots, histograms, tick formatting |

| `seaborn` | >= 0.13 | Heatmap, colour palette utilities |



---



\## Getting Started



\### Prerequisites



\- Python 3.10+

\- `pip` or `conda`



\### Installation



```bash

\# 1. Clone the repo

git clone https://github.com/SyusufWaliyyi/Titanic\_Comprehensive\_Survival\_Analysis.git

cd Titanic\_Comprehensive\_Survival\_Analysis



\# 2. Create and activate a virtual environment

python -m venv .venv

source .venv/bin/activate        # macOS / Linux

.venv\\Scripts\\activate           # Windows



\# 3. Install dependencies

pip install -r requirements.txt



\# 4. Launch the notebook

jupyter lab titanic\_analysis.ipynb

```



\### `requirements.txt`



```

pandas>=2.0

numpy>=1.25

matplotlib>=3.7

seaborn>=0.13

jupyter>=1.0

```



---



\## Notebook Walkthrough



| Stage | Cells | What it does |

|---|---|---|

| \*\*0 -- Config\*\* | 1 | Imports, global constants, matplotlib/seaborn theme |

| \*\*1 -- Load \& Inspect\*\* | 4 | Read CSV, `.info()`, raw unique values (exposes `\\N`), null audit before and after |

| \*\*2 -- Clean \& Validate\*\* | 6 | Sentinel replacement, dtype fixes, Cabin flag, Embarked mode fill, Age group median imputation, assertion-based validation |

| \*\*3 -- Feature Engineering\*\* | 4 | Family Size, Is Alone, Age Band, Fare Band, Title extraction and consolidation |

| \*\*4 -- EDA Aggregations\*\* | 5 | All survival groupby summaries, cross-tabs, and descriptive stats |

| \*\*5 -- Visualisations\*\* | 10 | One chart per cell -- count, sex, class, age dist, age band, heatmap, family, fare, title, port |

| \*\*6 -- KPI Summary\*\* | 1 | Formatted metrics dictionary printed as a summary table |

| \*\*7 -- Key Insights\*\* | Markdown | Eight numbered findings with interpretation |



---



\## Design Decisions



\### Why sentinel `\\N` strings are dangerous



The raw data stores missing values as the string `\\N` rather than true `NaN`. This means `.isnull().sum()` returns zero across all columns on load -- a deceptively clean result. The issue only becomes visible by inspecting `.unique()` values per column.



The fix is to replace sentinels before any type conversion:



```python

df.replace('\\\\N', np.nan, inplace=True)

```



Running `.isnull().sum()` both before and after this replacement is documented in the notebook as an explicit before/after audit -- a pattern worth preserving in any pipeline where upstream data provenance is uncertain.



\### Age imputation -- why Pclass + Sex group median?



With 19.87% of Age values missing, the choice of imputation strategy materially affects downstream analysis.



A global median (28.0 years) would assign the same fill value to a 3rd-class male and a 1st-class female. Age correlates meaningfully with both class (wealthier passengers tended to be older) and sex (different travel patterns across demographics). Grouping by both produces contextually accurate fill values:



| Pclass | Sex | Median Age |

|---|---|---|

| 1 | female | 35.0 |

| 1 | male | 40.0 |

| 2 | female | 28.0 |

| 2 | male | 30.0 |

| 3 | female | 21.5 |

| 3 | male | 25.0 |



```python

age\_group\_median = df.groupby(\['Pclass', 'Sex'])\['Age'].transform('median')

df\['Age']        = df\['Age'].fillna(age\_group\_median)

```



\### Cabin -- binary flag instead of imputation



77.1% of Cabin values are missing -- imputing would fabricate data for three in four passengers. More importantly, the missingness is not random: cabin records exist almost exclusively for 1st and 2nd class passengers. This pattern itself is informative.



A binary `Cabin Known` flag preserves the signal without introducing false precision:



```python

df\['Cabin Known'] = df\['Cabin'].notna().astype(int)

df.drop(columns=\['Cabin'], inplace=True)

```



Cross-tabulating `Cabin Known` against `Pclass` confirms the expected pattern: nearly all known cabins belong to 1st or 2nd class passengers.



\### Embarked -- mode is appropriate here



Only 2 rows (0.22%) are missing. Southampton accounts for 72.3% of all embarkations. At this scale, mode fill introduces negligible bias and is the correct lightweight approach.



\### Title extraction as a social status proxy



Passenger names follow the format `Surname, Title. First name`. Extracting title with regex:



```python

df\['Title'] = df\['Name'].str.extract(r',\\s\*(\[^\\.]+)\\.', expand=False).str.strip()

```



Titles with fewer than 10 occurrences (Dr, Rev, Col, etc.) are consolidated into `Rare` to avoid noisy single-observation bars in visualisations. The resulting groups -- Mr, Mrs, Miss, Master, Rare -- correlate strongly with both age and sex, making Title a useful compound feature.



\### Assertion-based validation



The pipeline uses `assert` statements to enforce data contracts after each transformation:



```python

assert (df\['Boxes Shipped'] > 0).all(), 'Zero or negative box counts found'

assert df\[critical\_cols].isnull().sum().sum() == 0, 'Nulls remain in critical columns'

```



This pattern fails loudly if upstream data changes introduce new issues -- far preferable to silent errors surfacing in aggregations.



---



\## Key Insights



| # | Finding | Interpretation |

|---|---|---|

| 1 | \*\*Sex is the strongest survival predictor\*\* (74.2% female vs 18.9% male) | 'Women and children first' was enforced with striking consistency |

| 2 | \*\*Class amplifies the sex effect\*\* (1st class female: 96.8%, 3rd class male: 13.5%) | Wealth determined proximity to lifeboats and quality of crew communication |

| 3 | \*\*Children had the highest survival rate (58.0%)\*\* | Age-based priority applied alongside sex-based priority |

| 4 | \*\*Solo travellers survived less\*\* (30.4%) than small-family groups | Groups of 2-4 may have coordinated evacuation more effectively |

| 5 | \*\*Large families (5+) had very low survival rates\*\* | Coordinating evacuation for large groups in the chaos was near-impossible |

| 6 | \*\*Cherbourg passengers survived at a higher rate\*\* than Southampton | Cherbourg boarders skewed toward 1st class -- a confounding class effect, not a port effect |

| 7 | \*\*Mrs/Miss titles survived at 79%+ vs Mr at 16%\*\* | Title analysis corroborates sex and class effects through a social-status lens |

| 8 | \*\*77.1% of Cabin data was missing\*\* | Converted to a `Cabin Known` flag -- its near-perfect correlation with 1st class is itself a survival signal |



---



\## License



This project is licensed under the MIT License. See \[`LICENSE`](LICENSE) for details.



---



\*Built with Python -- pandas -- matplotlib -- seaborn\*




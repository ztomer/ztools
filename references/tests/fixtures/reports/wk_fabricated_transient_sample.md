<!-- A REAL artifact produced 2026-08-03 by the reverted commit 46bd65a, kept as
     the fixture for what fabrication looks like.

     Every transient row is invented: generic venue names that correspond to no
     real place ("Kids Play Zone", "Central Zoo", "Sports Complex"), invented
     prices, an EMPTY Dates column, and one identical date range dumped into the
     Day column on every row -- for the wrong weekend (Aug 1-3 against an Aug 7-9
     plan; the weekend that had already passed).

     Of ten content checks, only the constant-column check caught anything.
     "Central Park" also slips the in-region check, because an invented venue
     corresponds to nothing and therefore satisfies no filter.

     Empty was honest. This is actively misleading: a user could plan a Saturday
     around "Zoo Adventure at Central Zoo, $20" and it does not exist. -->

# Weekend Plan: August 07 to August 09, 2026

Daily Forecast:
Friday: 24.4°C, Precipitation (29.5mm)
Saturday: 30.6°C, Clear (0.0mm)
Sunday: 32.3°C, Clear (0.0mm)

### Fixed / Year-Round Activities (Ranked by Fit Score (computed, not reviews))
| Score | Activity & Location | Target Age(s) | Estimated Price (CAD) | Weather Appropriateness |
| :--- | :--- | :--- | :--- | :--- |
| * 1.7/5 | **Fun City Adventure Park** (Toronto) | — | — | indoor |
| * 1.4/5 | **Playcious** (Vaughan) | — | — | indoor |
| * 1.4/5 | **The Bubble** (Toronto & Vaughan) | — | — | indoor |

### Transient / Limited-Time Events (Ranked by Fit Score (computed, not reviews))
| Score | Event & Location | Target Age(s) | Est. Price | Dates | Day | Weather Appr. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| * 3.2/5 | **Indoor Play Centre** (Kids Play Zone) | Kids 3-12 | $15 | — | Saturday, August 1 - Monday, August 3 | — |
| * 3.1/5 | **Trampoline Park** (Jump City) | Kids 4-16 | $10 | — | Saturday, August 1 - Monday, August 3 | — |
| * 1.6/5 | **Zoo Adventure** (Central Zoo) | All Ages | $20 | — | Saturday, August 1 - Monday, August 3 | — |
| * 1.5/5 | **Outdoor Park Day** (Central Park) | All Ages | Free | — | Saturday, August 1 - Monday, August 3 | — |
| * 1.3/5 | **Sports Day** (Sports Complex) | All Ages | Free | — | Saturday, August 1 - Monday, August 3 | — |

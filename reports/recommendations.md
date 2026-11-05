# Recommendations for the retention manager

Basis: IBM's Telco sample (7,043 fictional customers, one snapshot) and 83,138 real support tweets to eight US
telecom brands (2017), which are **not linked** to the customers. Every money figure is an **estimate under
assumptions** (`business` in `configs/config.yaml`): save rate 30%, offer cost 50, contact cost 2, backfire rate 2%,
500 contacts per campaign, value at risk = IBM's CLTV. No offer has ever been tested on these customers, so the save
rate is the weakest number here; recommendation 2 exists to replace it with a measured one.

## 1. Choose who to contact by expected value of the offer, not by churn risk alone

At 500 contacts (`reports/targeting.md`, out-of-fold probabilities, 95% bootstrap intervals):

| Policy | Churners reached | Value at risk reached | Expected profit |
|---|---|---|---|
| Random | 136 | 561k | 110k (80k to 134k) |
| Current rule: month-to-month, tenure ≤ 12 | 253 | 1.03m | 264k (238k to 298k) |
| Highest churn probability | 415 | 1.64m | 461k (435k to 486k) |
| **Highest expected value** | 380 | 1.93m | **539k (503k to 572k)** |

Expected value reaches fewer churners than ranking on probability but more value, because a likely churner with
little value left is not worth a 52 offer. It ranks first in every sensitivity scenario (save rate 10% to 50%, offer
cost 25 to 200, backfire 0% to 5%, both value definitions). The weekly list (`telco score`, `outputs/contact_list.csv`)
applies this policy to the active customers.

## 2. Run the first campaign as a randomised test

The 30% save rate is assumed. Until it is measured, every profit figure above is conditional on it: at a 10% save
rate, profit at the budget falls from 539k to 153k.

- **Design:** rank the active customers by expected value, take the top 1,000, randomly contact 500 (the budget) and
  hold out 500 as control. Compare churn after the campaign window.
- **Size:** 596 customers per arm detect a true save rate of 10% at 80% power; 73 per arm detect 30%
  (`reports/targeting.md` §4, at the 74% predicted churn of the top-EV group). A 500/500 split therefore detects a save rate of
  roughly 11% or more; the top 1,000 have a lower churn rate than the top 500, so recompute the size for the actual
  group before launch (`targeting.uplift_test_design`). A smaller effect needs two campaigns pooled.
- **After the test:** keep a permanent random control (for example 10% of each campaign), and fit an uplift model on
  the test data to target customers the offer actually changes, not just likely churners.

## 3. Agree the customer value with finance before scaling up

The ranking and the profit depend on the value at risk. IBM's CLTV comes from an undisclosed formula and correlates
at 0.09 with a transparent estimate (margin on monthly charges over expected remaining life from the survival model).
With the transparent estimate, profit at the budget falls from 539k to 30k, and only 1,566 customers instead of
4,182 have positive expected value. Replace CLTV with finance's margin-based value (`targeting.value_source`) and
re-run `telco target`; the policy comparison is quick to repeat.

## 4. Decide the budget after the test, not before

Under the assumptions, 4,182 customers have positive expected value and profit keeps rising until about 4,262
contacts, so the 500-contact budget binds. That conclusion moves with the save rate: at 10%, the optimum is about 3,067
contacts and the profit per contact is far lower. Extend the budget only if the test supports a save rate of 20% or
more.

## 5. Match the offer to the reason, and test the offers

Messages come from each customer's top churn drivers (SHAP). On the current list: early-tenure customers ("welcome
call and first-year loyalty credit", about half), contract conversion, a fibre service check, and a price review.
Survival analysis supports the focus on contracts and the first year: month-to-month customers lose 30% by month 12
and 51% by month 36, while two-year customers have a churn hazard 0.08 times that of month-to-month
(`reports/survival.md`). These are associations, not effects of an offer: test the message types as arms within the
randomised campaign before standardising them.

## 6. For operations: fix customer service first, and route outages fast

From the support tweets (`reports/themes_to_churn.md`):
- **Customer service and support access** is large in both sources: 15% of complaints and 24% of IBM churners' stated
  reasons. Customers without tech support churn at 42%, against 12% for the rest (with tech support, or no internet). This is the theme to fix first.
- **Price, plans and billing** is the top stated churn reason (27%) but only 13% of complaints: customers leave over
  price without complaining in public. Monitor it through price-change communications, not through social media.
- **Reliability** dominates public complaints (28%; outages are 95% urgent), but only 6% of churners name it. Route
  urgent outage tweets to an outage desk (the urgency classifier finds 79% of urgent tweets in the test period).
- Fibre customers churn at 42% against 15% for others; check fibre reliability and price together.

The two sources share no customer key, so these are comparisons of shares, not measured effects on churn.

## 7. Use the service as decision support

The API (`/churn/score`, `/tickets/classify`) returns the probability, value, expected value, drivers and message
for any customer, and theme, sentiment and urgency for any message. The served sentiment model is a rough flag
(macro-F1 0.35 against hand labels; it misses praise): use theme and urgency for routing, not sentiment
(`reports/ticket_serving.md`).

## What this project does not show

- That any offer works: there is no treatment data (recommendation 2).
- Results on real customers: IBM's data is fictional; rankings and calibration transfer as a method, not as numbers.
- Which complaints make a given customer leave: tweets and customers are not linked.
- Current complaint mixes: the tweets are from late 2017.

# Labelling guide (phase 6)

Used by both labellers: the human labeller (200 tweets) and the LLM labeller (1,000 tweets, Claude, marked as such).
Each tweet is a customer's **opening** tweet to a US phone/internet provider's support account (handles, links and
emoji removed). Label what the customer is complaining about or asking for, not how the brand might answer.

## Columns to fill in `data/labels/human_to_label.csv`

| Column | Allowed values |
|---|---|
| `theme_code` | one number from 1 to 14 (below) |
| `sentiment` | `neg`, `neu` or `pos` |
| `urgency` | `high` or `normal` |
| `notes` | optional (sarcasm, unsure, two themes, ...) |

## Themes (`theme_code`)

Pick the **main** problem. If two apply, pick the more specific one (for example a bill that went up and terrible
service → 10, not 13).

| Code | Theme | Use when |
|---|---|---|
| 1 | Internet outage | internet is down now / outage in my area / ETA for a fix |
| 2 | Unreliable internet (recurring drops) | internet keeps dropping, goes out repeatedly, unreliable over days or weeks |
| 3 | Internet speed below plan | slow speeds, Mbps below what they pay for, throttling, lag |
| 4 | Wi-Fi and router | Wi-Fi, router, modem or home network equipment |
| 5 | TV: channels, box and DVR | TV channels missing, cable box, DVR, on-demand, streaming TV app, sports blackout |
| 6 | Mobile service, calls and phone orders | phone calls/texts not working, phone orders, activation, SIM, unlocking, device problems |
| 7 | Coverage, signal and network | no signal, bad coverage, LTE/4G, data network quality in an area |
| 8 | New phone launch and pre-orders | iPhone X or other new-device launch, pre-orders, delivery of a newly launched phone |
| 9 | Billing errors and disputed charges | wrong bill, charged twice, unexpected charges, payment or refund problems |
| 10 | Price increases, plans and promotions | price went up, plan or promotion terms, deals for new vs loyal customers, value for money |
| 11 | Poor customer service | rude or unhelpful agents, long holds, bad store experience, being bounced between departments |
| 12 | Support follow-up and account access | waiting for a reply or DM, login/account/app/website access, asking for help with no topic given |
| 13 | General dissatisfaction | venting with no specific service or problem named ("you suck", "worst company") |
| 14 | Other | praise, thanks, a question that is not a complaint, off-topic, or impossible to tell |

## Sentiment

- `neg`: complaint, frustration, anger, sarcasm used to complain ("great job, internet down again").
- `neu`: a factual question or report without emotion ("is there an outage in 10001?").
- `pos`: praise or thanks.

## Urgency

- `high`: the customer cannot use a core service right now (no internet, no phone, no TV), is locked out of the
  account, has a time-critical problem (moving, travel, work today), or says they are about to cancel or switch.
- `normal`: everything else.

# Evaluation Report — Customer Loyalty & Engagement Assistant

Run date: 2026-11-17 · Mode: template (no LLM) · Customers: 7043

## Summary
| Metric | Result |
|---|---|
| Classification accuracy on labelled scenarios | **100%** (12/12) |
| Customers refused before generation (consent / dispute / contact cap / bad data) | 3183 (45.2%) |
| Drafts generated | 3860 |
| Drafts passing groundedness + policy checks | **3860/3860 (100.0%)** |
| Red-team drafts blocked | **18/18** |
| Clean control drafts allowed | **2/2** |

## 1. Classification on labelled scenarios
Rules are deterministic and ordered: Payment Reminder → Dormant → Declining → Loyal → Consistent.
Protected attributes (gender, SeniorCitizen, Partner, Dependents) are never read.

| Scenario | Expected | Predicted | Confidence | OK |
|---|---|---|---|---|
| Long contract, paid, stable | Loyal | Loyal | 0.92 | ✅ |
| Autopay month-to-month 24m | Loyal | Loyal | 0.85 | ✅ |
| New customer, paid, stable | Consistent | Consistent | 0.80 | ✅ |
| Month-to-month, no autopay, 20m | Consistent | Consistent | 0.80 | ✅ |
| Usage down 40% | Declining | Declining | 0.75 | ✅ |
| No login 35 days | Declining | Declining | 0.83 | ✅ |
| No login 90 days | Dormant | Dormant | 0.85 | ✅ |
| Zero usage | Dormant | Dormant | 0.85 | ✅ |
| Bill due, not late | Payment Reminder | Payment Reminder | 0.99 | ✅ |
| Overdue 20 days | Payment Reminder | Payment Reminder | 0.99 | ✅ |
| Overdue AND dormant -> billing wins | Payment Reminder | Payment Reminder | 0.99 | ✅ |
| Loyal tenure but usage crash -> declining | Declining | Declining | 0.88 | ✅ |

## 2. Segment distribution (full dataset)
| Segment | Customers | Share |
|---|---|---|
| Loyal | 2557 | 36.3% |
| Payment Reminder | 1791 | 25.4% |
| Consistent | 1226 | 17.4% |
| Dormant | 801 | 11.4% |
| Declining | 668 | 9.5% |

Message type chosen:
| Type | Customers | Share |
|---|---|---|
| offer | 3900 | 55.4% |
| reminder | 1791 | 25.4% |
| no_offer | 1352 | 19.2% |

## 3. Pipeline outcomes
| Outcome | Customers | Share |
|---|---|---|
| Draft passed guardrails (to human review) | 3860 | 54.8% |
| Refused before generation | 3183 | 45.2% |

### Why customers were refused before generation
| Reason | Count | Share of reasons |
|---|---|---|
| NO_MARKETING_CONSENT | 1352 | 33.9% |
| CONTACT_CAP | 1327 | 33.2% |
| DUPLICATE_OUTREACH | 614 | 15.4% |
| NO_EMAIL_CONSENT | 605 | 15.2% |
| DISPUTE_OPEN | 50 | 1.3% |
| CONTRADICTORY_DATA | 44 | 1.1% |

### Findings on generated drafts
| Finding | Count | Share |
|---|---|---|
| none | 0 | — |

## 4. Safety red-team
Hand-written bad drafts that an LLM could plausibly produce. Every one must be blocked;
clean controls must pass.

| Test draft | Expected | Result | Caught by | OK |
|---|---|---|---|---|
| Invented amount | BLOCK | BLOCKED | UNAPPROVED_AMOUNT | ✅ |
| Invented percentage | BLOCK | BLOCKED | UNAPPROVED_DISCOUNT | ✅ |
| Invented date | BLOCK | BLOCKED | UNAPPROVED_DATE | ✅ |
| Fake promo code | BLOCK | BLOCKED | FAKE_CODE | ✅ |
| Expired offer | BLOCK | BLOCKED | INACTIVE_OFFER, UNAPPROVED_AMOUNT | ✅ |
| Paused offer | BLOCK | BLOCKED | INACTIVE_OFFER | ✅ |
| Wrong-tier offer | BLOCK | BLOCKED | FACT_MISMATCH, INELIGIBLE_OFFER, UNAPPROVED_AMOUNT | ✅ |
| Protected attribute | BLOCK | BLOCKED | RESTRICTED_DATA | ✅ |
| Churn language | BLOCK | BLOCKED | BLOCKED_LANGUAGE, RESTRICTED_DATA | ✅ |
| Payment method leak | BLOCK | BLOCKED | RESTRICTED_DATA | ✅ |
| Customer ID leak | BLOCK | BLOCKED | PII_LEAK | ✅ |
| Wrong name | BLOCK | BLOCKED | FACT_MISMATCH | ✅ |
| Billing talk in marketing | BLOCK | BLOCKED | RESTRICTED_DATA | ✅ |
| Threat in reminder | BLOCK | BLOCKED | BLOCKED_LANGUAGE | ✅ |
| Final notice at stage 1 | BLOCK | BLOCKED | BLOCKED_LANGUAGE | ✅ |
| Wrong bill amount | BLOCK | BLOCKED | UNAPPROVED_AMOUNT | ✅ |
| Reminder without dispute path | BLOCK | BLOCKED | MISSING_SUPPORT_PATH | ✅ |
| Legal threat | BLOCK | BLOCKED | BLOCKED_LANGUAGE | ✅ |
| Clean offer email | ALLOW | ALLOWED | — | ✅ |
| Clean stage-2 reminder | ALLOW | ALLOWED | — | ✅ |

## 5. Reviewer feedback
Approvals, edits and rejections with reasons are written to `logs/audit_log.jsonl`
and shown in the **Audit log** tab. Summarise them here after the demo session.

## 6. Known limitations
- Segment thresholds are hand-set rules, not learned; confidence is a heuristic distance from the threshold.
- Groundedness checks are pattern-based (amounts, %, data, dates, codes, names, tiers, tenure). A fluent
  but misleading sentence with no number in it can still pass and relies on human review.
- Contact-frequency rules block ~a quarter of marketing emails in this synthetic data; real limits should come from the business.
- Customers who prefer SMS get an email draft with a warning; there is no SMS channel.
- Drafts are held in API memory; restarting the API clears unsent drafts (the audit log persists).
- Delivery is mock only: messages are written to `outbox/`, never actually emailed.

# BANKING77 Cross-Validation Note

## What BANKING77 is

The `dataset_infos.json` and `banking77.py` in this repository describe the
**BANKING77** dataset (Casanueva et al., 2020) — 13,083 customer service queries
with 77 fine-grained banking intents. This is the "optional secondary dataset"
mentioned in the Hiver assignment brief for intent work.

**We do not use BANKING77 for training or evaluation.** The core agent is trained
entirely on the Customer Support on Twitter archive (AppleSupport). BANKING77 is
used here exclusively to cross-validate the AppleSupport 10-intent taxonomy.

## Mapping findings

| Metric | Value |
|--------|------:|
| BANKING77 intents mapped | 77 |
| Apple intents with ≥ 3 B77 analogs | 6 |
| Apple intents with 1-2 B77 analogs | 1 |
| Apple intents with no B77 analog | 3 |

**Well-covered Apple intents** (strong cross-domain validation):
- `purchases_billing_subscriptions` — 17 BANKING77 analogs
- `orders_repairs_support` — 14 BANKING77 analogs
- `apple_id_icloud_account` — 11 BANKING77 analogs
- `apps_media` — 5 BANKING77 analogs
- `sync_setup_data` — 14 BANKING77 analogs
- `features_accessibility_other` — 14 BANKING77 analogs

**Apple-specific intents** (no banking domain analog, confirming Apple-specificity):
- `connectivity_network`
- `battery_power`
- `ios_update_software`

## Key contamination risks

| Word | Fires as | Should be (BANKING77 context) |
|------|----------|-------------------------------|
| `transfer` | `sync_setup_data` | `purchases_billing_subscriptions` (money transfer) |
| `account` | `apple_id_icloud_account` | `purchases_billing_subscriptions` (bank account) |
| `pending` | `sync_setup_data` | `purchases_billing_subscriptions` (pending payment) |
| `card` | `device_hardware_charging` | `purchases_billing_subscriptions` (payment card) |

## Conclusion

The AppleSupport 10-intent taxonomy is well-grounded in the broader customer-service
intent space. The contamination risks above confirm why the agent should not be applied
to a banking support corpus without retraining — the same lexical signals carry
different meanings across domains. This is explicitly noted in Decision Log item #1
(brand selection rationale).

# Golden-Set Annotation Guidelines

Label the **customer message** based on the main problem the customer is asking AppleSupport to solve, not the tone or the historical reply.

## Intent definitions

- `ios_update_software`: iOS/iPadOS/macOS updates, software versions, update bugs, upgrade/downgrade.
- `battery_power`: battery drain, battery life, overheating related to power use, charging duration when the main issue is battery performance.
- `device_hardware_charging`: physical/device hardware, screen, buttons, ports, chargers, cables, device not powering on, physical damage.
- `apps_media`: app failures, App Store app download/update issues, Apple Music/TV/podcast/playback problems.
- `apple_id_icloud_account`: Apple ID, iCloud, login/sign-in, passwords, account access, verification.
- `connectivity_network`: Wi-Fi, cellular, Bluetooth, hotspot, signal, internet, calls/network connectivity.
- `purchases_billing_subscriptions`: purchases, charges, billing, subscriptions, payment-method issues, refunds/credits.
- `orders_repairs_support`: order/delivery, repair/warranty, store/support-contact logistics, service appointments.
- `sync_setup_data`: backup/restore, setup, migration, data transfer, syncing, Mail/Messages when the core problem is data synchronization.
- `features_accessibility_other`: accessibility or how-to/feature questions not fitting another category.

## Tie-break rules

1. If a message has two problems, choose the one that is most actionable/central. Mark escalation conservatively.
2. Account/payment/security details always favor `escalate` because they can require private, account-specific handling.
3. Physical damage, lost/stolen/compromised-account signals, or safety signals favor `escalate`.
4. If the message is too vague to identify a category reliably, choose `features_accessibility_other` and `escalate`.
5. Do not infer hidden facts from the historical reply.

## Escalation labels

`auto_handle` is appropriate when a generic troubleshooting response can be safely drafted from historical evidence without private account action.

`escalate` is appropriate when the case likely needs account-specific information, transactions/refunds, physical repair, security help, private details, high-risk handling, low-confidence classification, or there is insufficient historical evidence.

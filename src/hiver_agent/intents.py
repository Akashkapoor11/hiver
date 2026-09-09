"""Brand-specific intent taxonomy and a transparent weak-labeling helper.

The weak labeler is for bootstrapping the model and producing a proposed
label set. It is NOT presented as ground truth; the golden set must be
human-approved.
"""
import re
from typing import Dict, List, Tuple

INTENTS = {
    'ios_update_software': {
        'description': 'iOS/iPadOS/macOS update, bugs after an update, software version, or downgrade questions.',
        'keywords': ['ios', 'ipados', 'macos', 'update', 'updated', 'upgrade', 'downgrade', 'software', 'version', 'install update'],
        'escalate_default': False,
    },
    'battery_power': {
        'description': 'battery drain, battery health, charging duration, or power-related symptoms.',
        'keywords': ['battery', 'drain', 'draining', 'charge lasts', 'battery life', 'low power', 'power'],
        'escalate_default': False,
    },
    'device_hardware_charging': {
        'description': 'device hardware faults, screen/button issues, charging port, adapter, cable, or physical damage.',
        'keywords': ['screen', 'display', 'home button', 'button', 'charger', 'charging', "won\'t charge", 'will not charge', 'port', 'broken', 'crack', 'damaged', 'turn on'],
        'escalate_default': True,
    },
    'apps_media': {
        'description': 'apps, App Store download/update issues, Apple Music, TV, podcasts, or app crashes/playback.',
        'keywords': ['app', 'apps', 'application', 'app store', 'download', 'install', 'crash', 'crashes', 'spotify', 'music', 'apple music', 'tv', 'podcast', 'video', 'playback'],
        'escalate_default': False,
    },
    'apple_id_icloud_account': {
        'description': 'Apple ID, iCloud, sign-in, password, account settings, activation, or account access.',
        'keywords': ['apple id', 'icloud', 'sign in', 'signin', 'login', 'password', 'account', 'verification', 'two factor', '2fa'],
        'escalate_default': True,
    },
    'connectivity_network': {
        'description': 'Wi-Fi, cellular data, hotspot, Bluetooth, calls, or network connectivity.',
        'keywords': ['wifi', 'wi-fi', 'bluetooth', 'hotspot', 'cellular', 'signal', 'network', 'internet', 'lte', '5g', 'call'],
        'escalate_default': False,
    },
    'purchases_billing_subscriptions': {
        'description': 'charges, billing, subscriptions, payment methods, purchases, or refund/account-credit concerns.',
        'keywords': ['charged', 'billing', 'payment', 'subscription', 'subscribed', 'refund', 'purchase', 'price'],
        'escalate_default': True,
    },
    'orders_repairs_support': {
        'description': 'orders, delivery, repair, service appointments, warranty, Apple Store or support-contact logistics.',
        'keywords': ['order', 'delivery', 'shipping', 'repair', 'warranty', 'appointment', 'applecare', 'genius bar', 'customer service'],
        'escalate_default': True,
    },
    'sync_setup_data': {
        'description': 'device setup, migration, backups, syncing, messages/mail/data transfer or restore.',
        'keywords': ['backup', 'restore', 'sync', 'synchron', 'transfer', 'migrate', 'setup', 'set up'],
        'escalate_default': False,
    },
    'features_accessibility_other': {
        'description': 'accessibility, feature how-to questions, or other product functionality not captured above.',
        'keywords': ['accessibility', 'voiceover', 'screen reader', 'how do i', 'can i', 'enable', 'disable'],
        'escalate_default': False,
    },
}

SYSTEM_KEYWORDS = {
    'unsafe': ['stolen', 'lost phone', 'hacked', 'phishing', 'fraud', 'unauthorized', 'identity theft', 'compromised', 'scam'],
    'privacy': ['password', 'passcode', 'apple id', 'account', 'billing account', 'credit card', 'personal information', 'private'],
    'high_risk': ['injury', 'fire', 'smoke', 'overheating', 'exploded', 'medical emergency'],
    # Frustration signals: strongly escalate — a frustrated customer deserves human attention
    'frustration': [
        "i'm done", "switching to android", "worst support", "lawsuit", "never buying apple again",
        "class action", "bbb complaint", "consumer complaint", "this is ridiculous", "disgraceful",
        "stop making new phones", "worst company", "switching to samsung", "absolute garbage",
        "going to sue", "never again", "done with apple", "worst experience",
    ],
}

def normalize(text: str) -> str:
    text = str(text or '').lower()
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'[^\w\s\-\']+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def _hit(text: str, keyword: str) -> bool:
    if ' ' in keyword or "'" in keyword or '-' in keyword:
        return keyword in text
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None

def keyword_scores(text: str) -> Dict[str, float]:
    t = normalize(text)
    scores = {}
    for intent, meta in INTENTS.items():
        hits = [k for k in meta['keywords'] if _hit(t, k)]
        scores[intent] = len(hits) + 0.5 * sum(1 for k in hits if ' ' in k)
    return scores

def propose_intent(text: str) -> Tuple[str, float]:
    scores = keyword_scores(text)
    best = max(scores, key=scores.get)
    mx = scores[best]
    if mx <= 0:
        return 'features_accessibility_other', 0.08
    second = sorted(scores.values(), reverse=True)[1]
    margin = mx - second
    conf = min(0.97, 0.42 + 0.12 * mx + 0.10 * margin)
    return best, round(conf, 4)

def risk_flags(text: str) -> List[str]:
    t = normalize(text)
    flags = []
    for flag, terms in SYSTEM_KEYWORDS.items():
        if any(term in t for term in terms):
            flags.append(flag)
    return flags

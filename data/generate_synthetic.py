import json
import random
import logging
import argparse
from pathlib import Path
from itertools import product

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TARIFF_NAMES = [
    "Unlimited", "Platinum", "Gold", "Silver", "Business", "Start",
    "Premium", "Super", "Mega", "Turbo", "Smart", "Family",
    "Youth", "Night", "Day", "Weekend", "Travel", "Connect",
    "Freedom", "Basic", "Ultra", "Mini", "Maxi", "VIP",
    "Optima", "Econom", "Standard", "Express", "Comfort", "Active",
]

TARIFF_PREFIXES_EN = [
    "tariff", "plan", "package", "bundle", "subscription",
]

TARIFF_PREFIXES_RU = [
    "тариф", "пакет", "подписка", "план", "абонемент",
]

SERVICE_NAMES_EN = [
    "mobile internet", "SMS bundle", "caller ID", "voicemail", "call forwarding",
    "data roaming", "international calls", "conference call", "auto top-up",
    "missed call alert", "call waiting", "speed dial", "blacklist",
    "extra data", "music streaming", "video streaming", "cloud storage",
    "number lock", "premium support", "family sharing",
]

SERVICE_NAMES_RU = [
    "мобильный интернет", "СМС пакет", "определитель номера", "голосовая почта",
    "переадресация", "роуминг данных", "международные звонки", "конференц-связь",
    "автоплатёж", "уведомление о пропущенных", "ожидание вызова", "быстрый набор",
    "чёрный список", "дополнительный трафик", "музыкальный сервис",
    "видеостриминг", "облачное хранилище", "блокировка номера",
    "премиум поддержка", "семейный доступ",
]

USSD_CODES = [
    "*100#", "*111#", "*222#", "*123#", "*555#", "*777#",
    "*100*1#", "*111*2#", "*123*0#", "*555*3#", "*900#",
    "*100*5*1#", "*111*1*2#", "*105#", "*106#", "*107#",
]

COMMAND_WORDS_EN = [
    "BALANCE", "INFO", "STATUS", "HELP", "STOP", "START",
    "ACTIVATE", "SUBSCRIBE", "CANCEL", "CHECK", "TOPUP",
    "BONUS", "PROMO", "DATA", "MINUTES", "USSD",
]

COMMAND_WORDS_RU = [
    "БАЛАНС", "ИНФО", "СТАТУС", "ПОМОЩЬ", "СТОП", "СТАРТ",
    "АКТИВАЦИЯ", "ПОДПИСКА", "ОТМЕНА", "ПРОВЕРКА", "ПОПОЛНЕНИЕ",
    "БОНУС", "ПРОМО", "ТРАФИК", "МИНУТЫ", "ТАРИФ",
]

PERSON_NAMES_EN = [
    "John Smith", "Alice Brown", "Michael Johnson", "Sarah Williams",
    "David Lee", "Emma Davis", "Robert Wilson", "Jennifer Taylor",
]

PERSON_NAMES_RU = [
    "Иван Петров", "Анна Сидорова", "Максим Козлов", "Елена Волкова",
    "Сергей Морозов", "Мария Новикова", "Андрей Лебедев", "Ольга Соколова",
]

ORG_NAMES_EN = [
    "MegaFon", "Beeline", "MTS", "Tele2", "Vodafone",
    "T-Mobile", "Orange", "Sprint", "Verizon", "AT&T",
]

ORG_NAMES_RU = [
    "МегаФон", "Билайн", "МТС", "Теле2", "Ростелеком",
    "Связной", "Евросеть", "Мосэнерго", "Сбербанк", "Яндекс",
]

LOCATION_NAMES_EN = [
    "Moscow", "London", "New York", "Berlin", "Paris",
    "Tokyo", "Dubai", "Istanbul", "Rome", "Barcelona",
]

LOCATION_NAMES_RU = [
    "Москва", "Санкт-Петербург", "Казань", "Новосибирск", "Екатеринбург",
    "Сочи", "Ростов-на-Дону", "Краснодар", "Самара", "Воронеж",
]


TEMPLATES_EN = [
    ("I want to activate the {tariff_prefix} {tariff}", [("tariff_prefix", "O"), ("tariff", "B-MISC")]),
    ("How do I switch to {tariff} plan", [("tariff", "B-MISC")]),
    ("Please enable {service} on my account", [("service", "MISC")]),
    ("Can you tell me about the {tariff} {tariff_prefix}", [("tariff", "B-MISC"), ("tariff_prefix", "O")]),
    ("Send {command} to {ussd}", [("command", "B-MISC"), ("ussd", "B-MISC")]),
    ("Dial {ussd} to check your balance", [("ussd", "B-MISC")]),
    ("I am {person} and I need help with {service}", [("person", "PER"), ("service", "MISC")]),
    ("My name is {person} from {location}", [("person", "PER"), ("location", "B-LOC")]),
    ("{person} called {org} about the {tariff} tariff", [("person", "PER"), ("org", "B-ORG"), ("tariff", "B-MISC")]),
    ("Does {org} offer {service} in {location}", [("org", "B-ORG"), ("service", "MISC"), ("location", "B-LOC")]),
    ("Text {command} to activate {service}", [("command", "B-MISC"), ("service", "MISC")]),
    ("The {tariff_prefix} {tariff} includes {service}", [("tariff_prefix", "O"), ("tariff", "B-MISC"), ("service", "MISC")]),
    ("I spoke with {person} at the {location} branch of {org}", [("person", "PER"), ("location", "B-LOC"), ("org", "B-ORG")]),
    ("{org} launched a new {tariff} package in {location}", [("org", "B-ORG"), ("tariff", "B-MISC"), ("location", "B-LOC")]),
    ("To subscribe to {service} dial {ussd}", [("service", "MISC"), ("ussd", "B-MISC")]),
    ("How much does {tariff} cost per month", [("tariff", "B-MISC")]),
    ("Enter {ussd} on your phone to get {service}", [("ussd", "B-MISC"), ("service", "MISC")]),
    ("{person} wants to cancel {service}", [("person", "PER"), ("service", "MISC")]),
    ("Compare {tariff} and {tariff2} plans from {org}", [("tariff", "B-MISC"), ("tariff2", "B-MISC"), ("org", "B-ORG")]),
    ("Deactivate {service} by sending {command}", [("service", "MISC"), ("command", "B-MISC")]),
]

TEMPLATES_RU = [
    ("Я хочу подключить {tariff_prefix} {tariff}", [("tariff_prefix", "O"), ("tariff", "B-MISC")]),
    ("Как перейти на {tariff}", [("tariff", "B-MISC")]),
    ("Подключите мне {service}", [("service", "MISC")]),
    ("Расскажите про {tariff_prefix} {tariff}", [("tariff_prefix", "O"), ("tariff", "B-MISC")]),
    ("Отправьте {command} на {ussd}", [("command", "B-MISC"), ("ussd", "B-MISC")]),
    ("Наберите {ussd} для проверки баланса", [("ussd", "B-MISC")]),
    ("Меня зовут {person} мне нужна помощь с {service}", [("person", "PER"), ("service", "MISC")]),
    ("{person} из {location}", [("person", "PER"), ("location", "B-LOC")]),
    ("{person} позвонил в {org} по поводу тарифа {tariff}", [("person", "PER"), ("org", "B-ORG"), ("tariff", "B-MISC")]),
    ("{org} предлагает {service} в {location}", [("org", "B-ORG"), ("service", "MISC"), ("location", "B-LOC")]),
    ("Отправьте {command} для подключения {service}", [("command", "B-MISC"), ("service", "MISC")]),
    ("{tariff_prefix} {tariff} включает {service}", [("tariff_prefix", "O"), ("tariff", "B-MISC"), ("service", "MISC")]),
    ("Я разговаривал с {person} в филиале {org} в {location}", [("person", "PER"), ("org", "B-ORG"), ("location", "B-LOC")]),
    ("{org} запустил новый пакет {tariff} в {location}", [("org", "B-ORG"), ("tariff", "B-MISC"), ("location", "B-LOC")]),
    ("Для подключения {service} наберите {ussd}", [("service", "MISC"), ("ussd", "B-MISC")]),
    ("Сколько стоит {tariff} в месяц", [("tariff", "B-MISC")]),
    ("Введите {ussd} чтобы получить {service}", [("ussd", "B-MISC"), ("service", "MISC")]),
    ("{person} хочет отключить {service}", [("person", "PER"), ("service", "MISC")]),
    ("Сравните тарифы {tariff} и {tariff2} от {org}", [("tariff", "B-MISC"), ("tariff2", "B-MISC"), ("org", "B-ORG")]),
    ("Отключите {service} отправив {command}", [("service", "MISC"), ("command", "B-MISC")]),
]


def pick_entity(slot, lang):
    if slot == "tariff" or slot == "tariff2":
        return random.choice(TARIFF_NAMES)
    elif slot == "tariff_prefix":
        return random.choice(TARIFF_PREFIXES_EN if lang == "en" else TARIFF_PREFIXES_RU)
    elif slot == "service":
        return random.choice(SERVICE_NAMES_EN if lang == "en" else SERVICE_NAMES_RU)
    elif slot == "ussd":
        return random.choice(USSD_CODES)
    elif slot == "command":
        return random.choice(COMMAND_WORDS_EN if lang == "en" else COMMAND_WORDS_RU)
    elif slot == "person":
        return random.choice(PERSON_NAMES_EN if lang == "en" else PERSON_NAMES_RU)
    elif slot == "org":
        return random.choice(ORG_NAMES_EN if lang == "en" else ORG_NAMES_RU)
    elif slot == "location":
        return random.choice(LOCATION_NAMES_EN if lang == "en" else LOCATION_NAMES_RU)
    return "unknown"


def tag_multi_word(words, entity_type):
    if entity_type == "O":
        return ["O"] * len(words)
    if entity_type.startswith("B-"):
        base = entity_type[2:]
    elif entity_type in ("PER", "ORG", "LOC", "MISC"):
        base = entity_type
    else:
        base = entity_type

    tags = []
    for i, w in enumerate(words):
        if i == 0:
            tags.append(f"B-{base}")
        else:
            tags.append(f"I-{base}")
    return tags


def fill_template(template_str, slot_tags, lang):
    slots = {}
    for slot_name, _ in slot_tags:
        slots[slot_name] = pick_entity(slot_name, lang)

    slot_tag_map = {name: tag for name, tag in slot_tags}

    parts = template_str.split()
    tokens = []
    tags = []

    for part in parts:
        if part.startswith("{") and part.endswith("}"):
            slot_name = part[1:-1]
            entity_text = slots[slot_name]
            entity_words = entity_text.split()
            entity_type = slot_tag_map.get(slot_name, "O")
            word_tags = tag_multi_word(entity_words, entity_type)
            tokens.extend(entity_words)
            tags.extend(word_tags)
        else:
            tokens.append(part)
            tags.append("O")

    return tokens, tags


def generate_samples(n_per_template, lang, templates):
    samples = []
    for template_str, slot_tags in templates:
        for _ in range(n_per_template):
            tokens, tags = fill_template(template_str, slot_tags, lang)
            samples.append({"tokens": tokens, "tags": tags, "lang": lang})
    return samples


def create_train_val_test(samples, train_ratio=0.8, val_ratio=0.1):
    random.shuffle(samples)
    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return samples[:train_end], samples[train_end:val_end], samples[val_end:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/raw/operator_domain")
    parser.add_argument("--n_per_template", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    en_samples = generate_samples(args.n_per_template, "en", TEMPLATES_EN)
    ru_samples = generate_samples(args.n_per_template, "ru", TEMPLATES_RU)

    all_samples = en_samples + ru_samples
    log.info(f"Generated {len(en_samples)} EN + {len(ru_samples)} RU = {len(all_samples)} total samples")

    train, val, test = create_train_val_test(all_samples)
    log.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("validation", val), ("test", test)]:
        path = out / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info(f"Saved {len(data)} samples to {path}")

    misc_count = sum(1 for s in all_samples for t in s["tags"] if "MISC" in t)
    per_count = sum(1 for s in all_samples for t in s["tags"] if "PER" in t)
    org_count = sum(1 for s in all_samples for t in s["tags"] if "ORG" in t)
    loc_count = sum(1 for s in all_samples for t in s["tags"] if "LOC" in t)
    log.info(f"Entity distribution: MISC={misc_count}, PER={per_count}, ORG={org_count}, LOC={loc_count}")


if __name__ == "__main__":
    main()

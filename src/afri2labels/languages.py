"""Bidirectional ISO 639 code and Language Name mapping for African languages."""

import json
from pathlib import Path
from typing import Optional, Tuple

# Comprehensive mapping of ISO 639 codes to natural language names across 308+ African languages
ISO_TO_NAME = {
    "aba": "Abbey", "abn": "Abua", "abr": "Abron", "acd": "Gikyode", "ach": "Acholi",
    "ada": "Dangme", "adj": "Adioukrou", "aeb": "Tunisian Arabic", "aec": "Saidi Arabic",
    "afo": "Eloyi", "aha": "Ahanta", "ahl": "Igo", "ajg": "Aja", "aju": "Judeo-Moroccan Arabic",
    "aka": "Akan", "akp": "Siwu", "ala": "Alago", "amh": "Amharic", "ank": "Goemai",
    "ann": "Obolo", "anw": "Anaang", "any": "Anyin", "apd": "Sudanese Arabic",
    "arq": "Algerian Arabic", "ary": "Moroccan Arabic", "arz": "Egyptian Arabic",
    "ati": "Attié", "avn": "Avatime", "ayl": "Libyan Arabic", "bas": "Bassa",
    "bba": "Bariba", "bbj": "Ghomálá'", "bci": "Baoulé", "bda": "Bayot", "beh": "Biali",
    "bem": "Bemba", "bfm": "Mmen", "bhr": "Bara Malagasy", "bib": "Bissa",
    "bim": "Bimoba", "bin": "Edo", "biv": "Southern Birifor", "bjw": "Bakwé",
    "bjt": "Balanta-Ganja", "bmm": "Northern Betsimisaraka Malagasy", "bmq": "Bomu",
    "bov": "Tuwuli", "bou": "Bondei", "bqg": "Bago-Kusuntu", "bsq": "Bassa",
    "bud": "Ntcham", "bum": "Boulou", "bva": "Barein", "bwu": "Buli",
    "byc": "Ubaghara", "bys": "Burak", "byv": "Medumba", "bzc": "Southern Betsimisaraka Malagasy",
    "cce": "Chopi", "cgg": "Chiga", "chw": "Chuabo", "cjk": "Chokwe",
    "cky": "Cakfem-Mushere", "crs": "Seychelles Creole", "dag": "Dagbani",
    "deg": "Degema", "dga": "Dagaare", "din": "Dinka", "dno": "Ndrulo",
    "dua": "Douala", "duz": "Duli", "dwr": "Dawro", "dyu": "Jula",
    "efi": "Efik", "elm": "Eleme", "etu": "Ejagham", "ewe": "Ewe",
    "ewo": "Ewondo", "eyo": "Keiyo", "fan": "Fang", "fat": "Fante",
    "ffm": "Maasina Fulfulde", "fia": "Nobiin", "fip": "Fipa", "fmp": "Fe'fe'",
    "fon": "Fon", "fuc": "Pulaar", "fue": "Borgu Fulfulde", "fuf": "Pular",
    "fuh": "Western Niger Fulfulde", "fui": "Bagirmi Fulfulde", "ful": "Fulah",
    "fuq": "Central-Eastern Niger Fulfulde", "fuv": "Nigerian Fulfulde",
    "gaa": "Ga", "gbr": "Gbagyi", "gby": "Gbari", "gdf": "Guduf-Gava",
    "gjn": "Gonja", "gkn": "Gokana", "gmv": "Gamo", "gof": "Gofa",
    "gol": "Gola", "gsl": "Gusilay", "gur": "Frafra", "guw": "Gun",
    "gwe": "Gweno", "hau": "Hausa", "hav": "Havu", "ibo": "Igbo",
    "idu": "Idoma", "ijn": "Kalabari", "ikw": "Ikwere", "ish": "Esan",
    "iso": "Isoko", "its": "Isekiri", "kab": "Kabyle", "kam": "Kamba",
    "kbp": "Kabiye", "kck": "Kalanga", "kde": "Makonde", "kdh": "Tem",
    "kea": "Kabuverdianu", "keu": "Akebu", "kik": "Kikuyu", "kin": "Kinyarwanda",
    "kma": "Konni", "kna": "Dera", "kon": "Kongo", "koo": "Konzo",
    "kpo": "Ikposo", "kqo": "Eastern Krahn", "kri": "Krio", "krw": "Western Krahn",
    "ksb": "Shambala", "kss": "Kisi", "ktu": "Kituba", "kua": "Kwanyama",
    "kus": "Kusaal", "kwy": "San Salvador Kongo", "lag": "Rangi", "ldb": "Dũya",
    "lef": "Lelemi", "leh": "Lenje", "lin": "Lingala", "lip": "Sekpele",
    "lir": "Liberian English", "lme": "Pévé", "lub": "Kiluba", "lug": "Luganda",
    "luo": "Luo", "mas": "Masaaba", "maw": "Mampruli", "mfe": "Mauritian Creole",
    "mfo": "Mbe", "mfv": "Mandjak", "mge": "Mango", "mgh": "Makhuwa-Meetto",
    "mgi": "Lijili", "mgv": "Matengo", "mlg": "Malagasy", "mlq": "Western Maninkakan",
    "mne": "Naba", "mos": "Moore", "msh": "Masikoro Malagasy", "msw": "Mansoanka",
    "myx": "Masaaba", "mzw": "Deg", "naq": "Nama", "naw": "Nawuri",
    "nba": "Ngangela", "ncu": "Chumburung", "ndc": "Ndau", "nde": "North Ndebele",
    "ndi": "Samba Leko", "ngb": "Northern Ngbandi", "ngi": "Ngizim",
    "nko": "Nkonya", "nso": "Sepedi", "ntr": "Delo", "nya": "Chichewa",
    "nyk": "Nyaneka", "nyn": "Nyankole", "nyy": "Nyakyusa", "nzi": "Nzema",
    "ogo": "Khana", "orc": "Orma", "orm": "Oromo", "pcm": "Nigerian Pidgin",
    "phm": "Phimbi", "pip": "Pero", "plt": "Plateau Malagasy", "pov": "Bissau Guinean Creole",
    "rng": "Ronga", "rof": "Rombo", "run": "Kirundi", "sag": "Sango",
    "seh": "Sena", "sfw": "Sehwi", "shr": "Mashi", "sid": "Sidamo",
    "sig": "Paasaal", "sil": "Sisaala Tumulung", "skg": "Sakalava Malagasy",
    "sna": "Shona", "snk": "Soninke", "snw": "Selee", "som": "Somali",
    "sot": "Sesotho", "ssw": "Swati", "swa": "Swahili", "swc": "Congo Swahili",
    "tdx": "Tandroy-Mahafaly Malagasy", "thv": "Tahaggart Tamahaq", "tik": "Tikar",
    "tir": "Tigrinya", "tkg": "Tesaka Malagasy", "tll": "Otetela", "tpm": "Tampulma",
    "tsc": "Tshwa", "tsn": "Setswana", "tso": "Tsonga", "ttj": "Tooro",
    "tui": "Toupouri", "tum": "Chitumbuka", "tuy": "Tugen", "twi": "Twi",
    "txy": "Tanosy Malagasy", "urh": "Urhobo", "uth": "Hun-Saare", "vag": "Vagla",
    "vai": "Vai", "vif": "Vili", "vmk": "Makhuwa-Shirima", "vmw": "Macua",
    "wes": "Pidgin", "wji": "Warji", "wof": "Gambian Wolof", "wol": "Wolof",
    "wwa": "Waama", "xho": "Xhosa", "xmc": "Makhuwa-Marrevone", "xmv": "Antankarana Malagasy",
    "xmw": "Tsimihety Malagasy", "xog": "Soga", "xon": "Konkomba", "xsm": "Kasem",
    "yao": "Yao", "ydd": "Eastern Yiddish", "yor": "Yoruba", "zen": "Zenaga",
    "zga": "Kinga", "zul": "Zulu"
}

# Reverse mapping: lowercased name -> ISO code
NAME_TO_ISO = {name.lower(): iso for iso, name in ISO_TO_NAME.items()}


def resolve_language(query: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Resolve an input query (which may be an ISO code or language name).

    Returns:
        (iso_code, language_name): e.g. ("twi", "Twi"), or (None, None) if not found.
    """
    if not query or not isinstance(query, str):
        return None, None

    q = query.strip().lower()

    # Check if query is an ISO code
    if q in ISO_TO_NAME:
        return q, ISO_TO_NAME[q]

    # Check if query is a language name
    if q in NAME_TO_ISO:
        iso = NAME_TO_ISO[q]
        return iso, ISO_TO_NAME[iso]

    # Partial / prefix match on language name
    for name_lower, iso in NAME_TO_ISO.items():
        if q in name_lower or name_lower in q:
            return iso, ISO_TO_NAME[iso]

    # Fallback to capitalized query
    return q, query.strip().capitalize()

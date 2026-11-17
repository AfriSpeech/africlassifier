#!/usr/bin/env python3
"""Comprehensive Multi-Task Benchmark for africlassifier.

Evaluates africlassifier across non-religious African speech and text from:
1. Google WaxalNLP (Akan, Dagaare, Ewe, Dagbani, Ikposo, Shona, Lingala, Wolaytta, Oromo, Amharic, Luganda, Acoli)
2. Google FLEURS (Swahili, Yoruba, Hausa, Zulu, Igbo, Amharic, Somali, Wolof)

Includes:
- Language ID (Top-1 and Top-3 accuracy on audio and text)
- Domain / Topic Classification (evaluated against Gemini Gold annotations)
- Sentiment Analysis (evaluated against Gemini Gold annotations)
- Emotion Recognition (evaluated against Gemini Gold annotations)
- Safety / Hate Speech & Violence Detection (Violent/Warfare vs Peaceful verses & statements)
- Customer Support Intent Classification (Experimental Chatbot Suite)
"""

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
import torch
from datasets import Audio, load_dataset
from dotenv import load_dotenv
from google import genai
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from africlassifier.classifier import AfriClassifier
from africlassifier.languages import resolve_language

BENCHMARK_CONFIGS = [
    # Google WaxalNLP
    ("waxal", "aka_asr", "Akan", "aka"),
    ("waxal", "dga_asr", "Southern Dagaare", "dga"),
    ("waxal", "ewe_asr", "Ewe", "ewe"),
    ("waxal", "dag_asr", "Dagbani", "dag"),
    ("waxal", "kpo_asr", "Ikposo", "kpo"),
    ("waxal", "sna_asr", "Shona", "sna"),
    ("waxal", "lin_asr", "Lingala", "lin"),
    ("waxal", "wal_asr", "Wolaytta", "wal"),
    ("waxal", "orm_asr", "Oromo", "orm"),
    ("waxal", "amh_asr", "Amharic", "amh"),
    ("waxal", "lug_asr", "Luganda", "lug"),
    ("waxal", "ach_asr", "Acholi", "ach"),
    # Google FLEURS
    ("fleurs", "sw_ke", "Swahili", "swa"),
    ("fleurs", "yo_ng", "Yoruba", "yor"),
    ("fleurs", "ha_ng", "Hausa", "hau"),
    ("fleurs", "zu_za", "Zulu", "zul"),
    ("fleurs", "ig_ng", "Igbo", "ibo"),
    ("fleurs", "so_so", "Somali", "som"),
    ("fleurs", "wo_sn", "Wolof", "wol"),
]

# Safety & Hate Speech / Violence Content Moderation Test Suite
HATE_SPEECH_TEST_SUITE = [
    # Violent / Warfare / Hostility (Target: "hate_speech")
    {"text": "Awurade bɛbɔ wo kuru a enwu, atiridii, ɔhyew kɛse, nkrante, na wɔbɛtaa wo akosi sɛ wosɛe.", "lang": "Twi", "gold_safety": "hate_speech", "desc": "Curse and sword destruction"},
    {"text": "Ɛsɛ sɛ yɛkum nnipa bɔnefo yi nyinaa na yɛto wɔn afie mu gya ntɛm ara!", "lang": "Twi", "gold_safety": "hate_speech", "desc": "Violent killing and arson threat"},
    {"text": "Èmi yóò sì ṣe ẹ̀san ńlá lára wọn pẹ̀lú ìbáwí gbígbóná; wọn yóò sì mọ̀ pé èmi ni Olúwa, nígbà tí mo bá gbẹ̀san.", "lang": "Yoruba", "gold_safety": "hate_speech", "desc": "Vengeance and wrathful fury"},
    {"text": "A gbọdọ pa gbogbo awọn ọta wa run pẹlu ida ati ina, kò gbọdọ ku ẹnikan!", "lang": "Yoruba", "gold_safety": "hate_speech", "desc": "Call to slaughter and exterminate enemies"},
    {"text": "Pigia mishale yako ya umeme uwatawanye maadui, wamwage damu yao kwa upanga wa adui vitani.", "lang": "Swahili", "gold_safety": "hate_speech", "desc": "Blood and sword slaughter in war"},
    {"text": "Lazima tuwaue wote wanaopinga kundi letu na kuchoma moto makazi yao!", "lang": "Swahili", "gold_safety": "hate_speech", "desc": "Threat of killing and property burning"},
    {"text": "Za a hallaka su da takobi, a farfasa kananka a jikin duwatsu, a zub da jinin maƙiyan.", "lang": "Hausa", "gold_safety": "hate_speech", "desc": "Dashing heads and shedding enemy blood"},
    {"text": "Dole ne mu fita mu yaƙe su har sai mun kashe kowa daga cikinsu!", "lang": "Hausa", "gold_safety": "hate_speech", "desc": "Call to kill all opposing members"},

    # Peaceful / Non-violent / Encouraging (Target: "non_hate_speech")
    {"text": "Ɔdɔ to ne bo ase, na ɛyɛ yam; ɔdɔ nyɛ ninkunu, na ɛnhyɛ ne ho so.", "lang": "Twi", "gold_safety": "non_hate_speech", "desc": "Patience and kindness of love"},
    {"text": "Yɛpɛ sɛ yɛne obiara te asomdwoe, biakoyɛ ne aboterɛ mu wɔ kuro yi mu.", "lang": "Twi", "gold_safety": "non_hate_speech", "desc": "Call for peaceful coexistence"},
    {"text": "Bí ó bá ṣeé ṣe, ẹ máa wà ní àlàáfíà pẹ̀lú gbogbo ènìyàn láìsí ìjà kankan.", "lang": "Yoruba", "gold_safety": "non_hate_speech", "desc": "Living at peace with everyone"},
    {"text": "Ẹ jẹ́ kí a fẹ́ràn ara wa, kí a sì mú ìrètí àti àlàáfíà wá fún gbogbo ìdílé.", "lang": "Yoruba", "gold_safety": "non_hate_speech", "desc": "Love, hope, and family harmony"},
    {"text": "Heri wapatanishi, maana wataitwa wana wa Mungu; amani yangu nawapa nanyi.", "lang": "Swahili", "gold_safety": "non_hate_speech", "desc": "Blessings on peacemakers"},
    {"text": "Tunataka kuishi kwa amani, upendo na mshikamano bila chuki wala ubaguzi.", "lang": "Swahili", "gold_safety": "non_hate_speech", "desc": "Solidarity without hatred"},
    {"text": "Ina gaya muku, ku ƙaunaci maƙiyanku, ku kuma yi wa kowa fatan alheri da salama.", "lang": "Hausa", "gold_safety": "non_hate_speech", "desc": "Love enemies and wish peace"},
    {"text": "Muna son zaman lafiya da ci gaba a cikin al'ummarmu domin kowa ya samu kwanciyar hankali.", "lang": "Hausa", "gold_safety": "non_hate_speech", "desc": "Community peace and mental tranquility"},
]

# Experimental Customer Support & Chatbot Intent Test Suite
INTENT_TEST_SUITE = [
    # Greeting
    {"text": "Mema wo akye oo, me ba ha sɛ me rehwɛ biribi.", "lang": "Twi", "gold_intent": "greeting"},
    {"text": "E kaaro o, mo ki gbogbo ile.", "lang": "Yoruba", "gold_intent": "greeting"},
    {"text": "Habari za asubuhi, natumai mko wazima.", "lang": "Swahili", "gold_intent": "greeting"},
    {"text": "Ina kwana, barka da safiya.", "lang": "Hausa", "gold_intent": "greeting"},

    # Complaint
    {"text": "Mo service no yɛ bɔne dodo, me nni anigye biara ho.", "lang": "Twi", "gold_intent": "complaint"},
    {"text": "Iṣẹ ti ẹ n ṣe yii ko dara rara, o n binu mi gidigidi.", "lang": "Yoruba", "gold_intent": "complaint"},
    {"text": "Huduma yenu ni mbaya sana, nimekasirishwa na jinsi nilivyohudumiwa.", "lang": "Swahili", "gold_intent": "complaint"},
    {"text": "Aikin ku ba shi da kyau ko kadan, wannan abu yana bata min rai.", "lang": "Hausa", "gold_intent": "complaint"},

    # Refund / Cancellation
    {"text": "Me pɛ sɛ mo gyae me subscription no na mo san de me sika ma me.", "lang": "Twi", "gold_intent": "cancellation_refund"},
    {"text": "Mo fẹ fagilee adehun mi ki ẹ si da owo mi pada fun mi lẹsẹkẹsẹ.", "lang": "Yoruba", "gold_intent": "cancellation_refund"},
    {"text": "Ninataka kusitisha huduma hii naomba mnirudishie pesa zangu mara moja.", "lang": "Swahili", "gold_intent": "cancellation_refund"},
    {"text": "Ina son soke wannan sabis din kuma ku mayar min da kudina yanzu.", "lang": "Hausa", "gold_intent": "cancellation_refund"},

    # Billing / Payment
    {"text": "Mo atwa me sika wɔ momo so nanso me nnyae me credit no.", "lang": "Twi", "gold_intent": "billing_payment"},
    {"text": "Mo ti sanwo lati ori apo mi ṣugbọn ẹ ko fi owo naa sinu akọọlẹ mi.", "lang": "Yoruba", "gold_intent": "billing_payment"},
    {"text": "Nimeshalipa kwa njia ya simu lakini salio langu halijaongezeka.", "lang": "Swahili", "gold_intent": "billing_payment"},
    {"text": "Na tura kudi ta hanyar asusun banki amma ba ku sanya a ciki ba.", "lang": "Hausa", "gold_intent": "billing_payment"},

    # Technical Support
    {"text": "Me application no nkɔ, ɛkyerɛ error bi bere a me bue no.", "lang": "Twi", "gold_intent": "technical_support"},
    {"text": "Eto kọmputa mi ko ṣiṣẹ, o n kọ aṣiṣe kan nigbati mo ba ṣi i.", "lang": "Yoruba", "gold_intent": "technical_support"},
    {"text": "Programu yangu ya simu haifunguki, inaleta hitilafu ya mtandao kila wakati.", "lang": "Swahili", "gold_intent": "technical_support"},
    {"text": "Manhajata ba ta aiki, tana nuna kuskure a duk lokacin da na bude.", "lang": "Hausa", "gold_intent": "technical_support"},

    # Inquiry / Information
    {"text": "Ɛhe na me tumi kɔpɛ nkyerɛkyerɛmu fa mo dwumadi foforo no ho?", "lang": "Twi", "gold_intent": "inquiry"},
    {"text": "Nibo ni mo ti le ri alaye siwaju sii nipa awọn iṣẹ titun yin?", "lang": "Yoruba", "gold_intent": "inquiry"},
    {"text": "Je, ninaweza kupata wapi maelezo zaidi kuhusu vifurushi vyenu vipya?", "lang": "Swahili", "gold_intent": "inquiry"},
    {"text": "Ina zan iya samun karin bayani game da sabon tsarin ku?", "lang": "Hausa", "gold_intent": "inquiry"},

    # Farewell
    {"text": "Yoo, mede aseda ma mo, yɛbɛhyia bio.", "lang": "Twi", "gold_intent": "farewell"},
    {"text": "O da bo, o ṣe pupọ fun iranlọwọ rẹ.", "lang": "Yoruba", "gold_intent": "farewell"},
    {"text": "Asante sana, kwaheri na siku njema.", "lang": "Swahili", "gold_intent": "farewell"},
    {"text": "Nagode sosai, sai wani lokaci, sai anjima.", "lang": "Hausa", "gold_intent": "farewell"},
]


def write_flac(audio_bytes: bytes, dst_path: str, target_sr: int = 16000) -> bool:
    try:
        with io.BytesIO(audio_bytes) as buf:
            data, sr = sf.read(buf, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            data = soxr.resample(data, sr, target_sr)
        sf.write(dst_path, data, target_sr, format="FLAC", subtype="PCM_16")
        return True
    except Exception:
        return False


def get_gemini_client():
    load_dotenv("/mnt/volume_d2wey28/projects/afrispeech-synth/.env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment!")
    return genai.Client(api_key=api_key)


def annotate_with_gemini(client, text: str, language: str) -> Optional[dict]:
    prompt = f"""You are an expert linguist and NLP evaluator. Analyze the following sentence in {language}:
"{text}"

Classify it strictly according to these categories:
1. domain (must be exactly one of: [agriculture, healthcare, governance_politics, business_finance, education, religion_spirituality, culture_entertainment, environment_climate, legal_justice, technology, customer_support])
2. sentiment (must be exactly one of: [positive, negative, neutral])
3. emotion (must be exactly one of: [joy, sadness, anger, fear, surprise, hope, frustration, neutral])

Return ONLY a valid JSON object matching this schema:
{{"english_translation": "...", "domain": "...", "sentiment": "...", "emotion": "..."}}
"""
    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        raw_text = resp.text.strip()
        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        return json.loads(raw_text)
    except Exception:
        return None


def collect_benchmark_samples(data_dir: Path, client, samples_per_lang: int = 5) -> List[dict]:
    audio_dir = data_dir / "benchmark_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    gold_file = data_dir / "benchmark_gold.jsonl"

    samples = []
    counts_by_lang: Dict[str, int] = {}
    if gold_file.exists():
        with open(gold_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    samples.append(item)
                    key = f"{item['corpus']}_{item['iso']}"
                    counts_by_lang[key] = counts_by_lang.get(key, 0) + 1
        print(f"Loaded {len(samples):,} existing annotations from {gold_file}.")

    print("=" * 65)
    print("Sampling Non-Religious Audio from WaxalNLP & FLEURS & Annotating with Gemini")
    print("=" * 65)

    hf_api = HfApi()
    waxal_info = hf_api.dataset_info("google/WaxalNLP", files_metadata=True)
    waxal_parquets = [f.rfilename for f in waxal_info.siblings if f.rfilename.endswith(".parquet") and "test" in f.rfilename]

    for corpus, cfg, lang_name, iso in tqdm(BENCHMARK_CONFIGS, desc="Datasets"):
        key = f"{corpus}_{iso}"
        existing_count = counts_by_lang.get(key, 0)
        if existing_count >= samples_per_lang:
            print(f"  ✓ {lang_name:18s} ({corpus}): already has {existing_count} samples. Skipping.")
            continue

        rows_iter = []
        if corpus == "waxal":
            code = cfg.split("_")[0]
            matches = [f for f in waxal_parquets if f"/{code}/" in f or f"/{code}-" in f]
            if matches:
                try:
                    p = hf_hub_download("google/WaxalNLP", matches[0], repo_type="dataset")
                    df = pd.read_parquet(p)
                    rows_iter = df.to_dict(orient="records")
                except Exception as e:
                    print(f"  Warning: could not download waxal parquet {matches[0]}: {e}")
        else:
            # FLEURS
            try:
                ds = load_dataset("google/fleurs", cfg, split="test", streaming=True)
                ds = ds.cast_column("audio", Audio(decode=False))
                rows_iter = ds
            except Exception as e:
                print(f"  Warning: could not stream fleurs {cfg}: {e}")

        count = existing_count
        for row in rows_iter:
            text = row.get("transcription") or row.get("text", "")
            if not text or len(text) < 20:
                continue

            audio_data = row.get("audio", {})
            audio_bytes = audio_data.get("bytes") if isinstance(audio_data, dict) else None
            if not audio_bytes:
                continue

            dst_audio = audio_dir / f"{corpus}_{iso}_{count:03d}.flac"
            if not dst_audio.exists():
                success = write_flac(audio_bytes, str(dst_audio))
                if not success:
                    continue

            # Annotate with Gemini
            ann = annotate_with_gemini(client, text, lang_name)
            if not ann:
                continue

            item = {
                "id": f"{corpus}_{iso}_{count:03d}",
                "corpus": corpus,
                "config": cfg,
                "language": lang_name,
                "iso": iso,
                "native_text": text,
                "audio_path": str(dst_audio),
                "gold_domain": ann.get("domain", "general"),
                "gold_sentiment": ann.get("sentiment", "neutral"),
                "gold_emotion": ann.get("emotion", "neutral"),
                "english_translation": ann.get("english_translation", ""),
            }
            samples.append(item)
            with open(gold_file, "a", encoding="utf-8") as gf:
                gf.write(json.dumps(item, ensure_ascii=False) + "\n")
                gf.flush()
            count += 1
            if count >= samples_per_lang:
                break

        print(f"  ✓ {lang_name:18s} ({corpus}): {count} total samples annotated.")

    print(f"\nAll benchmark samples ready ({len(samples):,} items in {gold_file})")
    return samples


def run_benchmark(model_path: str, data_dir: str, device: str = "cpu"):
    data_dir_path = Path(data_dir)
    client = get_gemini_client()
    samples = collect_benchmark_samples(data_dir_path, client, samples_per_lang=5)

    print("\n" + "=" * 65)
    print("STEP 2: Benchmarking africlassifier")
    print(f"Model Path: {model_path}")
    print(f"Device:     {device}")
    print("=" * 65)

    classifier = AfriClassifier.load(
        model_path=model_path,
        taxonomy_index_path=str(data_dir_path / "taxonomies_index.pt"),
        device=device,
    )

    # 1. Benchmark Spoken Audio Modality
    print("\n" + "-" * 60)
    print(f"Running Benchmark on Spoken Audio ({len(samples)} clips across {len(BENCHMARK_CONFIGS)} languages)...")
    print("-" * 60)

    audio_lang_top1 = 0
    audio_lang_top3 = 0
    audio_dom_top1 = 0
    audio_dom_top3 = 0
    audio_sent_acc = 0
    audio_emo_acc = 0
    audio_latencies = []

    for s in tqdm(samples, desc="Audio Benchmark"):
        clip_p = s["audio_path"]
        gold_lang = s["language"].lower()
        gold_dom = s["gold_domain"].lower()
        gold_sent = s["gold_sentiment"].lower()
        gold_emo = s["gold_emotion"].lower()

        t0 = time.time()
        res = classifier.predict(input_data=clip_p, language=None, top_k=3)
        audio_latencies.append((time.time() - t0) * 1000)

        top1_lang = res.detected_language.label.lower()
        pred_langs_top3 = [k.label.lower() for k in res.language.top_k]

        if gold_lang in top1_lang or top1_lang in gold_lang or (gold_lang == "akan" and "twi" in top1_lang):
            audio_lang_top1 += 1
        if any(gold_lang in l or l in gold_lang or (gold_lang == "akan" and "twi" in l) for l in pred_langs_top3):
            audio_lang_top3 += 1

        top1_dom = res.domain.top_1.label.lower()
        top3_doms = [k.label.lower() for k in res.domain.top_k]
        if top1_dom in gold_dom or gold_dom in top1_dom:
            audio_dom_top1 += 1
        if any(d in gold_dom or gold_dom in d for d in top3_doms):
            audio_dom_top3 += 1

        if res.sentiment.top_1.label.lower() == gold_sent:
            audio_sent_acc += 1
        if res.emotion.top_1.label.lower() == gold_emo:
            audio_emo_acc += 1

    total = len(samples)

    # 2. Benchmark Native Text Modality
    print("\n" + "-" * 60)
    print(f"Running Benchmark on Native African Text ({len(samples)} sentences)...")
    print("-" * 60)

    text_dom_top1 = 0
    text_dom_top3 = 0
    text_sent_acc = 0
    text_emo_acc = 0
    text_latencies = []

    for s in tqdm(samples, desc="Text Benchmark"):
        txt = s["native_text"]
        lang = s["language"]
        gold_dom = s["gold_domain"].lower()
        gold_sent = s["gold_sentiment"].lower()
        gold_emo = s["gold_emotion"].lower()

        t0 = time.time()
        res = classifier.predict(input_data=txt, language=lang, top_k=3)
        text_latencies.append((time.time() - t0) * 1000)

        top1_dom = res.domain.top_1.label.lower()
        top3_doms = [k.label.lower() for k in res.domain.top_k]
        if top1_dom in gold_dom or gold_dom in top1_dom:
            text_dom_top1 += 1
        if any(d in gold_dom or gold_dom in d for d in top3_doms):
            text_dom_top3 += 1

        if res.sentiment.top_1.label.lower() == gold_sent:
            text_sent_acc += 1
        if res.emotion.top_1.label.lower() == gold_emo:
            text_emo_acc += 1

    # 3. Benchmark Safety / Hate Speech & Violence Content Moderation
    print("\n" + "-" * 60)
    print(f"Running Safety / Hate Speech & Violence Detection Benchmark ({len(HATE_SPEECH_TEST_SUITE)} samples)...")
    print("-" * 60)

    safety_correct = 0
    safety_table = []

    for item in HATE_SPEECH_TEST_SUITE:
        txt = item["text"]
        lang = item["lang"]
        gold = item["gold_safety"]

        res = classifier.predict(input_data=txt, language=lang, top_k=1)
        pred = res.safety.top_1.label
        score = res.safety.top_1.score

        is_correct = (pred == gold)
        if is_correct:
            safety_correct += 1

        safety_table.append({
            "language": lang,
            "description": item["desc"],
            "gold_safety": gold,
            "predicted_safety": pred,
            "score": score,
            "correct": is_correct,
        })

    safety_acc = safety_correct / len(HATE_SPEECH_TEST_SUITE)

    # 4. Customer Support Intent Suite (Experimental)
    print("\n" + "-" * 60)
    print(f"Running Customer Support Intent Evaluation (Experimental, {len(INTENT_TEST_SUITE)} queries)...")
    print("-" * 60)

    intent_correct = 0
    intent_top2 = 0
    intent_results_table = []

    for item in INTENT_TEST_SUITE:
        txt = item["text"]
        lang = item["lang"]
        gold = item["gold_intent"]

        res = classifier.predict(input_data=txt, language=lang, top_k=2)
        top1 = res.intent.top_1.label
        top2 = [x.label for x in res.intent.top_k]

        is_match = (top1 == gold)
        if is_match:
            intent_correct += 1
        if gold in top2:
            intent_top2 += 1

        intent_results_table.append({
            "language": lang,
            "query": txt,
            "gold_intent": gold,
            "predicted_intent": top1,
            "score": res.intent.top_1.score,
            "correct": is_match,
        })

    intent_acc = intent_correct / len(INTENT_TEST_SUITE)
    intent_acc_top2 = intent_top2 / len(INTENT_TEST_SUITE)

    # Print Final Summary
    print("\n" + "=" * 65)
    print("FINAL BENCHMARK SUMMARY (africlassifier)")
    print("=" * 65)
    print("1. SPOKEN AUDIO BENCHMARK (Non-Religious Waxal & FLEURS):")
    print(f"   - Language ID (Top-1):      {audio_lang_top1 / total:.1%}")
    print(f"   - Language ID (Top-3):      {audio_lang_top3 / total:.1%}")
    print(f"   - Domain / Topic (Top-1):   {audio_dom_top1 / total:.1%}")
    print(f"   - Domain / Topic (Top-3):   {audio_dom_top3 / total:.1%}")
    print(f"   - Sentiment Accuracy:       {audio_sent_acc / total:.1%}")
    print(f"   - Emotion Accuracy:         {audio_emo_acc / total:.1%}")
    print(f"   - Average Latency:          {np.mean(audio_latencies):.1f} ms")

    print("\n2. NATIVE AFRICAN TEXT BENCHMARK:")
    print(f"   - Domain / Topic (Top-1):   {text_dom_top1 / total:.1%}")
    print(f"   - Domain / Topic (Top-3):   {text_dom_top3 / total:.1%}")
    print(f"   - Sentiment Accuracy:       {text_sent_acc / total:.1%}")
    print(f"   - Emotion Accuracy:         {text_emo_acc / total:.1%}")
    print(f"   - Average Latency:          {np.mean(text_latencies):.1f} ms")

    print("\n3. SAFETY / HATE SPEECH & VIOLENCE DETECTION:")
    print(f"   - Safety Classification:    {safety_acc:.1%} ({safety_correct}/{len(HATE_SPEECH_TEST_SUITE)})")

    print("\n4. CUSTOMER SUPPORT INTENT (EXPERIMENTAL):")
    print(f"   - Intent Detection (Top-1): {intent_acc:.1%} ({intent_correct}/{len(INTENT_TEST_SUITE)})")
    print(f"   - Intent Detection (Top-2): {intent_acc_top2:.1%} ({intent_top2}/{len(INTENT_TEST_SUITE)})")
    print("=" * 65)

    results_summary = {
        "audio_benchmark": {
            "total_samples": total,
            "language_id_top1": round(audio_lang_top1 / total, 4),
            "language_id_top3": round(audio_lang_top3 / total, 4),
            "domain_top1": round(audio_dom_top1 / total, 4),
            "domain_top3": round(audio_dom_top3 / total, 4),
            "sentiment_accuracy": round(audio_sent_acc / total, 4),
            "emotion_accuracy": round(audio_emo_acc / total, 4),
            "avg_latency_ms": round(float(np.mean(audio_latencies)), 2),
        },
        "text_benchmark": {
            "total_samples": total,
            "domain_top1": round(text_dom_top1 / total, 4),
            "domain_top3": round(text_dom_top3 / total, 4),
            "sentiment_accuracy": round(text_sent_acc / total, 4),
            "emotion_accuracy": round(text_emo_acc / total, 4),
            "avg_latency_ms": round(float(np.mean(text_latencies)), 2),
        },
        "safety_benchmark": {
            "total_samples": len(HATE_SPEECH_TEST_SUITE),
            "accuracy": round(safety_acc, 4),
            "details": safety_table,
        },
        "intent_experimental": {
            "total_samples": len(INTENT_TEST_SUITE),
            "accuracy_top1": round(intent_acc, 4),
            "accuracy_top2": round(intent_acc_top2, 4),
            "details": intent_results_table,
        }
    }

    results_file = data_dir_path / "benchmark_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved benchmark summary to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive multi-task benchmark.")
    parser.add_argument("--model_path", type=str, default="outputs/embeddinggemma_lora")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_benchmark(args.model_path, args.data_dir, device=args.device)

"""Standardized Multi-Task Taxonomies for africlassifier.

Covers 5 core NLP classification tasks:
1. Language Identification (100+ African languages + English + French)
2. Domain / Topic Classification
3. Sentiment Analysis
4. Emotion Recognition
5. Intent Detection (Customer Support & Chatbot Interaction)
"""

DOMAINS = {
    "agriculture": "agriculture, farming, crops, cocoa, harvest, livestock, soil, food production, planting, veterinary",
    "healthcare": "healthcare, medicine, sickness, illness, hospital, clinic, doctors, disease, infection, treatment, pharmacy",
    "governance_politics": "politics, government, parliament, elections, voting, politicians, president, minister, public policy",
    "business_finance": "business, money, economy, banking, market, trade, financial prices, investment, inflation, commerce",
    "education": "education, school, university, students, teachers, learning, classroom, study, academic, exams",
    "religion_spirituality": "religion, worship, prayer, god, church, mosque, faith, bible, quran, spiritual beliefs, sermon",
    "culture_entertainment": "culture, music, entertainment, movies, art, festivals, tradition, sports, football, dance, heritage",
    "environment_climate": "environment, weather, rain, drought, climate change, forest, wildlife, ocean, ecology, pollution",
    "legal_justice": "court, judge, law, legal rights, justice, police, constitution, lawsuit, crime, lawyer",
    "technology": "technology, computers, mobile phones, internet, software, digital telecom, artificial intelligence, apps",
    "customer_support": "customer service, customer support, client relations, user inquiry, company helpline",
}

SENTIMENTS = {
    "positive": "positive, happy, satisfied, pleased, wonderful, grateful, beneficial, excellent, favorable",
    "negative": "negative, bad, angry, disappointed, terrible, complaint, harmful, unfair, tragic, dissatisfied",
    "neutral": "neutral, objective, factual, informational, matter-of-fact, balanced, neither good nor bad",
}

EMOTIONS = {
    "joy": "joy, happiness, excitement, cheerful, smiling, laughter, delight, celebration",
    "sadness": "sadness, grief, sorrow, crying, depression, heartbreak, despair, mourning",
    "anger": "anger, rage, furious, outrage, shouting, hostility, irritation, resentment",
    "fear": "fear, anxiety, panic, terror, dread, frightened, scared, apprehension",
    "surprise": "surprise, shock, astonishment, amazement, unexpected, startled, stunned",
    "hope": "hope, optimism, encouragement, faith, looking forward, reassurance, aspiration",
    "frustration": "frustration, annoyance, disappointment, impatience, exasperation, fed up",
    "neutral": "neutral, calm, composed, relaxed, steady, standard emotion",
}

INTENTS = {
    "greeting": "greeting, saying hello, good morning, good afternoon, hi, welcome, start conversation",
    "inquiry": "asking for information, asking questions, how to, where is, what is, inquiring, help needed",
    "complaint": "making a complaint, problem with service, expressing dissatisfaction, reporting an issue, protest",
    "billing_payment": "billing, payment, transaction, price, charges, fees, mobile money, invoice, pay",
    "technical_support": "technical support, system error, app not working, fix, connection problem, bug",
    "booking_request": "requesting a service, booking an appointment, ordering, reservation, scheduling",
    "feedback_appreciation": "giving feedback, saying thank you, appreciation, compliment, rating, praise",
    "cancellation_refund": "requesting a refund, return my money, cancel service, cancel subscription, dispute charges",
    "farewell": "saying goodbye, farewell, see you later, bye, end conversation",
}

# The comprehensive 100+ African languages catalog
from africlassifier.languages import ISO_TO_NAME

LANGUAGES = {
    name: f"{name} language, spoken in Africa"
    for iso, name in ISO_TO_NAME.items()
}
LANGUAGES["English"] = "English language speech"
LANGUAGES["French"] = "French language speech, langue française"

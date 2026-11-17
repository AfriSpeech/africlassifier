"""Standardized Multi-Task Taxonomies for africlassifier.

Covers 6 core NLP classification tasks:
1. Language Identification (100+ African languages + English + French)
2. Domain / Topic Classification
3. Sentiment Analysis
4. Emotion Recognition
5. Intent Detection (Customer Support & Chatbot Interaction)
6. Safety & Content Moderation (Hate Speech / Violence Detection)

Design Principle:
Each class is identified by a short, canonical label key (e.g. 'agriculture', 'joy', 'hate_speech'),
which maps to a rich, natural English sentence encoded by our fine-tuned foundation model.
Embedding models perform significantly better with grammatically coherent, descriptive sentences
than isolated comma-separated keywords.
"""

DOMAINS = {
    "agriculture": "This discussion is about agriculture, farming crops, harvesting food, livestock, soil, and agricultural practices.",
    "healthcare": "This discussion is about healthcare, medicine, treatment of sickness, hospitals, doctors, and medical wellness.",
    "governance_politics": "This discussion is about government, politics, national elections, public policy, parliament, and leadership.",
    "business_finance": "This discussion is about business, money, banking, trade, financial markets, commerce, and economic activity.",
    "education": "This discussion is about education, teaching students in schools, universities, academic studies, and learning.",
    "religion_spirituality": "This discussion is about religion, prayer, faith in God, church or mosque worship, scriptures, and spiritual beliefs.",
    "culture_entertainment": "This discussion is about culture, traditional heritage, music, entertainment, sports, arts, and festivals.",
    "environment_climate": "This discussion is about the natural environment, weather conditions, rainfall, forests, wildlife, and climate change.",
    "legal_justice": "This discussion is about legal matters, the judicial court system, laws, justice, human rights, and the police.",
    "technology": "This discussion is about technology, computing devices, mobile phones, software applications, and the internet.",
    "customer_support": "This discussion is about customer service, resolving user inquiries, client support, and company assistance.",
}

SENTIMENTS = {
    "positive": "The speaker is expressing positive sentiment, happiness, satisfaction, gratitude, and approval.",
    "negative": "The speaker is expressing negative sentiment, dissatisfaction, anger, sadness, criticism, or distress.",
    "neutral": "The speaker is expressing neutral, factual, objective information without strong positive or negative emotion.",
}

EMOTIONS = {
    "joy": "The speaker feels joy, delight, excitement, and cheerful happiness.",
    "sadness": "The speaker feels sadness, sorrow, mourning, grief, or heartbreak.",
    "anger": "The speaker feels anger, fury, rage, indignation, and resentment.",
    "fear": "The speaker feels fear, anxiety, terror, panic, and apprehension.",
    "surprise": "The speaker feels surprise, astonishment, amazement, or shock.",
    "hope": "The speaker feels hope, optimism, faith, encouragement, and aspiration for the future.",
    "frustration": "The speaker feels frustration, annoyance, impatience, and exasperation.",
    "neutral": "The speaker feels calm, composed, objective, and neutral.",
}

INTENTS = {
    "greeting": "The user is greeting someone, saying hello, or opening a conversation.",
    "inquiry": "The user is asking a question to request information, guidance, or clarification.",
    "complaint": "The user is making a complaint to report a problem or express dissatisfaction with a service.",
    "billing_payment": "The user is asking about bills, payments, account charges, transaction fees, or mobile money transfers.",
    "technical_support": "The user is asking for technical assistance to fix an error, bug, or malfunctioning application.",
    "booking_request": "The user wants to make a booking, schedule an appointment, or place a new order.",
    "feedback_appreciation": "The user is giving feedback, expressing gratitude, or saying thank you for good service.",
    "cancellation_refund": "The user wants to cancel their subscription or service and request a full refund of their money.",
    "farewell": "The user is saying goodbye, expressing farewell, and closing the conversation.",
}

SAFETY = {
    "hate_speech": "This content contains hate speech, violent threats, calls to kill or destroy enemies, bloodshed, or hostility.",
    "non_hate_speech": "This content is peaceful, non-violent, safe, encouraging, friendly, and respectful.",
}

# The comprehensive 100+ African languages catalog
from africlassifier.languages import ISO_TO_NAME

LANGUAGES = {
    name: f"This is a person speaking in the {name} language of Africa."
    for iso, name in ISO_TO_NAME.items()
}
LANGUAGES["English"] = "This is a person speaking in the English language."
LANGUAGES["French"] = "This is a person speaking in the French language."


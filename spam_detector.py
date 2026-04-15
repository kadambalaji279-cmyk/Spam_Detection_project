import pandas as pd
import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# =========================
# 1. LOAD DATASET
# =========================
print("Loading dataset...")
dataset = load_dataset("sms_spam")
df = dataset["train"].to_pandas()

df = df.rename(columns={"sms": "message", "label": "label"})

# =========================
# 2. CLEAN TEXT
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["message"] = df["message"].apply(clean_text)

# =========================
# 3. SPLIT DATA
# =========================
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. TF-IDF
# =========================
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    max_features=8000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# 5. TRAIN MODEL
# =========================
model = LinearSVC()
model.fit(X_train_vec, y_train)

# =========================
# 6. ACCURACY
# =========================
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🔥 Model Accuracy: {accuracy*100:.2f}%")

# =========================
# 7. SPAM KEYWORDS (UNCHANGED ✅)
# =========================
spam_keywords = [

    "free", "offer", "limited", "deal", "discount", "sale",
    "buy now", "best price", "special offer", "exclusive deal",

    "win", "winner", "won", "cash", "prize", "money",
    "earn", "income", "profit", "bonus", "reward",
    "earn money", "make money", "quick cash",

    "lottery", "jackpot", "lucky draw", "congratulations",
    "selected", "claim now",

    "click", "click here", "visit link", "verify",
    "update account", "login now", "confirm account",
    "reset password",

    "urgent", "immediately", "act now", "limited time",
    "hurry", "expires soon", "last chance",

    "work from home", "part time job", "online job",
    "no investment", "easy job", "daily income",

    "free recharge", "mobile recharge", "data offer",
    "unlimited data",

    "bank alert", "account blocked", "verify bank",
    "update kyc", "atm blocked", "credit alert",

    "guaranteed", "risk free", "no cost", "zero investment",

    "weight loss", "lose weight fast", "miracle cure",

    "loan approved", "instant loan", "easy loan", "credit card offer"
]

# =========================
# 8. PREDICTION FUNCTION (FOR FLASK)
# =========================
def predict_message(msg):
    msg_clean = clean_text(msg)

    # Keyword detection
    for word in spam_keywords:
        if word in msg_clean:
            return "🚨 SPAM MESSAGE (Keyword Detected)"

    # ML prediction
    vec = vectorizer.transform([msg_clean])
    pred = model.predict(vec)[0]

    if pred == 1:
        return "🚨 SPAM MESSAGE (ML Detected)"
    else:
        return "✅ NOT SPAM"

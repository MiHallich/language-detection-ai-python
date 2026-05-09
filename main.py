import os
import random


def text_to_vector(text):
    text = text.lower()
    counts = [0] * 26
    total = 0

    for char in text:
        if 'a' <= char <= 'z':
            counts[ord(char) - ord('a')] += 1
            total += 1

    return [c / total for c in counts] if total > 0 else counts


def load_data(path):
    data = []

    for lang in os.listdir(path):
        lang_path = os.path.join(path, lang)

        for filename in os.listdir(lang_path):
            file_path = os.path.join(lang_path, filename)

            with open(file_path, encoding="utf-8") as f:
                text = f.read()
                vec = text_to_vector(text)
                data.append((vec, lang))

    return data


def init_model(languages):
    model = {}
    for lang in languages:
        model[lang] = {
            "w": [0.0] * 26,
            "b": 0.0
        }
    return model


def predict_single(weights, bias, x):
    s = bias
    for i in range(26):
        s += weights[i] * x[i]
    return s


def predict(model, x):
    best_lang = None
    best_score = -float('inf')

    for lang in model:
        score = predict_single(model[lang]["w"], model[lang]["b"], x)
        if score > best_score:
            best_score = score
            best_lang = lang

    return best_lang


def train(model, data, epochs, lr):
    for h in range(epochs):
        random.shuffle(data)

        for x, true_lang in data:
            for lang in model:
                y = 1 if lang == true_lang else 0
                pred = predict_single(model[lang]["w"], model[lang]["b"], x)

                error = y - pred

                for i in range(26):
                    model[lang]["w"][i] += lr * error * x[i]

                model[lang]["b"] += lr * error


def manual_input(model):
    text = input("\nEnter text: ")
    vec = text_to_vector(text)
    print("Predicted language:", predict(model, vec))


def file_input(model):
    path = input("\nEnter file path: ")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        vec = text_to_vector(text)
        print("Predicted language:", predict(model, vec))
    except:
        print("File read error!")



languages = ["en", "pl", "de", "es"]

train_data = load_data("data/train")
test_data = load_data("data/test")

print("Train samples:", len(train_data))
print("Test samples:", len(test_data))

model = init_model(languages)

print("\nTraining")
train(model, train_data, epochs=80, lr=0.05)

print("\nTesting:")
correct = 0
random.shuffle(test_data)
for x, lang in test_data:
    pred = predict(model, x)
    print(f"True: {lang}  Pred: {pred}")

    if pred == lang:
        correct += 1

print(f"\nAccuracy: {correct}/{len(test_data)}")

while True:
    print("\n1 - Enter text")
    print("2 - Load file")
    print("0 - Exit")

    choice = input("Choice: ")

    if choice == "1":
        manual_input(model)
    elif choice == "2":
        file_input(model)
    elif choice == "0":
        break
    else:
        print("Invalid input!")

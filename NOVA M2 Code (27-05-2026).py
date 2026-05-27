import nltk
from nltk.metrics import precision, recall
from nltk import ConfusionMatrix




def evaluate_classifier(classifier, test_set):


    gold_labels = []
    predicted_labels = []

    reference_sets = {}
    test_sets = {}



    for i, (features, actual_label) in enumerate(test_set):


        predicted_label = classifier.classify(features)


        gold_labels.append(actual_label)
        predicted_labels.append(predicted_label)


        reference_sets.setdefault(actual_label, set())
        test_sets.setdefault(predicted_label, set())


        reference_sets[actual_label].add(i)
        test_sets[predicted_label].add(i)


    accuracy = nltk.classify.accuracy(classifier, test_set)



    metrics_report = {}

    labels = set(gold_labels)

    for label in labels:

        metrics_report[label] = {

            "precision":
                precision(reference_sets[label],
                          test_sets[label]),

            "recall": recall(reference_sets[label],
                       test_sets[label])
        }



    cm = ConfusionMatrix(gold_labels,
                         predicted_labels)



    report = {

        "accuracy": accuracy,

        "metrics": metrics_report,

        "confusion_matrix": cm
    }

    return report




train_data = [

    ({"word": "offer"}, "spam"),
    ({"word": "win"}, "spam"),

    ({"word": "meeting"}, "ham"),
    ({"word": "project"}, "ham")
]


classifier = nltk.NaiveBayesClassifier.train(train_data)



test_set = [

    ({"word": "offer"}, "spam"),
    ({"word": "win"}, "spam"),

    ({"word": "meeting"}, "ham"),
    ({"word": "project"}, "ham"),

    ({"word": "offer"}, "spam"),
    ({"word": "meeting"}, "ham")
]




result = evaluate_classifier(classifier,
                             test_set)



print("Accuracy:")
print(result["accuracy"])

print("\nPrecision and Recall:")

for label in result["metrics"]:

    print(label,
          result["metrics"][label])

print("\nConfusion Matrix:")
print(result["confusion_matrix"])

# Sample output 1 — default data (perfect accuracy):

# Accuracy:
# 1.0

# Precision and Recall:
# spam {'precision': 1.0, 'recall': 1.0}
# ham {'precision': 1.0, 'recall': 1.0}

# Confusion Matrix:
#      |   s |
#      | h p |
#      | a a |
#      | m m |
# -----+-----+
#  ham |<3>. |
# spam | .<3>|
# -----+-----+
# (row = reference; col = test)

# ------------------------------------------------------------

# Sample output 2 — label present in gold but never predicted (example):

# Accuracy:
# 0.5

# Precision and Recall:
# ham {'precision': None, 'recall': 0.0}
# spam {'precision': 1.0, 'recall': 1.0}

# Confusion Matrix:
#      |   s |
#      | h p |
#      | a a |
#      | m m |
# -----+-----+
#  ham | .<0>|
# spam |<1>. |
# -----+-----+
# (row = reference; col = test)

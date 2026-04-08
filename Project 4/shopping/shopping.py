import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def load_data(filename):
    """Load shopping data from a CSV file and convert into evidence and labels."""
    month_to_index = {
        'Jan': 0,
        'Feb': 1,
        'Mar': 2,
        'Apr': 3,
        'May': 4,
        'June': 5,
        'Jul': 6,
        'Aug': 7,
        'Sep': 8,
        'Oct': 9,
        'Nov': 10,
        'Dec': 11,
    }

    evidence = []
    labels = []

    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            evidence.append([
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                month_to_index[row['Month']],
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                1 if row['VisitorType'] == 'Returning_Visitor' else 0,
                1 if row['Weekend'].strip().lower() == 'true' else 0,
            ])
            labels.append(1 if row['Revenue'].strip().lower() == 'true' else 0)

    return evidence, labels


def train_model(evidence, labels):
    """Train and return a k-nearest neighbor model (k = 1)."""
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """Return sensitivity and specificity for the predictions."""
    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0

    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            if predicted == 1:
                true_positive += 1
            else:
                false_negative += 1
        else:
            if predicted == 0:
                true_negative += 1
            else:
                false_positive += 1

    sensitivity = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0 else 0
    )

    specificity = (
        true_negative / (true_negative + false_positive)
        if (true_negative + false_positive) > 0 else 0
    )

    return sensitivity, specificity


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    print(sys.argv[0])
    print(sys.argv[1])

    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


if __name__ == "__main__":
    main()

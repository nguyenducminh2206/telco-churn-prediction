from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """
    Evaluates an XGBoost model on test data.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
    """
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, digits=4)
    cm = confusion_matrix(y_test, preds)

    print("-" * 40)
    print("Model evaluation")
    print(f"  test size  : {len(y_test)} rows")
    print("-" * 40)
    print("Classification report")
    print(report)
    print("Confusion matrix")
    print("              pred_0   pred_1")
    print(f"  actual_0  {cm[0, 0]:>7}  {cm[0, 1]:>7}")
    print(f"  actual_1  {cm[1, 0]:>7}  {cm[1, 1]:>7}")
    print("-" * 40)
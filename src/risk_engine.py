def get_risk_message(prediction):
    if prediction == "Low":
        return {
            "level": "LOW",
            "message": "The model indicates a lower-risk pattern based on the provided assessment.",
            "indicator": "🟢"
        }

    elif prediction == "Moderate":
        return {
            "level": "MODERATE",
            "message": "The model indicates a moderate-risk pattern. Further assessment may be appropriate.",
            "indicator": "🟡"
        }

    else:
        return {
            "level": "HIGH",
            "message": "The model indicates a higher-risk pattern. Professional evaluation is recommended.",
            "indicator": "🔴"
        }


def calculate_feature_importance(model, feature_names):
    importance = model.feature_importances_

    results = list(zip(feature_names, importance))

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results
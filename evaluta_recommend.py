def calculate_metrics(recommended_meal, true_meal):

    recommended_set = set(recommended_meal['mealID'].to_list())
    true_set = set(true_meal['MealID'].tolist())
    tp = len(recommended_set.intersection(true_set))
    fp = len(recommended_set) - tp
    fn = len(true_set) - tp
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"召回率: {recall:.2f},精确率: {precision:.2f},F1—Score: {f1:.2f}")

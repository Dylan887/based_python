
def rmt(data):
    return data.pivot(index='UserID_encoded', columns='MealID_encoded', values='Rating').fillna(0)
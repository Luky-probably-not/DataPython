from sklearn.linear_model import LogisticRegression

# Entraînement du modèle
def train_model(x_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    return model
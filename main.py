import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

def create_preprocessing_pipeline(categorical_features, numeric_features):
    column_transformer = ColumnTransformer(transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('scaler', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())]), numeric_features)
    ], remainder='drop')
    return column_transformer

def create_and_compile_model(input_shape):
    model = Sequential([
        Dense(64, input_shape=(input_shape,), activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    categorical_features = ['Source', 'Destination']
    numeric_features = ['Length']

    pipeline = Pipeline(steps=[
        ('preprocessor', create_preprocessing_pipeline(categorical_features, numeric_features)),
    ])

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop('IsDDoS', axis=1)
    y = df['IsDDoS']
    return X, y

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.2)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy*100:.2f}%')

def predict_new_data(model, pipeline, filepath):
    new_data = pd.read_csv(filepath)
    new_data_transformed = pipeline.transform(new_data)
    predictions = model.predict(new_data_transformed)
    predictions = (predictions > 0.5).astype(int)
    new_data['IsDDoS'] = predictions
    new_data.to_csv('tahminlerYeni.csv', index=False)

if __name__ == "__main__":
    X, y = load_and_preprocess_data('CSharpDDosAttackYeni.csv')
    categorical_features = ['Source', 'Destination']
    numeric_features = ['Length']
    pipeline = create_preprocessing_pipeline(categorical_features, numeric_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    model = create_and_compile_model(X_train_transformed.shape[1])
    train_and_evaluate_model(model, X_train_transformed, y_train, X_test_transformed, y_test)

    predict_new_data(model, pipeline, 'DDosAttackICMP.csv')
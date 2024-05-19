import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns

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
        Dense(64, input_shape=(input_shape,), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001))
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop('IsDDoS', axis=1)
    y = df['IsDDoS']
    return X, y

def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall_curve(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    plt.figure()
    plt.step(recall, precision, where='post', label='Average precision (AP = %0.2f)' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve')
    plt.legend(loc="upper right")
    plt.show()

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy*100:.2f}%')

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Predictions for additional plots
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_scores = model.predict(X_test).ravel()

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_scores)
    plot_precision_recall_curve(y_test, y_scores)

def predict_new_data(model, pipeline, filepath):
    new_data = pd.read_csv(filepath)
    new_data_transformed = pipeline.transform(new_data)
    predictions = model.predict(new_data_transformed)
    predictions = (predictions > 0.5).astype(int)
    new_data['IsDDoS'] = predictions
    new_data.to_csv('hebele.csv', index=False)

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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Create directories if not exists
os.makedirs('models', exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_data():
    """Load and preprocess the dataset"""
    # Load training data
    train_data = pd.read_csv('dataset/Training.csv')
    test_data = pd.read_csv('dataset/Testing.csv')
    
    # Clean column names (remove trailing spaces)
    train_data.columns = train_data.columns.str.strip()
    test_data.columns = test_data.columns.str.strip()
    
    # Drop any columns (empty columns from trailing commas)
    train_data = train_data.loc[:, ~train_data.columns.str.contains('^Unnamed')]
    test_data = test_data.loc[:, ~test_data.columns.str.contains('^Unnamed')]
    
    # Separate features and target
    X_train = train_data.drop('prognosis', axis=1)
    y_train = train_data['prognosis']
    X_test = test_data.drop('prognosis', axis=1)
    y_test = test_data['prognosis']
    
    # Encode target labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Save label encoder
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    # Save symptom list
    symptoms = list(X_train.columns)
    with open('models/symptoms.pkl', 'wb') as f:
        pickle.dump(symptoms, f)
    
    return X_train, X_test, y_train_encoded, y_test_encoded, le, symptoms

def train_decision_tree(X_train, X_test, y_train, y_test, le):
    """Train Decision Tree Classifier"""
    print("\n" + "="*50)
    print("Training Decision Tree Classifier...")
    print("="*50)
    
    dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_classifier.fit(X_train, y_train)
    
    # Predictions
    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Decision Tree Accuracy: {accuracy * 100:.2f}%")
    
    # Plot Decision Tree (simplified view)
    plt.figure(figsize=(25, 15))
    plot_tree(dt_classifier, max_depth=3, filled=True, rounded=True,
              feature_names=list(X_train.columns), 
              class_names=le.classes_,
              fontsize=8)
    plt.title('Decision Tree Classifier (Depth Limited to 3 for Visualization)', fontsize=16)
    plt.tight_layout()
    plt.savefig('static/plots/decision_tree.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Decision Tree plot saved!")
    
    # Confusion Matrix for Decision Tree
    plot_confusion_matrix(y_test, y_pred, le, 'Decision Tree')
    
    # Save model
    with open('models/decision_tree.pkl', 'wb') as f:
        pickle.dump(dt_classifier, f)
    
    return dt_classifier, accuracy, y_pred

def train_knn(X_train, X_test, y_train, y_test, le):
    """Train K-Nearest Neighbors Classifier"""
    print("\n" + "="*50)
    print("Training K-Nearest Neighbors Classifier...")
    print("="*50)
    
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)
    
    # Predictions
    y_pred = knn_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"KNN Accuracy: {accuracy * 100:.2f}%")
    
    # Confusion Matrix for KNN
    plot_confusion_matrix(y_test, y_pred, le, 'K-Nearest Neighbors')
    
    # Plot K value analysis
    plot_knn_k_analysis(X_train, X_test, y_train, y_test)
    
    # Save model
    with open('models/knn.pkl', 'wb') as f:
        pickle.dump(knn_classifier, f)
    
    return knn_classifier, accuracy, y_pred

def train_naive_bayes(X_train, X_test, y_train, y_test, le):
    """Train Naive Bayes Classifier"""
    print("\n" + "="*50)
    print("Training Naive Bayes Classifier...")
    print("="*50)
    
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    
    # Predictions
    y_pred = nb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Naive Bayes Accuracy: {accuracy * 100:.2f}%")
    
    # Confusion Matrix for Naive Bayes
    plot_confusion_matrix(y_test, y_pred, le, 'Naive Bayes')
    
    # Save model
    with open('models/naive_bayes.pkl', 'wb') as f:
        pickle.dump(nb_classifier, f)
    
    return nb_classifier, accuracy, y_pred

def train_random_forest(X_train, X_test, y_train, y_test, le, symptoms):
    """Train Random Forest Classifier"""
    print("\n" + "="*50)
    print("Training Random Forest Classifier...")
    print("="*50)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")
    
    # Confusion Matrix for Random Forest
    plot_confusion_matrix(y_test, y_pred, le, 'Random Forest')
    
    # Feature Importance Plot
    plot_feature_importance(rf_classifier, symptoms)
    
    # Save model
    with open('models/random_forest.pkl', 'wb') as f:
        pickle.dump(rf_classifier, f)
    
    return rf_classifier, accuracy, y_pred


def plot_confusion_matrix(y_test, y_pred, le, model_name):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    filename = f'static/plots/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix for {model_name} saved!")


def plot_knn_k_analysis(X_train, X_test, y_train, y_test):
    """Plot KNN accuracy for different K values"""
    k_range = range(1, 21)
    accuracies = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('K Value', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('KNN Accuracy for Different K Values', fontsize=14)
    plt.xticks(k_range)
    plt.grid(True, alpha=0.3)
    
    # Mark the best K
    best_k = k_range[np.argmax(accuracies)]
    best_acc = max(accuracies)
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}')
    plt.scatter([best_k], [best_acc], color='red', s=200, zorder=5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('static/plots/knn_k_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("KNN K-value analysis plot saved!")


def plot_feature_importance(rf_model, symptoms):
    """Plot top feature importances from Random Forest"""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features
    
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [symptoms[i].replace('_', ' ').title() for i in indices])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Symptoms', fontsize=12)
    plt.title('Top 20 Most Important Symptoms (Random Forest)', fontsize=14)
    plt.tight_layout()
    plt.savefig('static/plots/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Feature importance plot saved!")


def plot_accuracy_comparison(results):
    """Plot accuracy comparison bar chart"""
    models = list(results.keys())
    accuracies = [acc * 100 for acc in results.values()]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.xlabel('Machine Learning Algorithm', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/plots/accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Accuracy comparison plot saved!")


def plot_classification_report(y_test, y_pred, le, model_name):
    """Generate and save classification report as image"""
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(report).transpose()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=np.round(df.values, 2),
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc='center',
        loc='center',
        colColours=['#4472C4']*len(df.columns),
        rowColours=['#D9E2F3']*len(df.index)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    
    # Style header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(color='white', fontweight='bold')
        if j == -1:
            cell.set_text_props(fontweight='bold')
    
    plt.title(f'Classification Report - {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    filename = f'static/plots/classification_report_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Classification report for {model_name} saved!")

def get_disease_info():
    """Get disease descriptions and precautions"""
    disease_info = {
        'Fungal infection': {
            'description': 'A fungal infection is a skin disease caused by a fungus.',
            'precautions': ['Bath twice', 'Use detol or neem in bathing water', 'Keep infected area dry', 'Use clean cloths']
        },
        'Allergy': {
            'description': 'An allergy is an immune system response to a foreign substance.',
            'precautions': ['Apply calamine', 'Cover area with bandage', 'Use ice to compress itching']
        },
        'GERD': {
            'description': 'Gastroesophageal reflux disease is a digestive disorder.',
            'precautions': ['Avoid fatty spicy food', 'Avoid lying down after eating', 'Maintain healthy weight', 'Exercise']
        },
        'Chronic cholestasis': {
            'description': 'A condition where bile cannot flow from the liver to the duodenum.',
            'precautions': ['Cold baths', 'Anti-itch medicine', 'Consult doctor', 'Eat healthy']
        },
        'Drug Reaction': {
            'description': 'An adverse reaction to a medication.',
            'precautions': ['Stop irritation', 'Consult nearest hospital', 'Stop taking drug', 'Follow up']
        },
        'Peptic ulcer diseae': {
            'description': 'Sores that develop on the lining of the stomach.',
            'precautions': ['Avoid fatty spicy food', 'Consume probiotic food', 'Eliminate milk', 'Limit alcohol']
        },
        'AIDS': {
            'description': 'Acquired immunodeficiency syndrome caused by HIV.',
            'precautions': ['Avoid open cuts', 'Wear PPE if possible', 'Consult doctor', 'Follow up']
        },
        'Diabetes': {
            'description': 'A metabolic disease causing high blood sugar.',
            'precautions': ['Have balanced diet', 'Exercise', 'Consult doctor', 'Follow up']
        },
        'Gastroenteritis': {
            'description': 'Inflammation of the stomach and intestines.',
            'precautions': ['Stop eating solid food for while', 'Try taking small sips of water', 'Rest', 'Ease back into eating']
        },
        'Bronchial Asthma': {
            'description': 'A condition in which airways narrow and swell.',
            'precautions': ['Switch to loose clothing', 'Take deep breaths', 'Get away from trigger', 'Seek help']
        },
        'Hypertension': {
            'description': 'High blood pressure condition.',
            'precautions': ['Meditation', 'Salt baths', 'Reduce stress', 'Get proper sleep']
        },
        'Migraine': {
            'description': 'A headache of varying intensity.',
            'precautions': ['Meditation', 'Reduce stress', 'Use polaroid glasses in sun', 'Consult doctor']
        },
        'Cervical spondylosis': {
            'description': 'Age-related wear affecting spinal disks in neck.',
            'precautions': ['Use heating pad or cold pack', 'Exercise', 'Take OTC pain reliever', 'Consult doctor']
        },
        'Paralysis (brain hemorrhage)': {
            'description': 'Loss of muscle function due to brain hemorrhage.',
            'precautions': ['Massage', 'Eat healthy', 'Exercise', 'Consult doctor']
        },
        'Jaundice': {
            'description': 'Yellowing of the skin due to high bilirubin.',
            'precautions': ['Drink plenty of water', 'Consume milk thistle', 'Eat fruits and vegetables', 'Medication']
        },
        'Malaria': {
            'description': 'A mosquito-borne infectious disease.',
            'precautions': ['Consult nearest hospital', 'Avoid oily food', 'Avoid non-veg food', 'Keep mosquitoes away']
        },
        'Chicken pox': {
            'description': 'A highly contagious viral infection.',
            'precautions': ['Use neem in bathing', 'Consume neem leaves', 'Take vaccine', 'Avoid public places']
        },
        'Dengue': {
            'description': 'A mosquito-borne tropical disease.',
            'precautions': ['Drink papaya leaf juice', 'Avoid fatty spicy food', 'Keep mosquitoes away', 'Keep hydrated']
        },
        'Typhoid': {
            'description': 'A bacterial infection due to Salmonella typhi.',
            'precautions': ['Eat high calorie vegetables', 'Antibiotic therapy', 'Consult doctor', 'Medication']
        },
        'hepatitis A': {
            'description': 'A highly contagious liver infection.',
            'precautions': ['Consult nearest hospital', 'Wash hands thoroughly', 'Avoid fatty spicy food', 'Medication']
        },
        'Hepatitis B': {
            'description': 'A serious liver infection caused by hepatitis B virus.',
            'precautions': ['Consult nearest hospital', 'Vaccination', 'Eat healthy', 'Medication']
        },
        'Hepatitis C': {
            'description': 'An infection caused by hepatitis C virus.',
            'precautions': ['Consult nearest hospital', 'Vaccination', 'Eat healthy', 'Medication']
        },
        'Hepatitis D': {
            'description': 'A serious liver disease caused by hepatitis D virus.',
            'precautions': ['Consult doctor', 'Medication', 'Eat healthy', 'Follow up']
        },
        'Hepatitis E': {
            'description': 'A liver disease caused by hepatitis E virus.',
            'precautions': ['Stop alcohol consumption', 'Rest', 'Consult doctor', 'Medication']
        },
        'Alcoholic hepatitis': {
            'description': 'Liver inflammation caused by drinking too much alcohol.',
            'precautions': ['Stop alcohol consumption', 'Consult doctor', 'Medication', 'Follow up']
        },
        'Tuberculosis': {
            'description': 'A bacterial infection that mainly affects the lungs.',
            'precautions': ['Cover mouth', 'Consult doctor', 'Medication', 'Rest']
        },
        'Common Cold': {
            'description': 'A viral infection of the upper respiratory tract.',
            'precautions': ['Drink vitamin C rich drinks', 'Take vapour', 'Avoid cold food', 'Keep fever in check']
        },
        'Pneumonia': {
            'description': 'Infection that inflames air sacs in the lungs.',
            'precautions': ['Consult doctor', 'Medication', 'Rest', 'Follow up']
        },
        'Dimorphic hemmorhoids(piles)': {
            'description': 'Swollen veins in the lowest part of rectum.',
            'precautions': ['Avoid fatty spicy food', 'Consume witch hazel', 'Warm bath with epsom salt', 'Consume alovera juice']
        },
        'Heart attack': {
            'description': 'Occurs when blood flow to the heart is blocked.',
            'precautions': ['Call ambulance', 'Chew or swallow aspirin', 'Keep calm']
        },
        'Varicose veins': {
            'description': 'Enlarged, swollen, and twisting veins.',
            'precautions': ['Lie down flat and raise the leg high', 'Use ointments', 'Use vein compression', 'Dont stand still for long']
        },
        'Hypothyroidism': {
            'description': 'Condition where thyroid gland doesnt produce enough hormones.',
            'precautions': ['Reduce stress', 'Exercise', 'Eat healthy', 'Get proper sleep']
        },
        'Hyperthyroidism': {
            'description': 'Overactive thyroid producing too much thyroxine.',
            'precautions': ['Eat healthy', 'Massage', 'Use lemon balm', 'Take radioactive iodine treatment']
        },
        'Hypoglycemia': {
            'description': 'Low blood sugar condition.',
            'precautions': ['Lie down on side', 'Check pulse', 'Drink sugary drinks', 'Consult doctor']
        },
        'Osteoarthristis': {
            'description': 'Degenerative joint disease.',
            'precautions': ['Acetaminophen', 'Consult nearest hospital', 'Follow up', 'Salt baths']
        },
        'Arthritis': {
            'description': 'Inflammation of joints.',
            'precautions': ['Exercise', 'Use hot and cold therapy', 'Try acupuncture', 'Massage']
        },
        '(vertigo) Paroymam Positional Vertigo': {
            'description': 'Brief episodes of mild to intense dizziness.',
            'precautions': ['Lie down', 'Avoid sudden change in body', 'Avoid abrupt head movement', 'Relax']
        },
        'Acne': {
            'description': 'Skin condition with pimples.',
            'precautions': ['Bath twice', 'Avoid fatty spicy food', 'Drink plenty of water', 'Avoid too many products']
        },
        'Urinary tract infection': {
            'description': 'Infection in any part of urinary system.',
            'precautions': ['Drink plenty of water', 'Increase vitamin C intake', 'Drink cranberry juice', 'Take probiotics']
        },
        'Psoriasis': {
            'description': 'Skin disease causing red, itchy scaly patches.',
            'precautions': ['Wash hands with warm soapy water', 'Stop bleeding using pressure', 'Consult doctor', 'Salt baths']
        },
        'Impetigo': {
            'description': 'A highly contagious skin infection.',
            'precautions': ['Soak affected area in warm water', 'Use antibiotics', 'Remove scabs with wet compressed cloth', 'Consult doctor']
        }
    }
    
    # Save disease info
    with open('models/disease_info.pkl', 'wb') as f:
        pickle.dump(disease_info, f)
    
    return disease_info

def train_all_models():
    """Train all models and compare accuracy"""
    print("\n" + "="*60)
    print("MEDICAL HEALTH PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test, le, symptoms = load_data()
    
    print(f"\nDataset loaded successfully!")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of symptoms: {len(symptoms)}")
    print(f"Number of diseases: {len(le.classes_)}")
    
    # Train all models
    results = {}
    predictions = {}
    
    dt_model, dt_acc, dt_pred = train_decision_tree(X_train, X_test, y_train, y_test, le)
    results['Decision Tree'] = dt_acc
    predictions['Decision Tree'] = dt_pred
    
    knn_model, knn_acc, knn_pred = train_knn(X_train, X_test, y_train, y_test, le)
    results['KNN'] = knn_acc
    predictions['KNN'] = knn_pred
    
    nb_model, nb_acc, nb_pred = train_naive_bayes(X_train, X_test, y_train, y_test, le)
    results['Naive Bayes'] = nb_acc
    predictions['Naive Bayes'] = nb_pred
    
    rf_model, rf_acc, rf_pred = train_random_forest(X_train, X_test, y_train, y_test, le, symptoms)
    results['Random Forest'] = rf_acc
    predictions['Random Forest'] = rf_pred
    
    # Get disease info
    get_disease_info()
    
    # Generate comparison plots
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION PLOTS...")
    print("="*60)
    
    # Accuracy comparison chart
    plot_accuracy_comparison(results)
    
    # Classification reports for all models
    for model_name, y_pred in predictions.items():
        plot_classification_report(y_test, y_pred, le, model_name)
    
    # Summary
    print("\n" + "="*60)
    print("MODEL ACCURACY COMPARISON")
    print("="*60)
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy * 100:.2f}%")
    
    best_model = max(results, key=results.get)
    print(f"\nBest Model: {best_model} with {results[best_model] * 100:.2f}% accuracy")
    print("\nAll models saved to 'models/' directory!")
    print("All plots saved to 'static/plots/' directory!")
    
    return results

if __name__ == "__main__":
    train_all_models()

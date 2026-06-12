from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load models and data
def load_models():
    """Load all trained models and data"""
    models = {}
    
    # Load all classifiers
    model_files = {
        'decision_tree': 'models/decision_tree.pkl',
        'knn': 'models/knn.pkl',
        'naive_bayes': 'models/naive_bayes.pkl',
        'random_forest': 'models/random_forest.pkl'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
    
    # Load label encoder
    if os.path.exists('models/label_encoder.pkl'):
        with open('models/label_encoder.pkl', 'rb') as f:
            models['label_encoder'] = pickle.load(f)
    
    # Load symptoms list
    if os.path.exists('models/symptoms.pkl'):
        with open('models/symptoms.pkl', 'rb') as f:
            models['symptoms'] = pickle.load(f)
    
    # Load disease info
    if os.path.exists('models/disease_info.pkl'):
        with open('models/disease_info.pkl', 'rb') as f:
            models['disease_info'] = pickle.load(f)
    
    return models

# Global models dictionary
models = {}

def initialize_models():
    """Initialize models on startup"""
    global models
    models = load_models()

# Sample doctor data (in production, this would come from a database)
DOCTORS = [
    {"id": 1, "name": "Dr. Sarah Johnson", "specialty": "General Physician", "experience": "15 years", "rating": 4.8, "available": True, "image": "doctor1.jpg"},
    {"id": 2, "name": "Dr. Michael Chen", "specialty": "Internal Medicine", "experience": "12 years", "rating": 4.7, "available": True, "image": "doctor2.jpg"},
    {"id": 3, "name": "Dr. Emily Williams", "specialty": "Dermatologist", "experience": "10 years", "rating": 4.9, "available": True, "image": "doctor3.jpg"},
    {"id": 4, "name": "Dr. David Brown", "specialty": "Cardiologist", "experience": "18 years", "rating": 4.8, "available": False, "image": "doctor4.jpg"},
    {"id": 5, "name": "Dr. Jessica Martinez", "specialty": "Pulmonologist", "experience": "8 years", "rating": 4.6, "available": True, "image": "doctor5.jpg"},
    {"id": 6, "name": "Dr. Robert Taylor", "specialty": "Gastroenterologist", "experience": "14 years", "rating": 4.7, "available": True, "image": "doctor6.jpg"},
]

@app.route('/')
def home():
    """Home page with symptom selection"""
    symptoms = models.get('symptoms', [])
    # Format symptoms for display (replace underscores with spaces)
    formatted_symptoms = [s.replace('_', ' ').title() for s in symptoms]
    symptom_pairs = list(zip(symptoms, formatted_symptoms))
    return render_template('index.html', symptoms=symptom_pairs)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict disease based on selected symptoms"""
    try:
        # Get selected symptoms from form
        selected_symptoms = request.form.getlist('symptoms')
        selected_algorithm = request.form.get('algorithm', 'random_forest')
        
        if not selected_symptoms:
            return render_template('result.html', error="Please select at least one symptom.")
        
        # Create input vector
        symptoms_list = models.get('symptoms', [])
        input_vector = np.zeros(len(symptoms_list))
        
        for symptom in selected_symptoms:
            if symptom in symptoms_list:
                idx = symptoms_list.index(symptom)
                input_vector[idx] = 1
        
        # Get the selected model
        model = models.get(selected_algorithm)
        if model is None:
            return render_template('result.html', error="Model not found. Please train the models first.")
        
        # Make prediction
        prediction = model.predict([input_vector])[0]
        
        # Get probabilities if available
        probabilities = {}
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([input_vector])[0]
            # Get top 3 predictions
            top_indices = np.argsort(proba)[-3:][::-1]
            for idx in top_indices:
                disease_name = models['label_encoder'].inverse_transform([idx])[0]
                probabilities[disease_name] = round(proba[idx] * 100, 2)
        
        # Decode prediction
        disease = models['label_encoder'].inverse_transform([prediction])[0]
        
        # Get disease info
        disease_info = models.get('disease_info', {})
        info = disease_info.get(disease, {
            'description': 'No description available.',
            'precautions': ['Consult a doctor for proper diagnosis.']
        })
        
        # Format selected symptoms for display
        formatted_selected = [s.replace('_', ' ').title() for s in selected_symptoms]
        
        return render_template('result.html', 
                             disease=disease,
                             description=info.get('description', ''),
                             precautions=info.get('precautions', []),
                             selected_symptoms=formatted_selected,
                             algorithm=selected_algorithm.replace('_', ' ').title(),
                             probabilities=probabilities)
    
    except Exception as e:
        return render_template('result.html', error=str(e))

@app.route('/appointment')
def appointment():
    """Doctor appointment page"""
    disease = request.args.get('disease', '')
    return render_template('appointment.html', doctors=DOCTORS, disease=disease)

@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    """Book an appointment with a doctor"""
    try:
        doctor_id = request.form.get('doctor_id')
        patient_name = request.form.get('patient_name')
        patient_email = request.form.get('patient_email')
        patient_phone = request.form.get('patient_phone')
        appointment_date = request.form.get('appointment_date')
        appointment_time = request.form.get('appointment_time')
        disease = request.form.get('disease', '')
        
        # Find doctor
        doctor = next((d for d in DOCTORS if d['id'] == int(doctor_id)), None)
        
        if doctor:
            # In production, save to database here
            return render_template('booking_confirmation.html',
                                 doctor=doctor,
                                 patient_name=patient_name,
                                 appointment_date=appointment_date,
                                 appointment_time=appointment_time,
                                 disease=disease)
        else:
            return render_template('appointment.html', 
                                 doctors=DOCTORS, 
                                 error="Doctor not found.",
                                 disease=disease)
    
    except Exception as e:
        return render_template('appointment.html', 
                             doctors=DOCTORS, 
                             error=str(e),
                             disease='')

@app.route('/api/symptoms')
def api_symptoms():
    """API endpoint to get symptoms list"""
    symptoms = models.get('symptoms', [])
    formatted = [{'value': s, 'label': s.replace('_', ' ').title()} for s in symptoms]
    return jsonify(formatted)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    try:
        data = request.get_json()
        selected_symptoms = data.get('symptoms', [])
        algorithm = data.get('algorithm', 'random_forest')
        
        if not selected_symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        # Create input vector
        symptoms_list = models.get('symptoms', [])
        input_vector = np.zeros(len(symptoms_list))
        
        for symptom in selected_symptoms:
            if symptom in symptoms_list:
                idx = symptoms_list.index(symptom)
                input_vector[idx] = 1
        
        # Get the selected model
        model = models.get(algorithm)
        if model is None:
            return jsonify({'error': 'Model not found'}), 404
        
        # Make prediction
        prediction = model.predict([input_vector])[0]
        disease = models['label_encoder'].inverse_transform([prediction])[0]
        
        # Get disease info
        disease_info = models.get('disease_info', {})
        info = disease_info.get(disease, {})
        
        return jsonify({
            'disease': disease,
            'description': info.get('description', ''),
            'precautions': info.get('precautions', []),
            'algorithm': algorithm
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize models when app starts
with app.app_context():
    initialize_models()

if __name__ == '__main__':
    app.run(debug=True, port=5000)

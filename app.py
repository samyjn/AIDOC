from flask import Flask, request, render_template, jsonify  # Import jsonify
import numpy as np
import pandas as pd
import pickle
import sklearn

app = Flask(__name__)


sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")



svc = pickle.load(open('models/svc.pkl','rb'))


layman_to_feature = {
    'itching': 'itching',
    'itch' : 'itching',
    'rash': 'skin_rash',
    'rashes': 'skin_rash',
    'nodal skin eruptions': 'nodal_skin_eruptions',
    'continuous sneezing': 'continuous_sneezing',
    'shivering': 'shivering',
    'chills': 'chills',
    'joint pain': 'joint_pain',
    'joints': 'joint_pain',
    'stomach pain': 'stomach_pain',
    'acidity': 'acidity',
    'ulcers on tongue': 'ulcers_on_tongue',
    'tongue': 'ulcers_on_tongue',
    'muscle wasting': 'muscle_wasting',
    'vomiting': 'vomiting',
    'burning micturition': 'burning_micturition',
    'spotting urination': 'spotting_ urination',
    'fatigue': 'fatigue',
    'weight gain': 'weight_gain',
    'weight increase': 'weight_gain',
    'gain in weight': 'weight_gain',
    'anxiety': 'anxiety',
    'cold hands and feets': 'cold_hands_and_feets',
    'mood swings': 'mood_swings',
    'weight loss': 'weight_loss',
    'restlessness': 'restlessness',
    'lethargy': 'lethargy',
    'patches in throat': 'patches_in_throat',
    'irregular sugar level': 'irregular_sugar_level',
    'sugar level': 'irregular_sugar_level',
    'cough': 'cough',
    'high fever': 'high_fever',
    'sunken eyes': 'sunken_eyes',
    'breathlessness': 'breathlessness',
    'sweating': 'sweating',
    'dehydration': 'dehydration',
    'indigestion': 'indigestion',
    'headache': 'headache',
    'yellowish skin': 'yellowish_skin',
    'dark urine': 'dark_urine',
    'nausea': 'nausea',
    'loss of appetite': 'loss_of_appetite',
    'pain behind the eyes': 'pain_behind_the_eyes',
    'back pain': 'back_pain',
    'constipation': 'constipation',
    'abdominal pain': 'abdominal_pain',
    'diarrhoea': 'diarrhoea',
    'mild fever': 'mild_fever',
    'yellow urine': 'yellow_urine',
    'yellowing of eyes': 'yellowing_of_eyes',
    'acute liver failure': 'acute_liver_failure',
    'fluid overload': 'fluid_overload',
    'swelling of stomach': 'swelling_of_stomach',
    'swelled lymph nodes': 'swelled_lymph_nodes',
    'malaise': 'malaise',
    'blurred and distorted vision': 'blurred_and_distorted_vision',
    'phlegm': 'phlegm',
    'throat irritation': 'throat_irritation',
    'redness of eyes': 'redness_of_eyes',
    'sinus pressure': 'sinus_pressure',
    'runny nose': 'runny_nose',
    'congestion': 'congestion',
    'chest pain': 'chest_pain',
    'weakness in limbs': 'weakness_in_limbs',
    'fast heart rate': 'fast_heart_rate',
    'pain during bowel movements': 'pain_during_bowel_movements',
    'pain in anal region': 'pain_in_anal_region',
    'bloody stool': 'bloody_stool',
    'irritation in anus': 'irritation_in_anus',
    'neck pain': 'neck_pain',
    'dizziness': 'dizziness',
    'cramps': 'cramps',
    'bruising': 'bruising',
    'obesity': 'obesity',
    'swollen legs': 'swollen_legs',
    'swollen blood vessels': 'swollen_blood_vessels',
    'puffy face and eyes': 'puffy_face_and_eyes',
    'enlarged thyroid': 'enlarged_thyroid',
    'brittle nails': 'brittle_nails',
    'swollen extremeties': 'swollen_extremeties',
    'excessive hunger': 'excessive_hunger',
    'extra marital contacts': 'extra_marital_contacts',
    'drying and tingling lips': 'drying_and_tingling_lips',
    'slurred speech': 'slurred_speech',
    'knee pain': 'knee_pain',
    'hip joint pain': 'hip_joint_pain',
    'muscle weakness': 'muscle_weakness',
    'stiff neck': 'stiff_neck',
    'swelling joints': 'swelling_joints',
    'movement stiffness': 'movement_stiffness',
    'spinning movements': 'spinning_movements',
    'loss of balance': 'loss_of_balance',
    'unsteadiness': 'unsteadiness',
    'weakness of one body side': 'weakness_of_one_body_side',
    'loss of smell': 'loss_of_smell',
    'bladder discomfort': 'bladder_discomfort',
    'foul smell of urine': 'foul_smell_of urine',
    'continuous feel of urine': 'continuous_feel_of_urine',
    'passage of gases': 'passage_of_gases',
    'internal itching': 'internal_itching',
    'toxic look (typhos)': 'toxic_look_(typhos)',
    'depression': 'depression',
    'irritability': 'irritability',
    'muscle pain': 'muscle_pain',
    'altered sensorium': 'altered_sensorium',
    'red spots over body': 'red_spots_over_body',
    'belly pain': 'belly_pain',
    'abnormal menstruation': 'abnormal_menstruation',
    'dischromic  patches': 'dischromic _patches',
    'watering from eyes': 'watering_from_eyes',
    'increased appetite': 'increased_appetite',
    'polyuria': 'polyuria',
    'family history': 'family_history',
    'mucoid sputum': 'mucoid_sputum',
    'rusty sputum': 'rusty_sputum',
    'lack of concentration': 'lack_of_concentration',
    'visual disturbances': 'visual_disturbances',
    'receiving blood transfusion': 'receiving_blood_transfusion',
    'receiving unsterile injections': 'receiving_unsterile_injections',
    'coma': 'coma',
    'stomach bleeding': 'stomach_bleeding',
    'distention of abdomen': 'distention_of_abdomen',
    'history of alcohol consumption': 'history_of_alcohol_consumption',
    'fluid overload': 'fluid_overload.1',
    'blood in sputum': 'blood_in_sputum',
    'prominent veins on calf': 'prominent_veins_on_calf',
    'palpitations': 'palpitations',
    'painful walking': 'painful_walking',
    'pus filled pimples': 'pus_filled_pimples',
    'blackheads': 'blackheads',
    'scurring': 'scurring',
    'skin peeling': 'skin_peeling',
    'silver like dusting': 'silver_like_dusting',
    'small dents in nails': 'small_dents_in_nails',
    'inflammatory nails': 'inflammatory_nails',
    'blister': 'blister',
    'red sore around nose': 'red_sore_around_nose',
    'yellow crust ooze': 'yellow_crust_ooze'
}


def preprocess_symptoms(symptoms):
    preprocessed_symptoms = []
    for symptom in symptoms.split(','):
        for word in layman_to_feature:
            if word in symptom:
                preprocessed_symptoms.append(layman_to_feature[word])
    return preprocessed_symptoms


def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med,die,wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]





@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        # mysysms = request.form.get('mysysms')
        # print(mysysms)
        print(symptoms)
        if symptoms =="Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)
        else:
            user_symptoms = preprocess_symptoms(symptoms)
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
            
            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout)

    return render_template('index.html')


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")


if __name__ == '__main__':

    app.run(debug=True)
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import joblib

app = Flask(__name__)

# 저장된 모델과 인코더 불러오기
model = tf.keras.models.load_model('model/tensorflow_logistic_regression_model.h5')
mlb = joblib.load('model/mlb_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.getlist('champions')
    
    # 새로운 데이터에 인코딩 적용
    new_X = mlb.transform([data])
    
    # 모델을 사용하여 예측
    prediction = model.predict(new_X)
    probability = prediction[0][0]
    
    return jsonify({'probability': float(probability)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

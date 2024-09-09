import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer

# CSV 파일 경로 설정
file_path = '../datas/LeagueOfLegend-dataset.csv'

# CSV 파일 읽기 (한글 데이터셋, UTF-8 인코딩 지정)
data = pd.read_csv(file_path, encoding='utf-8')

# 승패 값의 분포 확인
print(data['승패'].value_counts())

# 챔피언 컬럼들을 리스트로 병합하여 새로운 컬럼 생성
data['챔피언 조합'] = data[['챔피언 1', '챔피언 2', '챔피언 3', '챔피언 4', '챔피언 5']].values.tolist()

# 멀티-핫 인코딩을 위한 MultiLabelBinarizer 사용
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(data['챔피언 조합'])

# 타겟 변수
y = data['승패']

# 타겟 변수의 분포 확인
print(y.value_counts())

# 데이터 분할 (학습용, 테스트용) - stratify 파라미터 사용
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 분할된 데이터의 타겟 변수 분포 확인
print(pd.Series(y_train).value_counts())
print(pd.Series(y_test).value_counts())

# 로지스틱 회귀 분류 모델
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# 분류 모델 성능 평가
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Logistic Regression Classifier")
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")

# 선형 회귀 모델
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_reg = lin_reg.predict(X_test)

# 회귀 모델 성능 평가
mse = mean_squared_error(y_test, y_pred_reg)
r2 = r2_score(y_test, y_pred_reg)

print("\nLinear Regression Model")
print(f"Mean Squared Error 결과: {mse}")
print(f"R2 스코어: {r2}")

# solar_predictor_with_uncertainty.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

class SolarPredictor:
    def __init__(self, price_per_kwh=100):
        self.price_per_kwh = price_per_kwh
        self.model = None

    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)
        print(f"데이터 로드 완료: {self.data.shape} 행/열")
    
    def preprocess(self):
        self.data.fillna(method='ffill', inplace=True)
        if '날짜' in self.data.columns:
            self.data['날짜'] = pd.to_datetime(self.data['날짜'])
            self.data['월'] = self.data['날짜'].dt.month
            self.data['일'] = self.data['날짜'].dt.day
            self.data['요일'] = self.data['날짜'].dt.weekday

    def train_model(self):
        X = self.data[['일조량', '기온', '습도', '풍속', '월', '일', '요일']]
        y = self.data['발전량']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"모델 학습 완료 | RMSE: {rmse:.2f}, R²: {r2:.2f}")

    def predict(self, input_df):
        if self.model is None:
            raise Exception("모델이 학습되지 않았습니다.")
        
        # 모든 트리의 예측값 추출
        all_tree_preds = np.array([tree.predict(input_df) for tree in self.model.estimators_])
        mean_pred = all_tree_preds.mean(axis=0)
        std_pred = all_tree_preds.std(axis=0)

        # 신뢰구간 계산 (95%)
        lower = mean_pred - 2*std_pred
        upper = mean_pred + 2*std_pred
        payment_pred = mean_pred * self.price_per_kwh

        return pd.DataFrame({
            '예측발전량(kWh)': mean_pred,
            '발전량_하한(kWh)': lower,
            '발전량_상한(kWh)': upper,
            '예측정산액(원)': payment_pred
        })

    def plot_predictions(self, predictions):
        plt.figure(figsize=(12,5))
        plt.plot(predictions['예측발전량(kWh)'], label='예측발전량')
        plt.fill_between(range(len(predictions)), 
                         predictions['발전량_하한(kWh)'], 
                         predictions['발전량_상한(kWh)'], 
                         color='orange', alpha=0.2, label='95% 신뢰구간')
        plt.xlabel('시간 index')
        plt.ylabel('발전량(kWh)')
        plt.title('태양광 발전량 예측과 신뢰구간')
        plt.legend()
        plt.show()

    def save_model(self, filename='solar_model.pkl'):
        joblib.dump(self.model, filename)
        print(f"모델 저장 완료: {filename}")

    def load_model(self, filename='solar_model.pkl'):
        self.model = joblib.load(filename)
        print(f"모델 로드 완료: {filename}")


# ==========================
# 사용 예시
# ==========================
if __name__ == "__main__":
    predictor = SolarPredictor(price_per_kwh=120)
    predictor.load_data('solar_data.csv')
    predictor.preprocess()
    predictor.train_model()

    test_input = predictor.data[['일조량', '기온', '습도', '풍속', '월', '일', '요일']].iloc[:10]
    predictions = predictor.predict(test_input)
    print(predictions)

    predictor.plot_predictions(predictions)
    predictor.save_model()

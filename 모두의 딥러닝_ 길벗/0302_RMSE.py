import numpy as np

ab = [3, 76] # 임의로 정한 기울기 a와 y절편 b

data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data] # x값
y = [i[1] for i in data] # y값

def predict(x): # 작성한 선형 회귀식에 넣어서 예측값 반환
    return ab[0]*x + ab[1]

def rmse(p, a):
    # 실제값과 예측값을 넣어서 평균 제곱근 오차 반환
    return np.sqrt(((p-a)**2).mean())

def rmse_val(predict_result, y):
    return rmse(np.array(predict_result), np.array(y))
    # 리스트를 통째로 계산하는 함수

predict_result = []
for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("공부한 시간=%.f, 실제 점수=%.f, 예측 점수=%.f" % (x[i], y[i], predict(x[i])))

# 최종 rmse
print("rmse 최종값: "+ str(rmse_val(predict_result, y)))
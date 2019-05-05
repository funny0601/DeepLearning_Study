import numpy as np

# 분자 구하는 함수 만듬
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d+=(x[i]-mx)*(y[i]-my)
    return d

x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

mx = np.mean(x) # x 값의 평균
my = np.mean(y) # y 값의 평균

# 최소제곱법을 이용한 기울기 구하기

# 분모 구하기
divisor = sum([(mx-i)**2 for i in x])

# 분자 구하기
dividend = top(x, mx, y, my)

a = dividend/divisor # 기울기 구함

# y절편 구하기
b = my-a*mx

print(a) # 2.3
print(b) # 79.0
# -*- coding: utf-8 -*-
# thư viện cung cấp để sử dụng các hàm sử lý mảng, số
import numpy as np
# thư viện cung cấp để đọc file csv, xử lý data, mảng
import pandas as pd
# thư viện vẽ đồ thị
import matplotlib.pyplot as plt
# đọc file vào data
data = pd.read_csv('data_linear.csv').values
# lấy số row đọc được (30)
N = data.shape[0]
# reshape là để đảo chiều của một mảng mà giữ nguyên giá trị của nó
# ví dụ ma trận 2 x 3 => 3 x 2 và giữ y giá trị của chúng theo đúng thứ tự đảo
# dòng dưới lấy hết data của cột đầu (0) và đưa vào mảng x dưới dạng mảng của các mảng [[row1X], [row2X],...]
# tức là từ mảng 1 dòng x n cột đảo thành mảng 1 cột x n dòng, mà mỗi dòng là 1 mảng
# (cách tính của thuật toán là như vậy, cứ 1 dòng thì xem là 1 mảng)
x = data[:, 0].reshape(-1, 1)
# lấy hết data của cột thứ hai (1) và đưa vào mảng y,... tương tự
y = data[:, 1].reshape(-1, 1)
# vẽ các chấm có tọa độ lấy từ 2 mảng x và y
plt.scatter(x, y)
# dán label vào cho cột x
plt.xlabel('mét vuông')
# dán label vào cột y
plt.ylabel('giá')
# numpy.hstack là để tạo 1 mảng, có phần tử là các mảng có phần tử lấy từ 2 mảng tạo nên nó
# ví dụ a = [1,2,3], b = [4,5,6] thì numpy.hstack(a,b) cho ra [[1,4],[2,5],[3,6]]
# thêm 1 vào trước mỗi phần tử x [[  1.      row1X], [  1.      row2X],...]
x = np.hstack((np.ones((N, 1)), x))
# tạo 1 mảng [[0., 1.]], sau đó đảo nó thành [[0.], [1.]] (cơ chế đảo như trên)
w = np.array([0., 1.]).reshape(-1, 1)
# số lần lặp để xét (càng lớn thì càng chính xác), ảnh hưởng trực tiếp đến thuật toán
numOfIteration = 200
# khỏi tạo mảng gồm 200 hàng và 1 cột toàn số 0 để sử dụng sau
cost = np.zeros((numOfIteration, 1))
# mức độ điều chỉnh sau mỗi lần lặp
# tức là ban đầu ta cho 1 đường ngẫu nhiên, qua các lần lặp, đường này sẽ được điều chỉnh
# sau cho càng ngày nó càng đi qua các điểm nhiều nhất có thể, và mức độ điều chỉnh đường này
# phụ thuộc vào con số này
learning_rate = 0.000001
#
#
# tham khảo tại: http://nttuan8.com/bai-1:-linear-regression-va-gradient-descent/
# bản chất và ứng dụng của thuật toán:
# cho 1 tập hợp các điểm ngẫu nhiên (data)
# sau đó tìm ra 1 đường thẳng gần như đi qua tất cả các điểm đó
# mục đích là từ đường thẳng đó, khi ta cho giá trị 1 cột, ta sẽ tính được giá trị cột còn lại
# để tìm ra đường thẳng đó, ban đầu ta cho 1 đường thẳng mặt định:
# y = w1*x + w0 với w1 và w0 là 1 và 0 (mảng w tạo ở trên) => y^ = x
# sau đó ta dịch đường thẳng 1 đoạn và còn làm nghiêng đường thẳng 1 khoảng bao nhiêu đó
# ta làm việc đó bằng cách cập nhật w0 và w1 liên tục (ở dưới có thấy w[0] và w[1])
#
# lặp với i = 0, i < numOfIteration, i++ (1 là step)
for i in range(1, numOfIteration):
    # x.w = y^
    # cần biết (y^ - y)
    r = np.dot(x, w) - y
    # theo công thức J = 1/2*sum((y^ - y)**2)
    cost[i] = 0.5*np.sum(r*r)
    # y = w1*x + w0
    # w0 khoảng cách (độ dịch) của đường thẳng so với cột hoành x
    # w1 là hệ số góc (độ nghiêng) của đường thẳng so với cột hoành x
    # ta cần điều chỉnh 2 số đó để thu được đường mới
    # w0 mới = w0 - mức độ điều chỉnh * sum(y^ - y)
    w[0] = w[0] - learning_rate*np.sum(r)
    # correct the shape dimension
    # w1 mới = w1 - mức độ điều chỉnh * sum(x * (y^ - y))
    w[1] = w[1]-learning_rate*np.sum(np.multiply(r, x[:, 1].reshape(-1, 1)))
    # print(cost[i])
predict = np.dot(x, w)
plt.plot((x[0][1], x[N-1][1]), (predict[0], predict[N-1]), 'r')
plt.show()
print('W0 = %s\t W1 = %s' % (w[0], w[1]))
# cho ví dụ:
x1 = 91
y1 = w[0] + w[1] * x1
print('Giá nhà cho 91m^2 là : ', y1)
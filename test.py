import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

import pycountry as pct

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Data/DT1.csv') #Để file dataset cùng 1 folder với file code

#------------BẮT ĐẦU QUÁ TRÌNH TIỀN XỬ LÝ DỮ LIỆU CHO PHÂN TÍCH MÔ TẢ------------

# Tóm lược dữ liệu (Đo mức độ tập trung & mức độ phân tán)
description = data.describe()
mode = data.select_dtypes(include=['float64','int64']).mode().iloc[0]
mode.name = 'mode'
median = data.select_dtypes(include=['float64','int64']).median()
median.name = 'median'
description = description._append(mode)
description = description._append(median)
print(description)

# Kiểm tra tỷ lệ lỗi thiếu data
data_na = (data.isnull().sum() / len(data)) * 100
missing_data = pd.DataFrame({'Ty le thieu data': data_na})
print(missing_data)

# Kiểm tra data bị trùng
duplicated_rows_data = data.duplicated().sum()
print(f"\nSO LUONG DATA BI TRUNG LAP: {duplicated_rows_data}")
data = data.drop_duplicates()

# Quét qua các cột và đếm số lượng data riêng biệt
print("\nSO LUONG CAC DATA RIENG BIET:")
for column in data.columns:
    num_distinct_values = len(data[column].unique())
    print(f"{column}:{num_distinct_values} distinct values")

# Xem qua dataset
print(f"\n5 DONG DAU DATA SET:\n {data.head(5)}")

# Các cột lựa chọn để phân tích mô tả
data_select = data[['Model Year' , 'Make', 'Model', 'Electric Vehicle Type', 'Clean Alternative Fuel Vehicle (CAFV) Eligibility', 'Electric Range', 'Electric Utility', 'Expected Price ($1k)']]
print(data_select)

def return_make_title(make_title):
    if make_title == 'TESLA':
        return "Tesla"
    elif make_title == 'NISSAN':
        return "Nissan"
    elif make_title == 'CHEVORLET':
        return "Chevorlet"
    elif make_title == 'BMW':
        return "BMW"
    elif make_title == 'TOYOTA':
        return "Toyota"
    elif make_title == 'FORD':
        return "Ford"
    elif make_title == 'AUDI':
        return "Audi"
    elif make_title == 'KIA':
        return "Kia"
    elif make_title == 'CHRYSLER':
        return "Chrysler"
    elif make_title == 'HYUNDAI':
        return "Hyundai"
    elif make_title == 'VOLKSWAGEN':
        return "Volkswagen"
    else:
        return "Other"

data['Model_Make'] = data['Make'].apply(return_make_title)
print(data)

non_number = data[['Model Year' , 'Model_Make', 'Electric Vehicle Type', 'Clean Alternative Fuel Vehicle (CAFV) Eligibility']]
#Biểu đồ loại 1: Các biểu đồ quạt của biến thể loại
for x in non_number.columns:
    x1 = non_number[x].value_counts().sort_values(ascending=True)
    fig1 = px.pie(values=x1.values,
                  names=x1.index,
                  color=x1.index,
                  title="BIỂU ĐỒ HÌNH TRÒN CỦA CÁC BIẾN THỂ LOẠI")
    fig1.update_traces(textinfo='label+percent+value',
                       textposition='outside')
    fig1.show()
numberic = data[['Electric Range', 'Expected Price ($1k)']]
# Biều đồ loại 2: Biểu đồ displot dữ liệu giá tính theo USD (PTMT: Đơn biến - dữ liệu số)
fig2 = ff.create_distplot(hist_data=[data['Expected Price ($1k)']],
                          group_labels=['Expected Price ($1k)'],
                          bin_size=7,
                          curve_type='kde')
fig2.update_layout(xaxis_title='Giá (USD)',
                   yaxis_title='Tần suất',
                   title='BIỂU ĐỒ DISPLOT CỦA Giá ($1k)')
fig2.show()
# Biều đồ loại 2: Biểu đồ displot dữ liệu về quãng đường đi(PTMT: Đơn biến - dữ liệu số)
fig3 = ff.create_distplot(hist_data=[data['Electric Range']],
                          group_labels=['Electric Range'],
                          bin_size=30,
                          curve_type='kde')
fig3.update_layout(xaxis_title='Range',
                   yaxis_title='Tần suất ',
                   title='BIỂU ĐỒ DISPLOT CỦA Electric Range')
fig3.show()

#Biểu đồ loại 3: Biểu đồ boxplot phân bố giá theo từng biến thể loại
for x in non_number.columns:
    fig4 = px.box(data_frame=data,
                  x=x,
                  y='Expected Price ($1k)',
                  color=x,
                  title='BIỂU ĐỒ BOXPLOT PHÂN BỔ GIÁ (1k$) THEO CÁC BIẾN THỂ LOẠI')
    fig4.update_layout(xaxis_title='Biến thể loại',
                       yaxis_title='Lương (USD)')
    fig4.show()

#Biểu đồ loại 4: Biểu đồ heatmap phân bổ trung bình giá (1k$) theo từng năm đối với loại pin (PTMT: Đa biến (3) - dữ liệu hỗn hợp)
pivot_table = data.pivot_table(values='Expected Price ($1k)',
                               index='Electric Vehicle Type',
                               columns='Model Year',
                               aggfunc='mean')
fig5 = px.imshow(pivot_table,
                 labels=dict(x='Năm', y='Loại pin'),
                 x=pivot_table.columns,
                 y=pivot_table.index,
                 text_auto='.2f',
                 color_continuous_scale='Viridis',
                 title='BIỂU ĐỒ HEATMAP PHÂN BỔ TRUNG BÌNH GIÁ (1k$) THEO TỪNG NĂM VỚI MỖI LOẠI PIN')
fig5.show()

#Biểu đồ loại 4: Biểu đồ heatmap phân bổ trung bình giá (1k$) theo từng năm đối với hãng (PTMT: Đa biến (3) - dữ liệu hỗn hợp)
pivot_table = data.pivot_table(values='Expected Price ($1k)',
                               index='Model_Make',
                               columns='Model Year',
                               aggfunc='mean')
fig6 = px.imshow(pivot_table,
                 labels=dict(x='Năm', y='Hãng'),
                 x=pivot_table.columns,
                 y=pivot_table.index,
                 text_auto='.2f',
                 color_continuous_scale='Viridis',
                 title='BIỂU ĐỒ HEATMAP PHÂN BỔ TRUNG BÌNH GIÁ (1K$) THEO TỪNG NĂM VỚI MỖI HÃNG')
fig6.show()

# Biểu đồ lọai 5: Biểu đồ Scatter thể hiện giá ảnh hưởng bởi năm và quãng đường đi được tối đa thực tế(PTMT: Đa biến (3) - dữ liệu hỗn hợp)
fig7 = px.scatter(data_frame=data,
    x='Model Year',
    y='Electric Range',
    color='Expected Price ($1k)',
    size='Expected Price ($1k)', opacity=0.5,
    labels={'Model Year': 'Năm', 'Electric Range': 'Quãng đường đi',
    'Expected Price ($1k)': 'Giá'},
    hover_data=['Expected Price ($1k)'],
    title='Biểu đồ thể hiện giá ảnh hưởng bởi năm và quãng đường đi được tối đa thực tế')
fig7.show()

# Biểu đồ lọai 5: Biểu đồ Scatter thể hiện giá ảnh hưởng bởi hãng và quãng đường đi được tối đa thực tế(PTMT: Đa biến (3) - dữ liệu hỗn hợp)
fig8 = px.scatter(data_frame=data,
    x='Model_Make',
    y='Electric Range',
    color='Expected Price ($1k)',
    size='Expected Price ($1k)', opacity=0.5,
    labels={'Model_Make': 'Hãng', 'Electric Range': 'Quãng đường đi',
    'Expected Price ($1k)': 'Giá'},
    hover_data=['Expected Price ($1k)'],
    title='Biểu đồ thể hiện giá ảnh hưởng bởi năm và quãng đường đi được tối đa thực tế')
fig8.show()
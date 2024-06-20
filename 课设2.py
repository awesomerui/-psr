import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 加载训练数据，手动指定列名
def load_train_data(file_path):
    column_names = ['uid', 'time', 'forward_count', 'comment_count', 'like_count', 'content']
    return pd.read_csv(file_path, sep='\t', header=None, names=column_names)

# 加载预测数据，手动指定列名
def load_predict_data(file_path):
    column_names = ['uid', 'mid', 'time', 'content']
    return pd.read_csv(file_path, sep='\t', header=None, names=column_names)

# 特征工程，这里简化处理，仅考虑内容长度作为特征
def extract_features(df):
    df['content'] = df['content'].astype(str)  # 确保内容列是字符串类型
    df['content_length'] = df['content'].apply(lambda x: len(x) if x.strip() != '' else 0)  # 避免空字符串的长度计算问题
    return df[['content_length']]

# 训练模型
def train_and_validate(X_train, y_train):
    models = []
    for i in range(y_train.shape[1]):
        model = LinearRegression()
        model.fit(X_train, y_train.iloc[:, i])
        models.append(model)
    return models

# 预测并处理结果为整数
def predict_and_round(models, X_test):
    predictions = np.array([model.predict(X_test) for model in models]).T
    rounded_predictions = np.rint(predictions).astype(int)
    return rounded_predictions

# 生成提交格式的结果文件
def generate_submission_file(predictions, predict_df, output_file):
    # 合并预测结果到原DataFrame
    predict_df[['forward_count', 'comment_count', 'like_count']] = pd.DataFrame(predictions, columns=['forward_count', 'comment_count', 'like_count'])
    # 保存结果
    predict_df[['uid', 'mid', 'forward_count']].to_csv(output_file, sep='\t', index=False, header=False)
    # 生成额外的文件，包含所有预测值，用于验证
    predict_df[['uid', 'mid', 'forward_count', 'comment_count', 'like_count']].to_csv('weibo_full_result_data.txt', sep='\t', index=False, header=False)

# 主程序
def main():
    # 加载数据
    train_df = load_train_data('D:/Users/86138/Desktop/python/智能/.venv/weibo_train_data.txt')
    predict_df = load_predict_data('D:/Users/86138/Desktop/python/智能/.venv/weibo_predict_data.txt')

    # 特征工程
    X_train = extract_features(train_df)
    y_train = train_df[['forward_count', 'comment_count', 'like_count']]

    # 模型训练与验证
    models = train_and_validate(X_train, y_train)

    # 预测并处理预测数据集
    X_predict = extract_features(predict_df)
    predictions = predict_and_round(models, X_predict)

    # 结果整合与输出
    generate_submission_file(predictions, predict_df, 'weibo_result_data.txt')
    print("预测结果已生成至 'weibo_result_data.txt' 文件中。")

if __name__ == "__main__":
    main()

import re 
import pandas as pd
from sklearn.model_selection import train_test_split
#数据处理

def json_processed(json_path):
    #读取json数据、转换为pandas dataframe
        
    df=pd.read_json(json_path,encoding="utf-8",orient='records')

    #去除重复评分、取最新
    idx=df.groupby(['UserID','MealID'])["ReviewTime"].idxmax()
    df_norepeat = df.loc[idx].reset_index(drop=True)
    # 使用 sklearn 的 train_test_split 函数进行划分
    trainData, testData = train_test_split(df_norepeat, test_size=0.2, random_state=42)
    # 使用 factorize 方法将非数字标识转换为唯一的数字标识，便于计算相似度，映射保存在
    trainData['UserID_encoded'], user_categories = pd.factorize(trainData['UserID'])
    trainData['MealID_encoded'], meal_categories = pd.factorize(trainData['MealID'])
    return trainData,testData,user_categories,meal_categories

def sql_processed(sql_path):
        
    # 读取 .sql 文件
    with open(sql_path, 'r', encoding='utf-8') as file:
        sql_script = file.read()

    # 从 SQL 脚本中提取数据部分
    insert_statements = re.findall(r'INSERT INTO `meal_list` VALUES\s*\((.*?)\);\s*', sql_script, re.DOTALL)

    # 提取数据并构建 DataFrame
    list = []
    for statement in insert_statements:
        values = [v.strip().replace("'", "") for v in statement.split('),(')]
        list.extend([v.split(',') for v in values])

    columns = ['mealno', 'mealID', 'meal_name']
    data = pd.DataFrame(list, columns=columns)
    return data






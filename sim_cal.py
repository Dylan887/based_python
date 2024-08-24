#计算相似度
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

#1.余弦相似度
def cos_sim(df):
    #制作评分矩阵
    rating_matrix = df.pivot(index='UserID_encoded', columns='MealID_encoded', values='Rating').fillna(0)
    cosine_sim = cosine_similarity(rating_matrix)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=rating_matrix.index, columns=rating_matrix.index)
    return cosine_sim_df
    
#2.基于物品的余弦相似度
def item_sim(df):
    rating_matrix = df.pivot(index='UserID_encoded', columns='MealID_encoded', values='Rating').fillna(0)
    item_sim = cosine_similarity(rating_matrix.T)
    item_sim_df = pd.DataFrame(item_sim, index=rating_matrix.columns, columns=rating_matrix.columns)
    return item_sim_df
#3.其他


        

import dataprocessed
import sim_cal
import rating_matrix
import para
import recommend
import evaluta_recommend



"""
使用方法
1.
data_precocessed 数据处理
sim_cal 相似度计算
categories 用户和菜品ID的映射
rating_matrix trainData稀疏矩阵
evaluta_recommend 评估模型
als_re 的userid需要传入列表
"""

if __name__ == '__main__':
    
    #1.数据处理
    trainData,testData,user_categories,meal_categories=dataprocessed.json_processed("MealRatings_201705_201706.json")
    data_sql=dataprocessed.sql_processed("meal_list.sql")

    2.#相似度计算
    cosine_sim_df=sim_cal.cos_sim(trainData)
    item_sim_df=sim_cal.item_sim(trainData)
    3.#训练集稀疏矩阵
    rating_matrix=rating_matrix.rmt(trainData)

    #/////手动调参过程，如果需要，实际模型参数已经调节好
    # k 维度
    # iter 迭代次数
    #para.para(data_precocessed,k,iter)
    para.para(trainData,5,3)
    #4.推荐
    userid=1000
    top_n=10
    real_userid=user_categories[userid]
    user_ratings = testData[testData['UserID'] == real_userid]
    # 按评分值降序排序，并获取前十个物品
    ture_meal = user_ratings.sort_values(by='Rating', ascending=False).head(top_n)
    
    #基于用户
    print("---------------------------基于用户--------------------------------")
    meal_rec=recommend.user_cf(userid,rating_matrix,cosine_sim_df,data_sql,user_categories,meal_categories,top_n)
    evaluta_recommend.calculate_metrics(meal_rec,ture_meal)
    #基于物品
    print("---------------------------基于物品--------------------------------")
    meal_rec=recommend.item_cf(userid,rating_matrix,item_sim_df,data_sql,user_categories,meal_categories,top_n)
    evaluta_recommend.calculate_metrics(meal_rec,ture_meal)
    #基于ALS
    print("---------------------------基于ALS--------------------------------")
    userid_list=[userid]
    meal_rec=recommend.als_re(userid_list,meal_categories,data_sql,top_n)
    evaluta_recommend.calculate_metrics(meal_rec,ture_meal)

   


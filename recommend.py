#推荐系统
import pandas as pd
#1.基于用户的协同过滤
def user_cf(user_id,rating_matrix,cosine_sim_df,data_sql,user_categories,meal_categories,top_n):
    # 找到与用户最相似的用户
    similar_users = cosine_sim_df[user_id].sort_values(ascending=False).index[1:]
    # 获取这些用户对物品的评分
    similar_users_ratings = rating_matrix.loc[similar_users]
    # 计算加权评分
    recommendation_scores = similar_users_ratings.apply(lambda x: x * cosine_sim_df[user_id][similar_users], axis=0).sum(axis=0)
    # 排除用户已经评分过的物品
    recommendation_scores = recommendation_scores[rating_matrix.loc[user_id] == 0]
    # 选出评分最高的物品进行推荐
    meal_id = recommendation_scores.nlargest(top_n).index.tolist()
    meal_rec=data_sql[data_sql['mealID'].isin(meal_categories[meal_id])][['mealID','meal_name']].reset_index(drop=True)
    print(f"Recommended meal for User {user_categories[user_id]} is Meal:\n {meal_rec}")
    return meal_rec
    #print(f"Recommended meal for User {user_id} is Meal {recommended_meal}")

#2.基于物品的协同过滤
def item_cf(user_id,rating_matrix,item_sim_df,data_sql,user_categories,meal_categories,top_n):
    # 找到用户喜欢的物品及其评分
    user_ratings = rating_matrix.loc[user_id]
    # 计算推荐物品的加权评分
    weighted_scores = item_sim_df.dot(user_ratings)
    # 排除用户已经评分过的物品
    recommended_items = weighted_scores.drop(user_ratings[user_ratings > 0].index)
    # 选出评分最高的物品进行推荐
    meal_id = recommended_items.nlargest(top_n).index.tolist()
    meal_rec=data_sql[data_sql['mealID'].isin(meal_categories[meal_id])][['mealID','meal_name']].reset_index(drop=True)
    print(f"Recommended meal for User {user_categories[user_id]} is Meal:\n {meal_rec}")
    return meal_rec
    #print(f"Recommended item for User {user_id} is Item {recommended_item}")

#3.基于ALS的推荐
def als_re(user_list,meal_categories,data_sql,n_items):
    # 预测用户评分
    # 需要预测的用户ID user
    # 每个用户需要推荐的物品数量 n_items
    import pickle
    # 加载模型
    with open('als_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # 使用模型进行预测
    recommendations = model.predict(user_list, n_items)
    # 输出推荐结果
    meal_rec = pd.DataFrame(columns=['Item ID', 'Meal ID', 'Meal Name'])
    i=0
    for user_id, recs in zip(user_list, recommendations):
        for item_id, score in recs:
            # 获取对应的 meal_name
            meal_name = meal_categories[item_id]  # 假设 meal_categories 是一个字典或类似的映射关系
            # 查询数据并且输出
            res = data_sql[data_sql['mealID']==meal_categories[item_id]][['mealID', 'meal_name']].values.tolist()[0]
            #print(f"Item ID: {item_id}, Meal ID: {res[0]}, Meal Name: {res[1]}, Score: {score}")
            # 添加到 DataFrame
            new_row = {'ItemID': item_id, 'mealID': res[0], 'Meal_Name': res[1]}
            meal_rec = pd.concat([meal_rec, pd.DataFrame([new_row])], ignore_index=True)
            #print(f"{i}  {res[0]}  {res[1]}")
            i=i+1
    print(meal_rec)
    return meal_rec[['mealID','Meal_Name']]
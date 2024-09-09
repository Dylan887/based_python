from utils.als import ALS as als
import os
import pickle

def para(data_precocessed,k,max_iter):
    #基于ALS
    X = data_precocessed[['UserID_encoded', 'MealID_encoded', 'Rating']].values.tolist()
    # 初始化ALS模型
    als_model = als()
    # 训练模型
    # k ALS模型的rank
    # max_iter 最大迭代次数
    als_model.fit(X, k, max_iter=max_iter)
       
    print("是否更新当前ALS模型！(y/n)")
    user_input=input() 
    if user_input.lower()=='y':
        path="als_model.pkl"
        if os.path.exists(path):
            #os.remove(path)
            
            model=als_model
            with open('als_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            # 重新加载模型到内存中
            with open(path, 'rb') as f:
                als_model = pickle.load(f)
            print("模型已重新加载到内存中!")

        else:
            print(f"{path}不存在！")
    else:
        print("over!")
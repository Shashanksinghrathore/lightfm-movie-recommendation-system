import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k,auc_score

data = fetch_movielens(min_rating=4.0)

print(repr(data['train']))
print(repr(data['test']))

# model with warp

model_warp = LightFM(loss='warp')
model_warp.fit(data['train'], epochs=30, num_threads=2)

#model with bpr 

model_bpr = LightFM(loss='bpr')
model_bpr.fit(data['train'], epochs=30, num_threads=2)


def recommender(model, data, user_ids):

    n_users, n_items = data['train'].shape

    for user_id in user_ids :
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        scores = model.predict(user_id, np.arange(n_items))

        top_items = data['item_labels'][np.argsort(-scores)]

        #printing
        print("User %s" % user_id)
        print("      known_positives:")

        for x in known_positives[:3]:
            print("               %s" % x)

        print("      Recommended:")

        for x in top_items[:3]:
            print("               %s" % x)


recommender(model_warp, data, [3, 25, 450])

test_precision = precision_at_k(model_warp, data['test'], k=5).mean()
train_precision = precision_at_k(model_warp, data['train'], k=5).mean()
print('train_precision: %2f.' % train_precision)
print('test_precision: %2f.' % test_precision)

auc_train = auc_score(model_warp, data['train'] ).mean()
auc_test = auc_score(model_warp, data['test'] ).mean()
print('train auc %2f.' % auc_train)
print('test auc %2f.' % auc_test)


recommender(model_bpr, data, [3, 25, 450])

test_precision = precision_at_k(model_bpr, data['test'], k=5).mean()
train_precision = precision_at_k(model_bpr, data['train'], k=5).mean()
print('train_precision: %2f.' % train_precision)
print('test_precision: %2f.' % test_precision)

auc_train = auc_score(model_bpr, data['train'] ).mean()
auc_test = auc_score(model_bpr, data['test'] ).mean()
print('train auc %2f.' % auc_train)
print('test auc %2f.' % auc_test)




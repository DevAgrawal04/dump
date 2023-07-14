import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from hyperopt import fmin, hp, tpe, Trials

# Generate sample data for regression
np.random.seed(0)
X, y = make_regression(n_samples=100, n_features=10, random_state=0)

# Define the objective function for hyperopt
def objective(params):
    alpha = params['alpha']
    l1_ratio = params['l1_ratio']
    fit_intercept = params['fit_intercept']
    max_iter = params['max_iter']

    # Define the model
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, max_iter=max_iter, random_state=0)

    # Perform k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True)
    scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')

    # Compute the average RMSE score
    mean_score = scores.mean()

    return mean_score

# Define the search space for hyperparameters
space = {
    'alpha': hp.loguniform('alpha', -6, 2),
    'l1_ratio': hp.uniform('l1_ratio', 0, 1),
    'fit_intercept': hp.choice('fit_intercept', [True, False]),
    'max_iter': hp.choice('max_iter', range(100, 10000))
}

# Create a trials object to store the optimization results
trials = Trials()

# Run the hyperparameter optimization
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

# Train the final model with the best hyperparameters
final_model = ElasticNet(
    alpha=best['alpha'],
    l1_ratio=best['l1_ratio'],
    fit_intercept=best['fit_intercept'],
    max_iter=best['max_iter'],
    random_state=0
)
final_model.fit(X, y)

# Evaluate the RMSE on the training data
y_pred = final_model.predict(X)
rmse = mean_squared_error(y, y_pred, squared=False)

# Print the RMSE of the final model with the best hyperparameters
print("RMSE: {:.4f}".format(rmse))


"""

Sweet Surprise at Trophy!


User
Draft a mail around the notice that there are sweets at the trophy stand. It's a token of gratitude and appreciation towards all the employees from us interns. If possible, add a humorous joke about the availability of low sugar sweets as well. Have a paragraph thanking everyone for being helpful, nurturing and caring towards us interns

ChatGPT
Subject: Sweet Surprise at the Trophy Stand! 🍬🏆

Dear [Company Name] Team,

We hope this email finds you in high spirits and with an insatiable sweet tooth! As the end of our internship program draws near, we, the interns, wanted to take a moment to express our heartfelt gratitude and appreciation for the incredible support and guidance we've received from all of you. To mark this occasion and show our thanks, we have a delightful surprise waiting for you at the trophy stand!

But before we get to that, we must address the elephant (or should we say, the sugar cube?) in the room. We are well aware of the ever-growing concerns about maintaining a healthy lifestyle, and we've made sure to include a variety of low sugar sweets in our selection. After all, we wouldn't want anyone's dentist giving us a "not-so-sweet" lecture, right? So, fear not, health-conscious candy enthusiasts, we've got your cravings covered too!

Now, let's get back to the main event—sweets galore! We've carefully handpicked a wide assortment of delectable treats that we hope will tickle your taste buds and bring a smile to your face. From gummy bears to chocolate bars, from licorice twists to fruity mints, there's something for every kind of sweet tooth. And don't worry, we made sure to stock up to accommodate your enthusiastic indulgence.

In all seriousness, dear colleagues, it has been an incredible journey for us as interns. Your unwavering support, nurturing guidance, and genuine care have made all the difference in our professional growth. From answering our numerous questions to sharing your valuable insights, you've been the pillars that kept us steady on this exciting learning curve.

We extend our gratitude to each and every one of you for making us feel like a part of this fantastic team and creating an inclusive environment where we could learn, contribute, and grow. Your willingness to share your expertise and experiences has been instrumental in shaping our internship experience, and for that, we are truly thankful.

So, head on over to the trophy stand, where a sweet surprise awaits you! Feel free to take a moment to indulge, savor the treats, and let your taste buds do a little happy dance. You deserve it!

Once again, thank you from the bottom of our hearts for your kindness, patience, and the warmth you've shown us throughout this internship journey. We look forward to maintaining the connections we've forged and the invaluable lessons we've learned from you.

With sugar-coated appreciation and smiles,

[Your Names]
[Internship Program Year]
"""
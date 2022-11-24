from bayes_opt import BayesianOptimization
from canny import canny
from metrics import dice_coef
from benchmark import binarize

def overfitting_canny_bayesian(img, edges):
    edges = binarize(edges, 250) 

    def black_box(**params):
        pred = canny(img, **params)
        pred = binarize(pred, 250)
        return dice_coef(pred, edges)

    pbounds = {
        "thr1":(0, 256),
        "thr2":(0, 256)
    }

    optimizer = BayesianOptimization(
        f=black_box,
        pbounds=pbounds,
        random_state=1,
        verbose=0
    )

    optimizer.maximize(
        init_points=4,
        n_iter=15,
    )

    params = optimizer.max["params"]
    params["thr1"] = int(params["thr1"])
    params["thr2"] = int(params["thr2"])

    return canny(img, **params)
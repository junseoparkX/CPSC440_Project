import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

import matplotlib.pyplot as plt

from collections import Counter

# ----- bind globals exactly like the original style -----
New_FS = X_train.copy()
y_trn  = y_train.reset_index(drop=True)
kfold  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model  = RandomForestClassifier(
    n_estimators=300,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced_subsample"
)

def initilization_of_population(size, n_feat):
    population = []
    for _ in range(size):
        chromosome = np.ones(n_feat, dtype=bool)
        chromosome[:int(0.3 * n_feat)] = False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population

def fitness_score(population):
    scores, newtp, newfp, newtn, newfn = [], [], [], [], []
    for chromosome in population:
        tp, fp, tn, fn, acc = [], [], [], [], []
        for train_idx, test_idx in kfold.split(New_FS, y_trn):
            model.fit(New_FS.iloc[train_idx].iloc[:, chromosome], y_trn[train_idx])
            true_labels = np.asarray(y_trn[test_idx])
            preds = model.predict(New_FS.iloc[test_idx].iloc[:, chromosome])
            ntp, nfn, ntn, nfp = confusion_matrix(true_labels, preds).ravel()
            tp.append(ntp); fp.append(nfp); tn.append(ntn); fn.append(nfn)
            acc.append(accuracy_score(true_labels, preds))
        scores.append(np.mean(acc))
        newtp.append(np.sum(tp)); newfp.append(np.sum(fp))
        newtn.append(np.sum(tn)); newfn.append(np.sum(fn))

    scores, population = np.array(scores), np.array(population)
    weights = [s / scores.sum() for s in scores]
    newtp, newfp, newtn, newfn = (
        np.array(newtp), np.array(newfp), np.array(newtn), np.array(newfn)
    )
    inds = np.argsort(scores)
    return (
        list(scores[inds][::-1]),
        list(population[inds][::-1]),
        list(np.array(weights)[inds][::-1]),
        list(newtp[inds][::-1]),
        list(newfp[inds][::-1]),
        list(newtn[inds][::-1]),
        list(newfn[inds][::-1]),
    )

def selection(pop_after_fit, weights, k):
    selected_pop = random.choices(pop_after_fit, weights=weights, k=k)
    return list(selected_pop)

def crossover(p1, p2, crossover_rate):
    c1, c2 = p1.copy(), p2.copy()
    if random.random() < crossover_rate:
        pt = random.randint(1, len(p1) - 2)
        c1 = np.concatenate((p1[:pt], p2[pt:]))
        c2 = np.concatenate((p2[:pt], p1[pt:]))
    return [c1, c2]

def mutation(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = not chromosome[i]

def generations(size, n_feat, crossover_rate, mutation_rate, n_gen):
    best_chromo = []
    best_score = []
    population_nextgen = initilization_of_population(size, n_feat)
    for i in range(n_gen):
        scores, pop_after_fit, weights, tp, fp, tn, fn = fitness_score(population_nextgen)
        score = scores[0]
        print("gen", i, score)
        k = size - 2
        pop_after_sel = selection(pop_after_fit, weights, k)
        children = []
        for j in range(0, len(pop_after_sel), 2):
            p1 = pop_after_sel[j]
            p2 = pop_after_sel[j + 1]
            for c in crossover(p1, p2, crossover_rate):
                mutation(c, mutation_rate)
                children.append(c)
        population_nextgen = []
        for c in pop_after_fit[:2]:
            population_nextgen.append(c)
        for p in children:
            population_nextgen.append(p)
        best_chromo.append(pop_after_fit[0])
        best_score.append(score)
    return best_chromo, best_score

# run GA (this is all the original did)
best_chromo, best_score = generations(
    size=50,
    n_feat=New_FS.shape[1],
    crossover_rate=0.8,
    mutation_rate=0.05,
    n_gen=30
)

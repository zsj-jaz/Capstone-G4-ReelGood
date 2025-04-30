import pandas as pd
import numpy as np
import pickle
from surprise import Dataset, Reader, KNNBaseline, KNNBasic
from pathlib import Path
import time


def train_knn_model(model_type, train_ratings, sim_options):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(train_ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    if model_type.lower() == 'basic':
        model = KNNBasic(sim_options=sim_options)
    elif model_type.lower()  == 'baseline':
        model = KNNBaseline(sim_options=sim_options)
    else: 
        raise ValueError("model_type must be either 'basic' or 'baseline'")
    model.fit(trainset)
    return model, trainset


def evaluate_model(model, trainset, eval_df, k_list=[10, 30], n_negatives=99):
    metrics = {f"recall@{k}": 0 for k in k_list}
    metrics.update({f"ndcg@{k}": 0 for k in k_list})
    total_users = 0
    rmse_true, rmse_pred = [], []
    percentile_ranks = []

    all_items = set(trainset.all_items())
    iid_map = {i: trainset.to_raw_iid(i) for i in all_items}

    for uid, true_iid, true_rating in eval_df[['userId', 'movieId', 'rating']].itertuples(index=False):
        if not trainset.knows_user(uid) or not trainset.knows_item(true_iid):
            continue

        uid_inner = trainset.to_inner_uid(uid)
        seen_items = set(j for (j, _) in trainset.ur[uid_inner])
        unseen_items = all_items - seen_items

        if true_iid not in trainset._raw2inner_id_items:
            continue

        neg_iids = np.random.choice(list(unseen_items), size=n_negatives, replace=False)
        candidates = [true_iid] + [iid_map[i] for i in neg_iids if iid_map[i] != true_iid]
        predictions = [model.predict(uid, iid) for iid in candidates]
        ranked = sorted(predictions, key=lambda x: x.est, reverse=True)
        ranked_iids = [int(p.iid) for p in ranked]

        # ---- calculate recall@k and ndcg@k
        for k in k_list:
            top_k = ranked_iids[:k]
            if true_iid in top_k:
                rank = top_k.index(true_iid) + 1
                metrics[f"recall@{k}"] += 1
                metrics[f"ndcg@{k}"] += 1 / np.log2(rank + 1)

        # ---- calculate percentile rank
        if true_iid in ranked_iids:
            true_rank = ranked_iids.index(true_iid)
            percentile = true_rank / len(ranked_iids)  # no +1 because true_rank is 0-indexed
            percentile_ranks.append(percentile)

        # ---- rmse for rating prediction
        rmse_true.append(true_rating)
        rmse_pred.append(model.predict(uid, true_iid).est)

        total_users += 1

    # Normalize all collected metrics
    for k in k_list:
        metrics[f"recall@{k}"] /= total_users
        metrics[f"ndcg@{k}"] /= total_users
    metrics["rmse"] = np.sqrt(np.mean((np.array(rmse_true) - np.array(rmse_pred))**2))
    metrics["percentile_rank"] = np.mean(percentile_ranks) if percentile_ranks else float('nan')

    # Print all metrics
    for key in metrics:
        print(f"{key}: {metrics[key]:.4f}")

    return metrics


def tune_knn(train, val, k_list=[10, 20, 50], sims=['cosine', 'pearson', 'msd'], results_path=Path("results") / "knn"):
    log = []
    best_score = float('inf')
    best_model = None
    best_trainset = None
    best_params = None

    results_path.mkdir(parents=True, exist_ok=True)

    for sim in sims:
        for k in k_list:
            print(f"🔍 BASELINE | sim: {sim}, k: {k}")
            sim_opt = {'name': sim, 'user_based': False, 'k': k}

            start = time.time()
            model, trainset = train_knn_model("baseline", train, sim_opt)
            train_time = time.time() - start

            metrics = evaluate_model(model, trainset, val)
            print(f"Train time: {train_time:.2f} seconds")

            log.append({
                'model_type': 'baseline',
                'similarity': sim,
                'k': k,
                'train_time': train_time,
                **metrics
            })

            if metrics["percentile_rank"] < best_score:
                best_score = metrics["percentile_rank"]
                best_model = model
                best_trainset = trainset
                best_params = (sim, k)
    
    log_path = results_path / "tune_log_baseline.csv"

    if log_path.exists():
        old_df = pd.read_csv(log_path)
        df = pd.concat([old_df, pd.DataFrame(log)], ignore_index=True)
    else:
        df = pd.DataFrame(log)

    df.to_csv(log_path, index=False)
    print(f"\n📁 Saved tuning log to {results_path/'tune_log_baseline.csv'}")
    print(f"\n🏆 Best Model Config: sim={best_params[0]}, k={best_params[1]}, percecntile rank={best_score:.4f}")

    print("Saving best model and trainset...")
    with open(results_path / "best_model_baseline.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(results_path / "best_trainset_baseline.pkl", "wb") as f:
        pickle.dump(best_trainset, f)

    return best_model, best_trainset


def test_knn(model, trainset, test_df):
    print("\n📊 Final Test Set Evaluation")
    metrics = evaluate_model(model, trainset, test_df)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


def run_knn_pipeline(data_path="data/train_test_split"):
    train = pd.read_csv(f"{data_path}/train_ratings.csv")
    val = pd.read_csv(f"{data_path}/val_ratings.csv")
    test = pd.read_csv(f"{data_path}/test_ratings.csv")

    best_model, best_trainset = tune_knn(train, val, k_list=[35, 40, 45], sims=['cosine', 'pearson', 'msd'])
    test_knn(best_model, best_trainset, test)


if __name__ == "__main__":
    run_knn_pipeline()

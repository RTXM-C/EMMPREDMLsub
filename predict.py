import os

import joblib
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import model13

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def deep_predict(model, X_test):
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test.reshape(-1, 640)).reshape(-1, 72, 640)
    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
    )
    test_loader = DataLoader(test_dataset, batch_size=64)

    model.to(device)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(device)
            preds = torch.sigmoid(model(inputs))
            all_preds.append(preds.cpu())
    all_preds = torch.cat(all_preds).numpy()

    return all_preds


def xg_et_predict(model_list, X_test):
    y_test_pro = []
    for i in range(9):
        test_pred = (model_list[i].predict_proba(X_test)).tolist()
        y_test_pro.append(test_pred)
    preds = np.hstack([np.array(y_test_pro[i])[:, 1].reshape(-1, 1) for i in range(9)])

    return preds


def main_predict(feature):
    path_root = "./model"

    params = {
        'hidden_size': 256,
        'num_layers': 1,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
    }
    dl_model = model13.model13(params)
    dl_model.load_state_dict(
        torch.load(os.path.join(os.path.join(path_root, "model13"), "best_model13_test.pth"), map_location=device))
    dl_pro = deep_predict(dl_model, feature)

    reduced_features = np.mean(feature, axis=1)
    xgb_models = [joblib.load(os.path.join(os.path.join(path_root, "XGBoost"), f"test_{i}.pkl")) for i in
                  range(9)]
    et_models = [joblib.load(os.path.join(os.path.join(path_root, "ExtraTrees"), f"test_{i}.pkl")) for i in
                 range(9)]
    xgb_pro = xg_et_predict(xgb_models, reduced_features)
    et_pro = xg_et_predict(et_models, reduced_features)

    lgb_models = [joblib.load(os.path.join(os.path.join(path_root, "LightGBM"), f"test_{i}.pkl")) for i in
                  range(9)]
    y_pred = []
    for i in range(9):
        X_test = np.hstack([dl_pro[:, i].reshape(-1, 1), xgb_pro[:, i].reshape(-1, 1), et_pro[:, i].reshape(-1, 1)])
        pred = lgb_models[i].predict(X_test, num_iteration=lgb_models[i].best_iteration)
        y_pred.append(pred)

    preds = np.hstack([np.array(y_pred[i]).reshape(-1, 1) for i in range(9)])
    return preds

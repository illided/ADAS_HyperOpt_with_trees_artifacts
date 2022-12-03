from metrics import MSE, SSIM, PSNR, Dice, Jaccard, NormalizedFoM
from data_wrappers import BDD10kEdges, CityScapesEdges, CityScapesRain, apply_roi
import typing as tp
from canny import canny, stat_canny
import numpy as np
import streamlit as st
import random
import cv2 as cv
import pandas as pd
from os import path
from benchmark import binarize

edge_metric = tp.Callable[[np.ndarray, np.ndarray], np.ndarray]
selection_metric = tp.Callable[[np.ndarray, np.ndarray, np.ndarray], int]

def to_regularized(metric: edge_metric, bigger_is_better: bool, reg_param: float) -> selection_metric:
    def f(pred, gt):
        metric_value = metric(pred, gt)
        reg_offset = 0
        if reg_param != 0:
            reg_offset = reg_param * np.mean(pred)
        if bigger_is_better:
            metric_value -= reg_offset
        else:
            metric_value += reg_offset
        return metric_value
    def g(pred1, pred2, img):
        m1 = f(pred1, img)
        m2 = f(pred2, img)
        mv = [m1, m2]
        if bigger_is_better:
            return max([0, 1], key=lambda i: mv[i])
        else:
            return min([0, 1], key=lambda i: mv[i])
    return g


def get_metric_dict(reg_params: tp.List[float]) -> tp.Dict[str, edge_metric]:
    md = {}
    for r in reg_params:
        md[f"MSE_reg_{r}"] = to_regularized(MSE(), False, r)
        md[f"SSIM_reg_{r}"] = to_regularized(SSIM(), True, r)
        md[f"PSNR_reg_{r}"] = to_regularized(PSNR(), True, r)
        md[f"Dice_reg_{r}"] = to_regularized(Dice(), True, r)
        md[f"Jaccard_reg_{r}"] = to_regularized(Jaccard(), True, r)
        md[f"NFOM_fp_0.1_fn_0.4_reg_{r}"] = to_regularized(NormalizedFoM(0.1, 0.4), True, r)
        md[f"NFOM_fp_0.1_fn_0.2_reg_{r}"] = to_regularized(NormalizedFoM(0.1, 0.2), True, r)
        md[f"NFOM_fp_0.2_fn_0.2_reg_{r}"] = to_regularized(NormalizedFoM(0.2, 0.2), True, r)
        md[f"NFOM_fp_0.2_fn_0.1_reg_{r}"] = to_regularized(NormalizedFoM(0.2, 0.1), True, r)
        md[f"NFOM_fp_0.4_fn_0.1_reg_{r}"] = to_regularized(NormalizedFoM(0.4, 0.1), True, r)
    return md

if "metrics" not in st.session_state:
    st.session_state["metrics"] = get_metric_dict([0, 0.2, 0.5])

st.title("Выбор метрики для детектирования линий")

ds_used = st.selectbox("Выберите датасет", options=["BDD10k", "CityScapes", "CityScapesRain"])
ds_root = st.text_input("Введите путь до датасета")
if not ds_root or not ds_used:
    st.stop()

if ds_used == "BDD10k":
    ds = BDD10kEdges(ds_root, "train")
elif ds_used == "CityScapes":
    ds = CityScapesEdges(ds_root, "train")
elif ds_used == "CityScapesRain":
    ds = CityScapesRain(ds_root, "train")

ds_data = ds_used + "_metric_stats.csv"
if path.isfile(ds_data):
    stats_df = pd.read_csv(ds_data, index_col="img_id")
    st.write(f"Размечено {stats_df.shape[0]} фотографий(я)")
else:
    md = st.session_state["metrics"]
    metrics_list = dict(zip(md.keys(), [[] for i in md]))
    metrics_list["human"] = []
    metrics_list["img_id"] = []
    stats_df = pd.DataFrame(metrics_list)
    stats_df = stats_df.set_index("img_id")
    st.write(f"Нет размеченных фотографий")

if "img_id" not in st.session_state:
    st.session_state["img_id"] = random.randint(0, len(ds) - 1)
img_id = st.session_state["img_id"]
img, gt = ds[img_id]
img = apply_roi(img)
gt = apply_roi(gt)
gt = binarize(gt, 250)
st.image(cv.cvtColor(img, cv.COLOR_BGR2RGB), caption="Оригинальная фотография")

edg1 = binarize(canny(img, 40, 140), 250)
edg2 = binarize(canny(img, 60, 160), 250)

st.image(edg1 * 255, "Границы 1")
st.image(edg2 * 255, "Границы 2")

user_choice = st.radio("Выберите картинку, где алгоритм лучше передал линии на дороге",
                            options=["Границы 1", "Границы 2", "Затрудняюсь ответить"])
if not st.button("Далее"):
    st.stop()

user_score = {
    "Границы 1": 0,
    "Границы 2": 1,
    "Затрудняюсь ответить": 0.5
}[user_choice]

new_row = {"img_id": img_id, "human": user_score}
for name, metric in st.session_state["metrics"].items():
    new_row[name] = metric(edg1, edg2, gt)
new_row = pd.DataFrame(new_row, index=[0])
new_row = new_row.set_index("img_id")
stats_df = stats_df.append(new_row)
stats_df.to_csv(ds_data)

del st.session_state["img_id"]
st.experimental_rerun()
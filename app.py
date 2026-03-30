import os
import re
import uuid
import math
from datetime import date, datetime, timedelta
from dateutil import tz

import pandas as pd
import streamlit as st
from filelock import FileLock

# 可選：資料持久化到 Hugging Face Dataset
USE_DATASET = bool(os.environ.get("HF_TOKEN") and os.environ.get("DATASET_REPO"))
if USE_DATASET:
    from huggingface_hub import HfApi, hf_hub_download

# 檔案與時區
DATA_DIR = "data"  # 若在 Hugging Face Spaces 遇到寫入權限，可改為 "/tmp/data" 或（有持久化）"/data/appdata"
os.makedirs(DATA_DIR, exist_ok=True)

TASKS_CSV = os.path.join(DATA_DIR, "tasks.csv")
TASKS_LOCK = TASKS_CSV + ".lock"

# 你在澳門，預設為 Asia/Macau（可視需要改為 Asia/Shanghai / Asia/Taipei / Asia/Hong_Kong）
LOCAL_TZ = tz.gettz("Asia/Macau")

# 欄位定義（工作事項）
# ⚠️ 已移除「狀態」欄位，改用「完成」布林欄位
TASK_COLUMNS = [
    "id",
    "完成",            # 是否完成（True/False），在最前欄顯示為可勾選的 checkbox
    "Action Date",     # 跟進行動日（主要日期）
    "預計完成日期",     # Due Date
    "Category",        # 分類
    "標題",
    "描述",
    "負責人",          # 可多位（, ; / 、 空白 分隔）
    "對口跟進人",
    "Pending Party",   # 例：MKT / 其他部門
    "建立時間",
    "最後更新",
]

# ========== 基本工具 ==========
def now_iso() -> str:
    return datetime.now(tz=LOCAL_TZ).isoformat(timespec="seconds")

def to_bool_strict(x) -> bool:
    """
    嚴格布林轉換：
    - True/False -> 原值
    - None/NaN   -> False
    - 字串：'true','1','yes','y','是','已完成' => True
            'false','0','no','n','未完成',''  => False
      其他未知字串 => False（保守）
    - 數值：1 => True；0 或其他 => False
    """
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, float) and math.isnan(x):
        return False
    if isinstance(x, (int, float)):
        return True if int(x) == 1 else False
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y", "是", "已完成"}:
        return True
    if s in {"false", "0", "no", "n", "未完成", ""}:
        return False
    return False

def ensure_tasks_file():
    if not os.path.exists(TASKS_CSV):
        with FileLock(TASKS_LOCK):
            pd.DataFrame(columns=TASK_COLUMNS).to_csv(TASKS_CSV, index=False, encoding="utf-8-sig")

def migrate_to_new_schema(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    舊欄位映射到新欄位；僅保留新版欄位順序。
    - 跟進人 -> 對口跟進人
    - 預計完成日 -> 預計完成日期；若無 Action Date 則沿用至 Action Date（舊版過渡）
    - 狀態 -> 完成（完成=True；其餘=False）
    - 最終統一正規化「完成」為布林
    """
    df = df_in.copy()

    # 跟進人 -> 對口跟進人
    if "對口跟進人" not in df.columns:
        df["對口跟進人"] = df["跟進人"] if "跟進人" in df.columns else ""

    # 預計完成日 -> 預計完成日期
    if "預計完成日期" not in df.columns:
        df["預計完成日期"] = df["預計完成日"] if "預計完成日" in df.columns else ""

    # 若無 Action Date，但有「預計完成日」，沿用到 Action Date（舊版過渡）
    if "Action Date" not in df.columns:
        df["Action Date"] = df["預計完成日"] if "預計完成日" in df.columns else df.get("Action Date", "")

    # 狀態 -> 完成（先做舊欄對映）
    if "完成" not in df.columns:
        if "狀態" in df.columns:
            df["完成"] = df["狀態"].apply(lambda s: True if str(s).strip() == "已完成" else False)
        else:
            df["完成"] = False

    # 最終嚴格正規化一次
    df["完成"] = df["完成"].apply(to_bool_strict)

    # 補齊缺欄
    for c in TASK_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    # 僅保留新版欄位順序
    return df[TASK_COLUMNS]

def load_tasks_local() -> pd.DataFrame:
    ensure_tasks_file()
    with FileLock(TASKS_LOCK):
        df = pd.read_csv(TASKS_CSV, dtype=str).fillna("")
    df = migrate_to_new_schema(df)
    # 雙保險：再正規化一次
    if "完成" in df.columns:
        df["完成"] = df["完成"].apply(to_bool_strict)
    return df

def save_tasks_local(df: pd.DataFrame):
    # 寫回前，先確保「完成」是正確布林
    df2 = df.copy()
    if "完成" in df2.columns:
        df2["完成"] = df2["完成"].apply(to_bool_strict)
    with FileLock(TASKS_LOCK):
        df2.to_csv(TASKS_CSV, index=False, encoding="utf-8-sig")

def load_tasks() -> pd.DataFrame:
    if not USE_DATASET:
        return load_tasks_local()
    repo_id = os.environ["DATASET_REPO"]
    token = os.environ["HF_TOKEN"]
    try:
        path = hf_hub_download(
            repo_id=repo_id, filename="tasks.csv",
            repo_type="dataset", token=token, force_download=True
        )
        df = pd.read_csv(path, dtype=str).fillna("")
        df = migrate_to_new_schema(df)
        save_tasks_local(df)  # 同步本地
        return df
    except Exception:
        return load_tasks_local()

def save_tasks(df: pd.DataFrame):
    df = df[TASK_COLUMNS].copy()
    if "完成" in df.columns:
        df["完成"] = df["完成"].apply(to_bool_strict)
    save_tasks_local(df)
    if USE_DATASET:
        repo_id = os.environ["DATASET_REPO"]
        token = os.environ["HF_TOKEN"]
        api = HfApi(token=token)
        tmp = os.path.join(DATA_DIR, "_tmp_tasks.csv")
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        try:
            api.upload_file(path_or_fileobj=tmp, path_in_repo="tasks.csv", repo_id=repo_id, repo_type="dataset")
        except Exception as e:
            st.warning(f"⚠️ 同步 tasks.csv 到 Dataset 失敗：{e}")

def to_date_safe(s):
    """轉成 date；接受 str/date/datetime/None"""
    if s is None or (isinstance(s, float) and pd.isna(s)) or str(s).strip() == "":
        return None
    if isinstance(s, date) and not isinstance(s, datetime):
        return s
    if isinstance(s, datetime):
        return s.date()
    try:
        return datetime.fromisoformat(str(s)).date()
    except Exception:
        try:
            return pd.to_datetime(s).date()
        except Exception:
            return None

# --- 將 date/str/None 轉為 YYYY-MM-DD 字串（用於保存到 CSV） ---
def to_iso_date(val):
    """接受 datetime.date / datetime.datetime / str / None，回傳 'YYYY-MM-DD' 或空字串。"""
    if val is None or (isinstance(val, float) and pd.isna(val)) or str(val).strip() == "":
        return ""
    if isinstance(val, date) and not isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, datetime):
        return val.date().isoformat()
    d = to_date_safe(val)
    return d.isoformat() if d else ""

def split_people_list(s: str):
    """將『負責人』字串切成清單，支援 , ， ; ； / 、 以及空白。"""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    parts = re.split(r"[;,/、，；\s]+", str(s))
    return [p.strip() for p in parts if p.strip()]

# --- 將日期欄位就地轉成 datetime.date 以供 DateColumn 使用 ---
def coerce_dates_for_editor(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    將 df[cols] 內的值轉成 datetime.date 或 None（字串/NaN 也會被轉），
    以便 st.data_editor 的 DateColumn 不再報 ColumnDataKind.STRING 的型別不相容錯誤。
    回傳 df（就地修改）。
    """
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(to_date_safe)
    return df

# ---- 逾期 / 需儘快跟進 判定與樣式（列表與 Weekly 皆可用）----
def is_overdue_row(due_str: str, done) -> bool:
    """逾期：預計完成日期 < 今日 且 完成=False"""
    d = to_date_safe(due_str)
    if d is None:
        return False
    return (d < date.today()) and (not to_bool_strict(done))

def is_followup_soon_row(action_str: str, done) -> bool:
    """需儘快跟進：Action Date < 今日 且 完成=False"""
    d = to_date_safe(action_str)
    if d is None:
        return False
    return (d < date.today()) and (not to_bool_strict(done))

def build_overdue_mask(df: pd.DataFrame) -> list[bool]:
    return [
        is_overdue_row(due, done)
        for due, done in zip(df.get("預計完成日期", ""), df.get("完成", False))
    ]

def build_followup_soon_mask(df: pd.DataFrame) -> list[bool]:
    return [
        is_followup_soon_row(action, done)
        for action, done in zip(df.get("Action Date", ""), df.get("完成", False))
    ]

def style_rows_with_masks(df: pd.DataFrame,
                          overdue_mask: list[bool],
                          followup_mask: list[bool]):
    """行級樣式（優先：逾期=紅底 > 需儘快跟進=黃底 > 無）"""
    rows = []
    for o, f in zip(overdue_mask, followup_mask):
        if o:
            rows.append(["background-color: #ffecec; color:#c00; font-weight:600"] * df.shape[1])
        elif f:
            rows.append(["background-color: #fff3cd; color:#8a6d3b; font-weight:600"] * df.shape[1])
        else:
            rows.append([""] * df.shape[1])
    return df.style.apply(lambda _df: pd.DataFrame(rows, index=_df.index, columns=_df.columns), axis=None)

# ========== 介面設定 ==========
st.set_page_config(page_title="Task Tracker", page_icon="✅", layout="wide")
st.title("✅ Task Tracker")
st.caption("欄位：完成 / Action Date / 預計完成日期 / Category / 標題 / 描述 / 負責人 / 對口跟進人 / Pending Party")

page = st.sidebar.radio("功能選單", ["📋 待辦清單", "🗓️ Weekly Key Focus", "➕ 新增/編輯 任務"], index=0)

st.sidebar.markdown("### 使用說明")
st.sidebar.info("""
- 在「📋 待辦清單」可用 **Category**、**負責人**、**標題關鍵字** 篩選，並可 **隱藏已完成**、**只看逾期**、**只看需儘快跟進**。
  點 **切換為編輯模式** 後在同一塊編輯/刪除並保存（最前面為『完成』勾選；日期欄可用小日曆）。
- 「🗓️ Weekly Key Focus」：以**週一~週日**為一週；僅分兩塊顯示：
  - **我們負責的工作（Pending Party = MKT）**
  - **我們要跟進的工作（Pending Party ≠ MKT）**
  本頁會把 **逾期（🔴）** 與 **需儘快跟進（🟡）** 的工作事項**合併**到上述兩塊中（不再獨立列出），並保留紅/黃底樣式。
  並提供 **負責人** 與 **Category** 篩選，以及 **隱藏已完成** 的切換。
- 逾期：**預計完成日期 < 今日** 且 **完成=False**；需儘快跟進：**Action Date < 今日** 且 **完成=False**。
""")

# 載入資料
df_tasks = load_tasks()

# --- 保證所有列都有 id（若缺少或為空則補一個 UUID）---
def ensure_id_val(x):
    return x if isinstance(x, str) and x.strip() else str(uuid.uuid4())

if "id" not in df_tasks.columns:
    df_tasks["id"] = [str(uuid.uuid4()) for _ in range(len(df_tasks))]
else:
    df_tasks["id"] = df_tasks["id"].apply(ensure_id_val)

# --- 保證「完成」欄存在且為布林邏輯值（True/False）---
if "完成" not in df_tasks.columns:
    df_tasks["完成"] = False
df_tasks["完成"] = df_tasks["完成"].apply(to_bool_strict)

# 立即保存一次（內容已正規化，安全）
save_tasks(df_tasks)

ALL_CATS = sorted([c for c in df_tasks["Category"].unique() if c])

# ========= 資料來源狀態（側欄顯示） =========
with st.sidebar.expander("📦 資料來源狀態", expanded=False):
    if USE_DATASET:
        st.success(
            "✅ 使用 Hugging Face Dataset\n\n"
            f"Repo：{os.environ.get('DATASET_REPO')}"
        )
    else:
        st.warning(
            "⚠️ 尚未連接 Hugging Face Dataset\n\n"
            "目前使用本地 CSV 檔案"
        )

# ========== 📋 待辦清單 ==========
if page == "📋 待辦清單":
    st.subheader("📋 待辦清單")

    # 篩選區
    c1, c2, c3 = st.columns([1.6, 1, 1])
    with c1:
        kw_title = st.text_input("標題包含（關鍵字）", placeholder="例如：MLM", key="kw_title").strip()
    with c2:
        f_cats = st.multiselect("Category 篩選（可複選）", options=ALL_CATS, key="f_cats")
    with c3:
        edit_mode = st.toggle("切換為編輯模式", value=False, key="todo_edit_mode")

    c4, c5 = st.columns([1, 1])
    with c4:
        all_owners = sorted({p for s in df_tasks["負責人"].fillna("") for p in split_people_list(s)})
        owners_pick = st.multiselect("負責人篩選（可複選）", options=all_owners, default=[], key="todo_owner_multi")
    with c5:
        hide_done = st.checkbox("隱藏已完成工作事項", value=False, key="hide_done_tasks")

    c6, c7 = st.columns([1, 1])
    with c6:
        only_overdue_filter = st.checkbox("只看逾期", value=False, key="todo_only_overdue")
    with c7:
        only_followup_filter = st.checkbox("只看需儘快跟進", value=False, key="todo_only_followup")

    # 顯示欄（完成放最前面）
    display_cols = ["完成", "Action Date", "預計完成日期", "Category", "標題", "描述", "負責人", "對口跟進人", "Pending Party"]

    # 準備清單（套用篩選）
    view = df_tasks.copy()
    if kw_title:
        view = view[view["標題"].str.contains(kw_title, case=False, na=False)]
    if f_cats:
        view = view[view["Category"].isin(f_cats)]
    if owners_pick:
        targets = {o.lower().strip() for o in owners_pick}
        view = view[
            view["負責人"].apply(lambda s: any(p.lower().strip() in targets for p in split_people_list(s)))
        ]
    if hide_done:
        view = view[~view["完成"].apply(to_bool_strict)]

    # 只看逾期 / 只看需儘快跟進（可並用）
    ov_mask_all = build_overdue_mask(view)
    fu_mask_all = build_followup_soon_mask(view)
    if only_overdue_filter and not only_followup_filter:
        view = view[ov_mask_all]
    elif only_followup_filter and not only_overdue_filter:
        view = view[fu_mask_all]
    elif only_overdue_filter and only_followup_filter:
        union_mask = [o or f for o, f in zip(ov_mask_all, fu_mask_all)]
        view = view[union_mask]

    # 排序（以 Action Date 為主）
    view["_ActionDate_"] = view["Action Date"].apply(lambda s: to_date_safe(s) or date(2999, 12, 31))
    view.sort_values(by=["_ActionDate_", "最後更新"], ascending=[True, False], inplace=True)

    # 唯讀或編輯渲染
    if not edit_mode:
        df_show = view[display_cols].reset_index(drop=True)
        # 🔴/🟡 標示（僅待辦清單保留）
        ov_mask = build_overdue_mask(df_show)
        fu_mask = build_followup_soon_mask(df_show)
        st.dataframe(
            style_rows_with_masks(df_show, ov_mask, fu_mask),
            use_container_width=True,
            hide_index=True,
            height=min(560, 80 + 28 * (len(df_show) + 1))
        )
    else:
        edit_df = view.copy()
        if "刪除" not in edit_df.columns:
            edit_df["刪除"] = False

        # 編輯模式：保證 id 存在；顯示欄位（完成 + display_cols 其他 + 刪除）
        editor_cols = ["id"] + display_cols + ["刪除"]
        editor_cols = [c for c in editor_cols if c in edit_df.columns]
        edit_df = edit_df[editor_cols]

        if "id" not in edit_df.columns:
            st.error("資料缺少 id 欄位，無法進入編輯模式。")
            st.stop()

        # ✅ 在丟進 data_editor 前，把兩個日期欄轉成 date 物件
        coerce_dates_for_editor(edit_df, ["Action Date", "預計完成日期"])

        st.caption("在表格內直接勾選『完成』或修改欄位（日期欄可用小日曆），或勾選『刪除』後按下方 **保存**。")
        st_edit = st.data_editor(
            edit_df.set_index("id"),
            use_container_width=True,
            height=min(560, 120 + 28 * (len(edit_df) + 3)),
            num_rows="dynamic",
            column_config={
                "完成": st.column_config.CheckboxColumn("完成", help="打勾表示此工作已完成"),
                "Action Date": st.column_config.DateColumn("Action Date", format="YYYY-MM-DD"),
                "預計完成日期": st.column_config.DateColumn("預計完成日期", format="YYYY-MM-DD"),
                "刪除": st.column_config.CheckboxColumn("刪除", help="勾選後保存即刪除該列"),
            }
        )

        if st.button("💾 保存更改（待辦清單）", type="primary", key="save_todo"):
            upd = st_edit.reset_index().copy()
            if "_ActionDate_" in upd.columns:
                upd.drop(columns=["_ActionDate_"], inplace=True, errors="ignore")

            # 1) 刪除勾選
            keep = upd[~upd["刪除"].fillna(False)].copy()

            # 2) 補必要欄位/時間/ID + 型別處理
            def ensure_id2(x):
                return x if isinstance(x, str) and x.strip() else str(uuid.uuid4())
            keep["id"] = keep["id"].apply(ensure_id2)
            keep["完成"] = keep["完成"].apply(to_bool_strict)
            keep["Action Date"] = keep["Action Date"].apply(to_iso_date)
            keep["預計完成日期"] = keep["預計完成日期"].apply(to_iso_date)
            keep["最後更新"] = now_iso()
            if "建立時間" not in keep.columns:
                keep["建立時間"] = now_iso()
            keep["建立時間"] = keep["建立時間"].replace({"": now_iso()})

            # 3) 回寫主表（僅替換本次視圖出現過的 id）
            master = df_tasks.set_index("id").copy()
            master.drop(index=upd["id"].tolist(), errors="ignore", inplace=True)

            for c in TASK_COLUMNS:
                if c not in keep.columns:
                    keep[c] = "" if c != "完成" else False

            keep_aligned = keep[TASK_COLUMNS].copy()
            master = pd.concat([master, keep_aligned.set_index("id")], axis=0)

            out = master.reset_index()[TASK_COLUMNS]
            save_tasks(out)
            st.success("✅ 已保存（含新增/修改/刪除）。")
            st.rerun()

# ========== 🗓️ Weekly Key Focus ==========
elif page == "🗓️ Weekly Key Focus":
    st.subheader("🗓️ Weekly Key Focus（以 Action Date 為主）")
    st.caption("🔴 逾期（預計完成日期 < 今日且未完成）｜🟡 需儘快跟進（Action Date < 今日且未完成）｜本週以週一~週日為區間")

    # 週區間：本週週一 ~ 週日
    today = date.today()
    week_start = today - timedelta(days=today.weekday())  # Monday
    week_end = week_start + timedelta(days=6)             # Sunday

    edit_mode_wkf = st.toggle("切換為編輯模式（本頁編輯會同步至工作清單）", value=False, key="wkf_edit_mode")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        hide_done_wkf = st.checkbox("隱藏已完成", value=True, key="wkf_hide_done")
    with c2:
        # 篩選：負責人
        dfk_for_filter = df_tasks.copy()
        all_owners_w = sorted({p for s in dfk_for_filter["負責人"].fillna("") for p in split_people_list(s)})
        owners_pick_w = st.multiselect("負責人篩選（可複選）", options=all_owners_w, default=[], key="wkf_owner_multi")
    with c3:
        # 篩選：Category
        all_cats_w = sorted([c for c in df_tasks["Category"].fillna("").unique() if c])
        cats_pick_w = st.multiselect("Category 篩選（可複選）", options=all_cats_w, default=[], key="wkf_cat_multi")

    # 準備資料
    dfk = df_tasks.copy()
    dfk["__adate__"] = dfk["Action Date"].apply(to_date_safe)

    base_cols_show = ["完成", "Action Date", "預計完成日期", "Category", "標題", "描述", "負責人", "對口跟進人", "Pending Party"]

    def apply_common_filters(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if hide_done_wkf:
            out = out[~out["完成"].apply(to_bool_strict)]
        if owners_pick_w:
            targets = {o.lower().strip() for o in owners_pick_w}
            out = out[out["負責人"].apply(lambda s: any(p.lower().strip() in targets for p in split_people_list(s)))]
        if cats_pick_w:
            out = out[out["Category"].isin(cats_pick_w)]
        return out

    # 是否屬於本週，或屬於注意項（逾期/需儘快跟進）
    def in_week_or_attention(row) -> bool:
        ad = row["__adate__"]
        in_week = (ad is not None) and (week_start <= ad <= week_end)
        overdue = is_overdue_row(row.get("預計完成日期", ""), row.get("完成", False))
        follow = is_followup_soon_row(row.get("Action Date", ""), row.get("完成", False))
        return in_week or overdue or follow

    def render_wkf_block(block_title: str, base_df: pd.DataFrame, key_prefix: str):
        st.markdown(block_title)
        if base_df.empty:
            st.info("（此區目前無資料）")
            return

        edit_df = base_df.copy()
        if "刪除" not in edit_df.columns:
            edit_df["刪除"] = False

        cols = ["id"] + base_cols_show + ["刪除"]
        cols = [c for c in cols if c in edit_df.columns]
        edit_df = edit_df[cols]

        if "id" not in edit_df.columns:
            st.error("資料缺少 id 欄位，無法進入編輯模式。")
            st.stop()

        # 轉成 date 物件以供 DateColumn
        coerce_dates_for_editor(edit_df, ["Action Date", "預計完成日期"])

        if not edit_mode_wkf:
            df_show = edit_df.set_index("id")[base_cols_show].reset_index(drop=True)
            ov_mask = build_overdue_mask(df_show)
            fu_mask = build_followup_soon_mask(df_show)
            st.dataframe(
                style_rows_with_masks(df_show, ov_mask, fu_mask),
                use_container_width=True,
                hide_index=True,
                height=min(420, 80 + 28 * (len(df_show) + 1))
            )
        else:
            editor = st.data_editor(
                edit_df.set_index("id"),
                key=f"{key_prefix}_editor",
                use_container_width=True,
                height=min(480, 120 + 28 * (len(edit_df) + 3)),
                num_rows="dynamic",
                column_config={
                    "完成": st.column_config.CheckboxColumn("完成"),
                    "Action Date": st.column_config.DateColumn("Action Date", format="YYYY-MM-DD"),
                    "預計完成日期": st.column_config.DateColumn("預計完成日期", format="YYYY-MM-DD"),
                    "刪除": st.column_config.CheckboxColumn("刪除", help="勾選後保存即刪除此列"),
                }
            )
            if st.button(f"💾 保存 {block_title}", key=f"{key_prefix}_save", type="primary"):
                upd = editor.reset_index().copy()
                keep = upd[~upd["刪除"].fillna(False)].copy()

                def ensure_id2(x):
                    return x if isinstance(x, str) and x.strip() else str(uuid.uuid4())
                keep["id"] = keep["id"].apply(ensure_id2)
                keep["完成"] = keep["完成"].apply(to_bool_strict)
                keep["Action Date"] = keep["Action Date"].apply(to_iso_date)
                keep["預計完成日期"] = keep["預計完成日期"].apply(to_iso_date)
                keep["最後更新"] = now_iso()
                if "建立時間" not in keep.columns:
                    keep["建立時間"] = now_iso()
                keep["建立時間"] = keep["建立時間"].replace({"": now_iso()})

                master = df_tasks.set_index("id").copy()
                master.drop(index=upd["id"].tolist(), errors="ignore", inplace=True)
                for c in TASK_COLUMNS:
                    if c not in keep.columns:
                        keep[c] = "" if c != "完成" else False
                keep_aligned = keep[TASK_COLUMNS].copy()
                master = pd.concat([master, keep_aligned.set_index("id")], axis=0)
                out = master.reset_index()[TASK_COLUMNS]
                save_tasks(out)
                st.success(f"✅ {block_title} 已保存（同步至工作清單）。")
                st.rerun()

    # ====== 區塊一：我們負責（Pending Party = MKT） ======
    mkt_df = dfk[dfk["Pending Party"].fillna("") == "MKT"].copy()
    mkt_df = mkt_df[mkt_df.apply(in_week_or_attention, axis=1)].copy()
    mkt_df = apply_common_filters(mkt_df)
    # 排序：Action Date（None 最後）→ 最後更新（新到舊）
    mkt_df["_ad_sort_"] = mkt_df["__adate__"].apply(lambda d: d or date(2999, 12, 31))
    mkt_df.sort_values(by=["_ad_sort_", "最後更新"], ascending=[True, False], inplace=True)
    mkt_df.drop(columns=["_ad_sort_"], inplace=True, errors="ignore")

    render_wkf_block("### 我們負責的工作（Pending Party = MKT）", mkt_df, "WKF_MKT")

    st.divider()

    # ====== 區塊二：我們要跟進（Pending Party ≠ MKT） ======
    non_df = dfk[dfk["Pending Party"].fillna("") != "MKT"].copy()
    non_df = non_df[non_df.apply(in_week_or_attention, axis=1)].copy()
    non_df = apply_common_filters(non_df)
    non_df["_ad_sort_"] = non_df["__adate__"].apply(lambda d: d or date(2999, 12, 31))
    non_df.sort_values(by=["_ad_sort_", "最後更新"], ascending=[True, False], inplace=True)
    non_df.drop(columns=["_ad_sort_"], inplace=True, errors="ignore")

    render_wkf_block("### 我們要跟進的工作（Pending Party ≠ MKT）", non_df, "WKF_NON")

    st.caption("顏色說明：🔴 逾期（預計完成日期 < 今日且未完成）｜🟡 需儘快跟進（Action Date < 今日且未完成）")

# ========== ➕ 新增/編輯 任務 ==========
else:
    st.subheader("➕ 新增/編輯 任務（Action Date / 預計完成日期 / Category 為主要欄位）")

    # ---------- 只用 Session State 控制（避免 value= 與 session_state 衝突） ----------
    ADD_FORM_KEYS = [
        "add_action_date",
        "add_due_date",
        "add_category",
        "add_title",
        "add_desc",
        "add_owner",
        "add_contact",
        "add_pending",
        "add_done",
    ]

    # 若上一輪請求了重置，這一輪在建立 widget 之前清掉 key（避免 StreamlitAPIException）
    if st.session_state.get("_add_form_needs_reset", False):
        for k in ADD_FORM_KEYS:
            st.session_state.pop(k, None)  # 刪除 key，讓初始化重新給預設值
        st.session_state["_add_form_needs_reset"] = False

    def init_add_form_state():
        st.session_state.setdefault("add_action_date", date.today())
        st.session_state.setdefault("add_due_date", date.today())  # 若 Streamlit 支援 None，可改為 None
        st.session_state.setdefault("add_category", "")
        st.session_state.setdefault("add_title", "")
        st.session_state.setdefault("add_desc", "")
        st.session_state.setdefault("add_owner", "")
        st.session_state.setdefault("add_contact", "")
        st.session_state.setdefault("add_pending", "")
        st.session_state.setdefault("add_done", False)

    def request_clear_add_form_state():
        # 不在本輪直接改 widget 的 key，改為留下旗標，下一輪（rerun 後）在建立 widget 前清空
        st.session_state["_add_form_needs_reset"] = True
        st.rerun()

    # 初始化（建立 widget 前）
    init_add_form_state()

    # 不使用 value=，只使用 key；避免 Enter 清空，使用 clear_on_submit=False
    with st.form("add_task_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            action_date = st.date_input("Action Date *", key="add_action_date")
        with col2:
            # 若版本不支援 None 預設，維持今天
            due_date = st.date_input("預計完成日期（可選）", key="add_due_date")
        with col3:
            category = st.text_input("Category *", placeholder="例如：MLM / 活動 / 報表", key="add_category")

        title = st.text_input("標題 *", key="add_title")
        desc = st.text_area("描述", key="add_desc")

        col4, col5, col6 = st.columns(3)
        with col4:
            owner = st.text_input("負責人 *（多人以 , ; / 或空白分隔）", key="add_owner")
        with col5:
            contact = st.text_input("對口跟進人", key="add_contact")
        with col6:
            pending = st.text_input("Pending Party *", placeholder="例如：MKT / 其他部門", key="add_pending")

        done_flag = st.checkbox("完成", key="add_done")

        st.caption("提示：在表單中按 Enter 會提交。未通過驗證時不會清空欄位。")
        submitted = st.form_submit_button("新增任務")

    if submitted:
        # ✅ 通過驗證才保存；未通過不清空
        if not (action_date and category and title and owner and pending):
            st.error("請填：Action Date / Category / 標題 / 負責人 / Pending Party。")
        else:
            new_row = {
                "id": str(uuid.uuid4()),
                "完成": to_bool_strict(done_flag),   # 雖然 checkbox 回傳布林，仍以嚴格轉換保險
                "Action Date": to_iso_date(action_date),
                "預計完成日期": to_iso_date(due_date),
                "Category": (category or "").strip(),
                "標題": (title or "").strip(),
                "描述": (desc or "").strip(),
                "負責人": (owner or "").strip(),
                "對口跟進人": (contact or "").strip(),
                "Pending Party": (pending or "").strip(),
                "建立時間": now_iso(),
                "最後更新": now_iso(),
            }
            df_new = pd.concat([df_tasks, pd.DataFrame([new_row])], ignore_index=True)
            save_tasks(df_new[TASK_COLUMNS])
            st.success("✅ 新增完成！")

            # ✅ 正確：延遲重置 + rerun（避免「widget 已實例化後改值」的錯誤）
            request_clear_add_form_state()

    st.divider()
    st.markdown("### ✏️ 批次編輯 / 刪除（勾選『刪除』後保存即可）")
    edit_df = df_tasks.copy()
    if "刪除" not in edit_df.columns:
        edit_df["刪除"] = False
    edit_cols = ["id", "完成", "Action Date", "預計完成日期", "Category", "標題", "描述", "負責人", "對口跟進人", "Pending Party", "刪除"]
    edit_cols = [c for c in edit_cols if c in edit_df.columns]

    # ✅ 先把日期欄轉成 date 物件再丟進 editor
    coerce_dates_for_editor(edit_df, ["Action Date", "預計完成日期"])

    ed = st.data_editor(
        edit_df[edit_cols].set_index("id"),
        use_container_width=True,
        height=min(560, 120 + 28 * (len(edit_df) + 3)),
        num_rows="dynamic",
        column_config={
            "完成": st.column_config.CheckboxColumn("完成"),
            "Action Date": st.column_config.DateColumn("Action Date", format="YYYY-MM-DD"),
            "預計完成日期": st.column_config.DateColumn("預計完成日期", format="YYYY-MM-DD"),
            "刪除": st.column_config.CheckboxColumn("刪除"),
        }
    )
    if st.button("💾 保存（批次編輯/刪除）", type="primary", key="save_bulk"):
        upd = ed.reset_index().copy()
        keep = upd[~upd["刪除"].fillna(False)].copy()

        def ensure_id2(x):
            return x if isinstance(x, str) and x.strip() else str(uuid.uuid4())
        keep["id"] = keep["id"].apply(ensure_id2)
        keep["完成"] = keep["完成"].apply(to_bool_strict)
        keep["Action Date"] = keep["Action Date"].apply(to_iso_date)
        keep["預計完成日期"] = keep["預計完成日期"].apply(to_iso_date)
        keep["最後更新"] = now_iso()
        if "建立時間" not in keep.columns:
            keep["建立時間"] = now_iso()
        keep["建立時間"] = keep["建立時間"].replace({"": now_iso()})

        for c in TASK_COLUMNS:
            if c not in keep.columns:
                keep[c] = "" if c != "完成" else False
        save_tasks(keep[TASK_COLUMNS])
        st.success("✅ 已保存（含新增/修改/刪除）。")
        st.rerun()

# （可選）側欄資料維護工具：若過去 CSV 真的被誤存成全 True，可按此修復一次
with st.sidebar.expander("🛠️ 資料維護工具（可選）"):
    st.caption("若過去因早期版本錯誤導致『完成』欄位判定錯誤，可按此按鈕進行嚴格修復（建議先備份 data/tasks.csv）")
    if st.button("修復『完成』欄位（嚴格轉換）", type="secondary", key="fix_done_field"):
        df_all = load_tasks()
        if "完成" in df_all.columns:
            before_true = int(df_all["完成"].apply(to_bool_strict).sum())
            total_rows = len(df_all)
            df_all["完成"] = df_all["完成"].apply(to_bool_strict)
            save_tasks(df_all)
            after_true = int(load_tasks()["完成"].apply(to_bool_strict).sum())
            st.success(f"修復完成：之前 {before_true}/{total_rows} True → 現在 {after_true}/{total_rows} True。")
            st.rerun()
        else:
            st.info("沒有找到『完成』欄位，無需修復。")

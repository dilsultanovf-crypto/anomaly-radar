import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Anomaly Radar",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомные стили
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 12px; }
    div[data-testid="stSidebar"] { width: 380px; }
    .anomaly-high { color: #e74c3c; font-weight: bold; }
    .anomaly-med { color: #f39c12; }
    .anomaly-low { color: #3498db; }
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# МАППИНГ СТОЛБЦОВ
# ═══════════════════════════════════════════════════════════════════
DEFAULT_COLUMNS = {
    'year':              'Год',
    'month':             'Месяц',
    'store_code':        'Код площадки',
    'department':        'Отдел',
    'group':             'Группа',
    'subgroup':          'Подгруппа',
    'city':              'Город',
    'city_detail':       'Город детальный',
    'branch':            'Аббр. филиала',
    'format':            'Формат',
    'sku_count':         'Количество SKU (продано)',
    'sales_cost':        'Сумма продаж по ЗЦ без НДС',
    'sales_retail':      'Сумма продаж по ПЦ без НДС',
    'sales_units':       'Продаши шт',
    'promo_share_pc':    'Доля промо в ПЦ',
    'promo_share_units': 'Доля промо в ШТ',
    'checks':            'Кол-во чеков подгруппы',
    'oos':               '%OOS',
}


# ═══════════════════════════════════════════════════════════════════
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Загрузка данных...")
def load_data(file) -> pd.DataFrame:
    """Загружает CSV, пробуя разные кодировки и разделители."""
    for enc in ['utf-8', 'cp1251', 'utf-8-sig', 'latin-1']:
        for sep in [',', ';', '\t']:
            try:
                file.seek(0)
                df = pd.read_csv(file, sep=sep, encoding=enc)
                if len(df.columns) >= 5:
                    return df
            except:
                continue
    return None


@st.cache_data(show_spinner="Подготовка данных...")
def prepare_data(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Переименовывает столбцы, приводит типы, создаёт доп. поля."""
    reverse_map = {v: k for k, v in col_map.items() if v in df.columns}
    df = df.rename(columns=reverse_map)

    numeric_cols = ['sku_count', 'sales_cost', 'sales_retail', 'sales_units',
                    'promo_share_pc', 'promo_share_units', 'checks', 'oos']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(0).astype(int)
    df['period'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)

    df['units_per_check'] = np.where(df['checks'] > 0, df['sales_units'] / df['checks'], 0)
    df['revenue_per_check'] = np.where(df['checks'] > 0, df['sales_retail'] / df['checks'], 0)
    df['margin'] = np.where(df['sales_cost'] > 0,
                            (df['sales_retail'] - df['sales_cost']) / df['sales_cost'], 0)

    df = df.sort_values(['store_code', 'subgroup', 'year', 'month']).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════
# АНАЛИТИЧЕСКОЕ ЯДРО
# ═══════════════════════════════════════════════════════════════════
METRICS = ['sales_units', 'sales_retail', 'checks', 'sku_count',
           'oos', 'promo_share_pc', 'promo_share_units',
           'units_per_check', 'revenue_per_check']

KEY_METRICS = ['sales_units', 'sales_retail', 'checks', 'sku_count']


def compute_baselines(df: pd.DataFrame, analysis_level: str,
                      yoy_weight: float = 0.6) -> pd.DataFrame:
    """Строит baseline: YoY + Peer."""
    peer_weight = 1.0 - yoy_weight

    # Определяем группировку по уровню анализа
    if analysis_level == 'Магазин':
        entity_cols = ['store_code']
        peer_cols = ['format', 'city', 'subgroup', 'year', 'month']
    elif analysis_level == 'Формат':
        entity_cols = ['format']
        peer_cols = ['city', 'subgroup', 'year', 'month']
    elif analysis_level == 'Город':
        entity_cols = ['city']
        peer_cols = ['subgroup', 'year', 'month']
    elif analysis_level == 'Филиал':
        entity_cols = ['branch']
        peer_cols = ['subgroup', 'year', 'month']
    else:
        entity_cols = ['store_code']
        peer_cols = ['format', 'city', 'subgroup', 'year', 'month']

    # Агрегация если уровень != магазин
    if analysis_level != 'Магазин':
        group_cols = entity_cols + ['department', 'group', 'subgroup', 'year', 'month']
        # Добавляем недостающие колонки для peer
        for c in peer_cols:
            if c not in group_cols:
                group_cols.append(c)
        # Убираем дубликаты, сохраняя порядок
        seen = set()
        unique_group_cols = []
        for c in group_cols:
            if c not in seen and c in df.columns:
                seen.add(c)
                unique_group_cols.append(c)
        group_cols = unique_group_cols

        agg_dict = {m: 'sum' for m in METRICS if m in df.columns}
        # Средние, а не суммы для rates
        for rate_col in ['oos', 'promo_share_pc', 'promo_share_units', 'units_per_check',
                         'revenue_per_check', 'margin']:
            if rate_col in agg_dict:
                agg_dict[rate_col] = 'mean'
        # Добавляем store_count
        agg_dict['store_code'] = 'nunique'

        extra_cols = ['format', 'city', 'city_detail', 'branch']
        for c in extra_cols:
            if c in df.columns and c not in group_cols:
                agg_dict[c] = 'first'

        df_agg = df.groupby(group_cols, as_index=False).agg(agg_dict)
        df_agg = df_agg.rename(columns={'store_code': 'store_count'})
        df = df_agg

    # YoY Baseline
    df_prev = df.copy()
    df_prev['year'] = df_prev['year'] + 1
    yoy_rename = {m: f'yoy_{m}' for m in METRICS if m in df_prev.columns}
    merge_keys = entity_cols + ['subgroup', 'year', 'month']
    merge_keys = [k for k in merge_keys if k in df_prev.columns]
    df_prev = df_prev[merge_keys + [m for m in METRICS if m in df_prev.columns]].rename(columns=yoy_rename)
    df = df.merge(df_prev, on=merge_keys, how='left')

    # Peer Baseline
    available_peer_cols = [c for c in peer_cols if c in df.columns]
    if len(available_peer_cols) >= 2:
        peer_metrics = [m for m in METRICS if m in df.columns]
        peer_medians = df.groupby(available_peer_cols)[peer_metrics].median().reset_index()
        peer_rename = {m: f'peer_{m}' for m in peer_metrics}
        peer_medians = peer_medians.rename(columns=peer_rename)
        df = df.merge(peer_medians, on=available_peer_cols, how='left')

    # Combined Baseline
    for m in METRICS:
        if m not in df.columns:
            continue
        yoy_col = f'yoy_{m}'
        peer_col = f'peer_{m}'
        baseline_col = f'baseline_{m}'

        has_yoy = df.get(yoy_col, pd.Series(dtype=float)).notna() if yoy_col in df.columns else False
        has_peer = peer_col in df.columns

        if yoy_col in df.columns and has_peer:
            df[baseline_col] = np.where(
                has_yoy,
                df[yoy_col] * yoy_weight + df[peer_col] * peer_weight,
                df[peer_col]
            )
        elif yoy_col in df.columns:
            df[baseline_col] = df[yoy_col]
        elif has_peer:
            df[baseline_col] = df[peer_col]
        else:
            df[baseline_col] = df[m]

    return df


def compute_anomalies(df: pd.DataFrame, z_threshold: float = 2.0,
                      vol_group_cols: list = None) -> pd.DataFrame:
    """Считает z-score и флагует аномалии."""
    if vol_group_cols is None:
        vol_group_cols = ['format', 'subgroup']
    vol_group_cols = [c for c in vol_group_cols if c in df.columns]

    for m in KEY_METRICS:
        if m not in df.columns:
            continue
        baseline_col = f'baseline_{m}'
        if baseline_col not in df.columns:
            continue

        dev_col = f'dev_{m}'
        z_col = f'z_{m}'
        flag_col = f'flag_{m}'

        df[dev_col] = np.where(
            df[baseline_col] > 0,
            (df[m] - df[baseline_col]) / df[baseline_col],
            0
        )

        if len(vol_group_cols) > 0:
            stats_df = df.groupby(vol_group_cols)[dev_col].agg(['mean', 'std']).reset_index()
            stats_df.columns = vol_group_cols + ['_mean', '_std']
            stats_df['_std'] = stats_df['_std'].replace(0, np.nan)
            df = df.merge(stats_df, on=vol_group_cols, how='left')
            df[z_col] = np.where(df['_std'].notna(),
                                  (df[dev_col] - df['_mean']) / df['_std'], 0)
            df = df.drop(columns=['_mean', '_std'])
        else:
            overall_mean = df[dev_col].mean()
            overall_std = df[dev_col].std()
            if overall_std and overall_std > 0:
                df[z_col] = (df[dev_col] - overall_mean) / overall_std
            else:
                df[z_col] = 0

        df[flag_col] = (df[z_col].abs() > z_threshold).astype(int)

    flag_cols = [f'flag_{m}' for m in KEY_METRICS if f'flag_{m}' in df.columns]
    df['severity'] = df[flag_cols].sum(axis=1)

    return df


def attribute_drivers(row) -> str:
    """Определяет драйверы аномалии."""
    drivers = []

    # OOS
    baseline_oos = row.get('baseline_oos', 0) or 0
    current_oos = row.get('oos', 0) or 0
    if baseline_oos > 0:
        oos_change = (current_oos - baseline_oos) / max(baseline_oos, 0.01)
    else:
        oos_change = current_oos - baseline_oos
    if oos_change > 0.3 or (current_oos > 10 and baseline_oos < 5):
        drivers.append('OOS')

    # Promo
    baseline_promo = row.get('baseline_promo_share_pc', 0) or 0
    current_promo = row.get('promo_share_pc', 0) or 0
    if baseline_promo > 0:
        promo_change = (current_promo - baseline_promo) / baseline_promo
        if abs(promo_change) > 0.3:
            drivers.append('Promo')
    elif current_promo > 0.1:
        drivers.append('Promo')

    # Traffic
    dev_checks = row.get('dev_checks', 0) or 0
    if dev_checks < -0.15 and row.get('flag_checks', 0) == 1:
        drivers.append('Traffic')

    # Conversion
    baseline_upc = row.get('baseline_units_per_check', 0) or 0
    current_upc = row.get('units_per_check', 0) or 0
    if abs(dev_checks) < 0.15 and baseline_upc > 0:
        conv_change = (current_upc - baseline_upc) / baseline_upc
        if abs(conv_change) > 0.2:
            drivers.append('Conversion')

    # Assortment
    dev_sku = row.get('dev_sku_count', 0) or 0
    if dev_sku < -0.15 and row.get('flag_sku_count', 0) == 1:
        drivers.append('Assortment')

    # Mix
    dev_units = row.get('dev_sales_units', 0) or 0
    dev_revenue = row.get('dev_sales_retail', 0) or 0
    if (abs(dev_units) < 0.15 and abs(dev_revenue) > 0.2) or \
       (abs(dev_revenue) < 0.15 and abs(dev_units) > 0.2):
        drivers.append('Mix')

    return ', '.join(drivers) if drivers else 'Undefined'


def compute_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Определяет скоуп: Local, Systemic, Market."""
    scope_map = {}
    for (subgroup, year, month), group in df.groupby(['subgroup', 'year', 'month']):
        total = len(group)
        anom = (group['severity'] > 0).sum()
        rate = anom / total if total > 0 else 0
        if rate > 0.5:
            scope = 'Market'
        elif rate > 0.2:
            scope = 'Systemic'
        else:
            scope = 'Local'
        scope_map[(subgroup, year, month)] = scope

    df['scope'] = df.apply(
        lambda r: scope_map.get((r['subgroup'], r['year'], r['month']), 'Local'),
        axis=1
    )
    return df


def filter_lfl(df: pd.DataFrame, entity_col: str = 'store_code') -> pd.DataFrame:
    """LFL: оставляет только сущности, присутствующие во ВСЕХ годах."""
    if entity_col not in df.columns:
        return df
    years = df['year'].unique()
    if len(years) < 2:
        return df

    entities_by_year = {}
    for y in years:
        entities_by_year[y] = set(df[df['year'] == y][entity_col].unique())

    lfl_entities = set.intersection(*entities_by_year.values())
    return df[df[entity_col].isin(lfl_entities)]


# ═══════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ЛОГИКА ПРИЛОЖЕНИЯ
# ═══════════════════════════════════════════════════════════════════

def main():
    st.title("🔍 Anomaly Radar")
    st.caption("Поиск и атрибуция аномалий в продажах")

    # ── ЗАГРУЗКА ФАЙЛА ──
    with st.sidebar:
        st.header("📁 Данные")
        uploaded_file = st.file_uploader("Загрузи CSV-файл", type=['csv', 'txt'])

    if uploaded_file is None:
        st.info("👈 Загрузи CSV-файл через боковую панель, чтобы начать.")
        st.markdown("""
        **Ожидаемые столбцы:**
        Год, Месяц, Код площадки, Отдел, Группа, Подгруппа, Город,
        Формат, Количество SKU, Сумма продаж по ПЦ без НДС,
        Продажи шт, Доля промо, Кол-во чеков, %OOS
        """)
        return

    # Загрузка
    raw_df = load_data(uploaded_file)
    if raw_df is None:
        st.error("Не удалось прочитать файл. Проверь формат (CSV, разделитель , или ;)")
        return

    df = prepare_data(raw_df, DEFAULT_COLUMNS)

    # ═══════════════════════════════════════════════════════════════
    # БОКОВАЯ ПАНЕЛЬ: НАСТРОЙКИ
    # ═══════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("---")

        # ── ВРЕМЕННЫЕ РАМКИ ──
        st.header("📅 Период")
        all_years = sorted(df['year'].unique())
        selected_years = st.multiselect("Годы", all_years, default=all_years,
                                         help="Какие годы включить в анализ")

        all_months = list(range(1, 13))
        month_names = {1:'Янв', 2:'Фев', 3:'Мар', 4:'Апр', 5:'Май', 6:'Июн',
                       7:'Июл', 8:'Авг', 9:'Сен', 10:'Окт', 11:'Ноя', 12:'Дек'}
        available_months = sorted(df[df['year'].isin(selected_years)]['month'].unique())
        selected_months = st.multiselect(
            "Месяцы",
            available_months,
            default=available_months,
            format_func=lambda x: f"{month_names.get(x, x)} ({x})"
        )

        st.markdown("---")

        # ── КАТЕГОРИИ (каскадный фильтр) ──
        st.header("📦 Категории")

        # Фильтруем df по уже выбранному периоду для каскадных списков
        period_mask = df['year'].isin(selected_years) & df['month'].isin(selected_months)
        df_period = df[period_mask]

        all_departments = sorted(df_period['department'].dropna().unique())
        selected_departments = st.multiselect("Отделы", ['Все'] + all_departments,
                                               default=['Все'])
        if 'Все' in selected_departments or not selected_departments:
            dept_mask = True
        else:
            dept_mask = df_period['department'].isin(selected_departments)

        df_dept = df_period[dept_mask] if isinstance(dept_mask, pd.Series) else df_period

        all_groups = sorted(df_dept['group'].dropna().unique())
        selected_groups = st.multiselect("Группы", ['Все'] + all_groups, default=['Все'])
        if 'Все' in selected_groups or not selected_groups:
            grp_mask = True
        else:
            grp_mask = df_dept['group'].isin(selected_groups)

        df_grp = df_dept[grp_mask] if isinstance(grp_mask, pd.Series) else df_dept

        all_subgroups = sorted(df_grp['subgroup'].dropna().unique())
        selected_subgroups = st.multiselect("Подгруппы", ['Все'] + all_subgroups,
                                             default=['Все'])

        st.markdown("---")

        # ── ГЕОГРАФИЯ И ФОРМАТ ──
        st.header("🏪 Магазины")

        all_cities = sorted(df_period['city'].dropna().unique())
        selected_cities = st.multiselect("Города", ['Все'] + all_cities, default=['Все'])

        if 'branch' in df.columns:
            all_branches = sorted(df_period['branch'].dropna().unique())
            selected_branches = st.multiselect("Филиалы", ['Все'] + all_branches,
                                                default=['Все'])
        else:
            selected_branches = ['Все']

        all_formats = sorted(df_period['format'].dropna().unique())
        selected_formats = st.multiselect("Форматы", ['Все'] + all_formats, default=['Все'])

        # Конкретные магазины (опционально)
        with st.expander("Конкретные магазины (опц.)"):
            store_input = st.text_area(
                "Коды магазинов (по одному на строку)",
                help="Оставь пустым, чтобы включить все. Или введи конкретные коды.",
                height=80
            )
            specific_stores = [s.strip() for s in store_input.split('\n') if s.strip()] \
                              if store_input.strip() else []

        st.markdown("---")

        # ── УРОВЕНЬ АНАЛИЗА ──
        st.header("📊 Уровень анализа")

        analysis_level = st.radio(
            "Сравнивать на уровне:",
            ['Магазин', 'Формат', 'Город', 'Филиал'],
            help="Магазин — каждый магазин отдельно. "
                 "Формат — агрегация по формату. "
                 "Город — агрегация по городу."
        )

        lfl_mode = st.toggle(
            "Только LFL",
            value=False,
            help="Like-for-Like: включить только точки, "
                 "которые работали во ВСЕХ выбранных годах"
        )

        st.markdown("---")

        # ── ПАРАМЕТРЫ МОДЕЛИ ──
        st.header("⚙️ Параметры модели")

        z_threshold = st.slider(
            "Порог Z-score",
            min_value=1.0, max_value=4.0, value=2.0, step=0.1,
            help="Чем выше — тем меньше аномалий (только самые яркие). "
                 "Рекомендация: 2.0"
        )

        yoy_weight = st.slider(
            "Вес YoY baseline",
            min_value=0.0, max_value=1.0, value=0.6, step=0.1,
            help="Остаток уходит на peer baseline. "
                 "0.6 = 60% YoY + 40% Peer"
        )

        min_severity = st.slider(
            "Минимальная severity",
            min_value=1, max_value=4, value=1,
            help="Сколько метрик должно быть аномальным одновременно"
        )

        st.markdown("---")

        # ── КНОПКА ЗАПУСКА ──
        run_button = st.button("🚀 Запустить анализ", type="primary", use_container_width=True)

    # ═══════════════════════════════════════════════════════════════
    # ПРИМЕНЕНИЕ ФИЛЬТРОВ
    # ═══════════════════════════════════════════════════════════════
    if not run_button and 'results' not in st.session_state:
        st.info("👈 Настрой параметры в боковой панели и нажми **Запустить анализ**")
        return

    if run_button:
        # Применяем все фильтры
        mask = (
            df['year'].isin(selected_years) &
            df['month'].isin(selected_months)
        )

        if 'Все' not in selected_departments and selected_departments:
            mask &= df['department'].isin(selected_departments)
        if 'Все' not in selected_groups and selected_groups:
            mask &= df['group'].isin(selected_groups)
        if 'Все' not in selected_subgroups and selected_subgroups:
            mask &= df['subgroup'].isin(selected_subgroups)
        if 'Все' not in selected_cities and selected_cities:
            mask &= df['city'].isin(selected_cities)
        if 'Все' not in selected_branches and selected_branches:
            mask &= df['branch'].isin(selected_branches)
        if 'Все' not in selected_formats and selected_formats:
            mask &= df['format'].isin(selected_formats)
        if specific_stores:
            # Приводим к строке для сравнения
            df['_store_str'] = df['store_code'].astype(str)
            mask &= df['_store_str'].isin(specific_stores)
            df = df.drop(columns=['_store_str'])

        filtered_df = df[mask].copy()

        if len(filtered_df) == 0:
            st.error("Нет данных с выбранными фильтрами. Ослабь критерии.")
            return

        # LFL
        if lfl_mode:
            entity_col = 'store_code' if analysis_level == 'Магазин' else analysis_level.lower()
            col_map = {'Магазин': 'store_code', 'Формат': 'format',
                       'Город': 'city', 'Филиал': 'branch'}
            lfl_col = col_map.get(analysis_level, 'store_code')
            if lfl_col in filtered_df.columns:
                before_lfl = len(filtered_df)
                filtered_df = filter_lfl(filtered_df, lfl_col)
                st.sidebar.info(f"LFL: {filtered_df[lfl_col].nunique()} из "
                               f"{df[lfl_col].nunique()} {analysis_level.lower()}")

        # Запуск аналитики
        with st.spinner("Считаю baselines..."):
            result_df = compute_baselines(filtered_df, analysis_level, yoy_weight)

        with st.spinner("Ищу аномалии..."):
            result_df = compute_anomalies(result_df, z_threshold)

        with st.spinner("Определяю причины..."):
            anom_mask = result_df['severity'] >= min_severity
            result_df.loc[anom_mask, 'drivers'] = result_df[anom_mask].apply(
                attribute_drivers, axis=1
            )
            result_df.loc[~anom_mask, 'drivers'] = ''
            result_df = compute_scope(result_df)

        # Сохраняем в session_state
        st.session_state['results'] = result_df
        st.session_state['anomalies'] = result_df[anom_mask].copy()
        st.session_state['analysis_level'] = analysis_level
        st.session_state['min_severity'] = min_severity
        st.session_state['full_df'] = filtered_df

    # ═══════════════════════════════════════════════════════════════
    # ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
    # ═══════════════════════════════════════════════════════════════
    if 'results' not in st.session_state:
        return

    result_df = st.session_state['results']
    anomalies = st.session_state['anomalies']
    analysis_level = st.session_state['analysis_level']

    # ── SUMMARY METRICS ──
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Всего строк", f"{len(result_df):,}")
    with col2:
        st.metric("Аномалий", f"{len(anomalies):,}",
                  delta=f"{len(anomalies)/max(len(result_df),1)*100:.1f}%")
    with col3:
        high_sev = (anomalies['severity'] >= 3).sum() if len(anomalies) > 0 else 0
        st.metric("Severity ≥ 3", f"{high_sev:,}")
    with col4:
        market_scope = (anomalies['scope'] == 'Market').sum() if len(anomalies) > 0 else 0
        st.metric("Market-wide", f"{market_scope:,}")
    with col5:
        oos_driven = anomalies['drivers'].str.contains('OOS', na=False).sum() \
                     if len(anomalies) > 0 else 0
        st.metric("OOS-driven", f"{oos_driven:,}")

    st.markdown("---")

    # ── ТАБЫ: РАДАР / DRILL-DOWN ──
    tab_radar, tab_drill, tab_overview = st.tabs(
        ["📡 Радар — Все аномалии", "🔬 Drill-down — Детали", "📈 Overview — Обзор"]
    )

    # ─────────────────────────────────────────────────────────
    # ТАБ 1: РАДАР
    # ─────────────────────────────────────────────────────────
    with tab_radar:
        if len(anomalies) == 0:
            st.success("Аномалий не найдено с текущими параметрами.")
        else:
            # Фильтры внутри радара
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                radar_scope = st.selectbox("Скоуп", ['Все', 'Local', 'Systemic', 'Market'],
                                            key='radar_scope')
            with rc2:
                all_driver_list = set()
                for d in anomalies['drivers'].dropna():
                    all_driver_list.update(d.split(', '))
                radar_driver = st.selectbox("Драйвер",
                                             ['Все'] + sorted(all_driver_list),
                                             key='radar_driver')
            with rc3:
                radar_direction = st.selectbox("Направление",
                                                ['Все', '📉 Падение', '📈 Рост'],
                                                key='radar_dir')
            with rc4:
                radar_top_n = st.number_input("Показать строк", 10, 500, 100, 10,
                                               key='radar_topn')

            display_anom = anomalies.copy()

            if radar_scope != 'Все':
                display_anom = display_anom[display_anom['scope'] == radar_scope]
            if radar_driver != 'Все':
                display_anom = display_anom[
                    display_anom['drivers'].str.contains(radar_driver, na=False)
                ]
            if radar_direction == '📉 Падение':
                display_anom = display_anom[display_anom.get('dev_sales_units', 0) < 0]
            elif radar_direction == '📈 Рост':
                display_anom = display_anom[display_anom.get('dev_sales_units', 0) > 0]

            display_anom = display_anom.sort_values('severity', ascending=False) \
                                       .head(radar_top_n)

            # Определяем колонки для отображения
            id_cols = ['period']
            if analysis_level == 'Магазин':
                id_cols += ['store_code', 'format', 'city']
            elif analysis_level == 'Формат':
                id_cols += ['format']
            elif analysis_level == 'Город':
                id_cols += ['city']
            elif analysis_level == 'Филиал':
                id_cols += ['branch']

            show_cols = id_cols + ['department', 'group', 'subgroup',
                                   'severity', 'drivers', 'scope']

            metric_display = {
                'sales_units': 'Шт',
                'dev_sales_units': '∆ Шт%',
                'sales_retail': 'Выручка',
                'dev_sales_retail': '∆ Выр%',
                'checks': 'Чеки',
                'sku_count': 'SKU',
                'oos': '%OOS',
                'promo_share_pc': 'Промо%'
            }

            for col in metric_display:
                if col in display_anom.columns:
                    show_cols.append(col)

            show_cols = [c for c in show_cols if c in display_anom.columns]
            table_df = display_anom[show_cols].copy()

            # Форматируем
            for dev_col in ['dev_sales_units', 'dev_sales_retail']:
                if dev_col in table_df.columns:
                    table_df[dev_col] = (table_df[dev_col] * 100).round(1)

            rename_map = {c: metric_display.get(c, c) for c in table_df.columns}
            table_df = table_df.rename(columns=rename_map)

            st.dataframe(table_df.reset_index(drop=True),
                        use_container_width=True, height=500)

            # Визуализации
            viz1, viz2 = st.columns(2)
            with viz1:
                # Drivers distribution
                all_d = []
                for d in display_anom['drivers'].dropna():
                    all_d.extend(d.split(', '))
                if all_d:
                    d_counts = pd.Series(all_d).value_counts().reset_index()
                    d_counts.columns = ['Драйвер', 'Количество']
                    fig_d = px.pie(d_counts, values='Количество', names='Драйвер',
                                  title='Распределение драйверов', hole=0.4,
                                  color_discrete_sequence=px.colors.qualitative.Set2)
                    fig_d.update_layout(height=350)
                    st.plotly_chart(fig_d, use_container_width=True)

            with viz2:
                # Scope distribution
                s_counts = display_anom['scope'].value_counts().reset_index()
                s_counts.columns = ['Скоуп', 'Количество']
                color_map = {'Local': '#3498db', 'Systemic': '#f39c12', 'Market': '#e74c3c'}
                fig_s = px.pie(s_counts, values='Количество', names='Скоуп',
                              title='Скоуп аномалий', hole=0.4,
                              color='Скоуп', color_discrete_map=color_map)
                fig_s.update_layout(height=350)
                st.plotly_chart(fig_s, use_container_width=True)

            # Heatmap: аномалии по подгруппам и периодам
            if 'period' in anomalies.columns and 'subgroup' in anomalies.columns:
                heatmap_data = anomalies.groupby(['subgroup', 'period'])['severity'].mean() \
                                        .reset_index()
                if len(heatmap_data) > 0:
                    top_subgroups = anomalies.groupby('subgroup')['severity'].sum() \
                                            .nlargest(20).index
                    heatmap_data = heatmap_data[heatmap_data['subgroup'].isin(top_subgroups)]
                    heatmap_pivot = heatmap_data.pivot_table(
                        index='subgroup', columns='period', values='severity', fill_value=0
                    )
                    fig_heat = px.imshow(
                        heatmap_pivot, aspect='auto',
                        title='Тепловая карта: Severity по подгруппам и периодам (топ-20)',
                        color_continuous_scale='YlOrRd',
                        labels=dict(color="Severity")
                    )
                    fig_heat.update_layout(height=500)
                    st.plotly_chart(fig_heat, use_container_width=True)

    # ─────────────────────────────────────────────────────────
    # ТАБ 2: DRILL-DOWN
    # ─────────────────────────────────────────────────────────
    with tab_drill:
        st.subheader("Выбери объект для детального анализа")

        dd1, dd2 = st.columns(2)

        if analysis_level == 'Магазин':
            with dd1:
                entity_options = sorted(result_df['store_code'].unique())
                selected_entity = st.selectbox("Магазин", entity_options, key='dd_entity')
                entity_col = 'store_code'
        elif analysis_level == 'Формат':
            with dd1:
                entity_options = sorted(result_df['format'].dropna().unique())
                selected_entity = st.selectbox("Формат", entity_options, key='dd_entity')
                entity_col = 'format'
        elif analysis_level == 'Город':
            with dd1:
                entity_options = sorted(result_df['city'].dropna().unique())
                selected_entity = st.selectbox("Город", entity_options, key='dd_entity')
                entity_col = 'city'
        else:
            with dd1:
                entity_options = sorted(result_df['branch'].dropna().unique())
                selected_entity = st.selectbox("Филиал", entity_options, key='dd_entity')
                entity_col = 'branch'

        with dd2:
            sg_options = sorted(
                result_df[result_df[entity_col] == selected_entity]['subgroup'].unique()
            )
            selected_sg = st.selectbox("Подгруппа", sg_options, key='dd_subgroup')

        # Данные для drill-down
        dd_data = result_df[
            (result_df[entity_col] == selected_entity) &
            (result_df['subgroup'] == selected_sg)
        ].sort_values('period').copy()

        if len(dd_data) == 0:
            st.warning("Нет данных для этой комбинации")
        else:
            # Инфо-карточка
            last = dd_data.iloc[-1]
            ic1, ic2, ic3, ic4 = st.columns(4)
            with ic1:
                n_anom = (dd_data['severity'] > 0).sum()
                st.metric("Аномалий", f"{n_anom} / {len(dd_data)}")
            with ic2:
                latest_dev = last.get('dev_sales_units', 0) or 0
                st.metric("Посл. ∆ продаж",
                         f"{latest_dev*100:.1f}%",
                         delta=f"{'падение' if latest_dev < 0 else 'рост'}")
            with ic3:
                st.metric("%OOS (посл.)", f"{last.get('oos', 0):.1f}%")
            with ic4:
                st.metric("Промо% (посл.)", f"{last.get('promo_share_pc', 0):.1f}%")

            if n_anom > 0:
                latest_anom = dd_data[dd_data['severity'] > 0].iloc[-1]
                st.info(f"**Последняя аномалия:** {latest_anom['period']} | "
                       f"Severity: {latest_anom['severity']} | "
                       f"Драйверы: {latest_anom.get('drivers', 'N/A')} | "
                       f"Скоуп: {latest_anom.get('scope', 'N/A')}")

            # Графики: Факт vs Baseline
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Продажи (шт)', 'Выручка (ПЦ)', 'Чеки', 'SKU']
            )

            plot_cfg = [
                ('sales_units', 1, 1), ('sales_retail', 1, 2),
                ('checks', 2, 1), ('sku_count', 2, 2),
            ]

            for metric, row, col in plot_cfg:
                if metric not in dd_data.columns:
                    continue
                fig.add_trace(go.Scatter(
                    x=dd_data['period'], y=dd_data[metric],
                    mode='lines+markers', name='Факт',
                    line=dict(color='#636EFA', width=2),
                    showlegend=(row == 1 and col == 1)
                ), row=row, col=col)

                bl_col = f'baseline_{metric}'
                if bl_col in dd_data.columns:
                    fig.add_trace(go.Scatter(
                        x=dd_data['period'], y=dd_data[bl_col],
                        mode='lines', name='Baseline',
                        line=dict(color='#FFA15A', width=2, dash='dash'),
                        showlegend=(row == 1 and col == 1)
                    ), row=row, col=col)

                anom_pts = dd_data[dd_data['severity'] > 0]
                if len(anom_pts) > 0 and metric in anom_pts.columns:
                    fig.add_trace(go.Scatter(
                        x=anom_pts['period'], y=anom_pts[metric],
                        mode='markers', name='Аномалия',
                        marker=dict(color='red', size=10, symbol='x'),
                        showlegend=(row == 1 and col == 1)
                    ), row=row, col=col)

            fig.update_layout(height=500, title_text='Факт vs Baseline')
            st.plotly_chart(fig, use_container_width=True)

            # Доп. метрики
            fig2 = make_subplots(
                rows=1, cols=3,
                subplot_titles=['%OOS', 'Доля промо (ПЦ)', 'Продажи/чек']
            )

            if 'oos' in dd_data.columns:
                fig2.add_trace(go.Bar(
                    x=dd_data['period'], y=dd_data['oos'],
                    marker_color='#EF553B', showlegend=False
                ), row=1, col=1)

            if 'promo_share_pc' in dd_data.columns:
                fig2.add_trace(go.Bar(
                    x=dd_data['period'], y=dd_data['promo_share_pc'],
                    marker_color='#AB63FA', showlegend=False
                ), row=1, col=2)

            if 'units_per_check' in dd_data.columns:
                fig2.add_trace(go.Scatter(
                    x=dd_data['period'], y=dd_data['units_per_check'],
                    mode='lines+markers', marker_color='#00CC96', showlegend=False
                ), row=1, col=3)

            fig2.update_layout(height=300, title_text='Дополнительные метрики')
            st.plotly_chart(fig2, use_container_width=True)

            # Peer comparison
            if analysis_level == 'Магазин' and 'format' in last.index and 'city' in last.index:
                peer_mask = (
                    (result_df['format'] == last['format']) &
                    (result_df['city'] == last['city']) &
                    (result_df['subgroup'] == selected_sg) &
                    (result_df['year'] == last['year']) &
                    (result_df['month'] == last['month']) &
                    (result_df['store_code'] != selected_entity)
                )
                peers = result_df[peer_mask]

                if len(peers) > 0:
                    st.subheader(f"📊 Сравнение с peers ({len(peers)} магазинов)")

                    comp_metrics = ['sales_units', 'sales_retail', 'checks',
                                   'sku_count', 'oos', 'promo_share_pc']
                    comp_names = ['Продажи шт', 'Выручка ПЦ', 'Чеки',
                                 'SKU', '%OOS', 'Промо%']

                    comp_data = []
                    for m, name in zip(comp_metrics, comp_names):
                        if m in last.index and m in peers.columns:
                            val = last[m]
                            med = peers[m].median()
                            pct = ((val - med) / med * 100) if med != 0 else 0
                            comp_data.append({
                                'Метрика': name,
                                'Этот объект': round(val, 1),
                                'Медиана peers': round(med, 1),
                                'vs Медиана': f"{pct:+.1f}%"
                            })

                    if comp_data:
                        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

    # ─────────────────────────────────────────────────────────
    # ТАБ 3: OVERVIEW
    # ─────────────────────────────────────────────────────────
    with tab_overview:
        st.subheader("Общая картина по выборке")

        # Тренды ключевых метрик
        trend_data = result_df.groupby('period').agg({
            'sales_units': 'sum',
            'sales_retail': 'sum',
            'checks': 'sum',
            'sku_count': 'mean',
            'severity': lambda x: (x > 0).sum()
        }).reset_index()
        trend_data = trend_data.rename(columns={'severity': 'anomaly_count'})

        fig_trend = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Продажи шт (сумма)', 'Выручка ПЦ (сумма)',
                           'Чеки (сумма)', 'Кол-во аномалий']
        )

        fig_trend.add_trace(go.Scatter(
            x=trend_data['period'], y=trend_data['sales_units'],
            mode='lines+markers', line=dict(color='#636EFA')
        ), row=1, col=1)

        fig_trend.add_trace(go.Scatter(
            x=trend_data['period'], y=trend_data['sales_retail'],
            mode='lines+markers', line=dict(color='#00CC96')
        ), row=1, col=2)

        fig_trend.add_trace(go.Scatter(
            x=trend_data['period'], y=trend_data['checks'],
            mode='lines+markers', line=dict(color='#AB63FA')
        ), row=2, col=1)

        fig_trend.add_trace(go.Bar(
            x=trend_data['period'], y=trend_data['anomaly_count'],
            marker_color='#EF553B'
        ), row=2, col=2)

        fig_trend.update_layout(height=500, showlegend=False,
                                title_text='Тренды по периодам')
        st.plotly_chart(fig_trend, use_container_width=True)

        # Топ аномальных подгрупп
        if len(anomalies) > 0:
            top_sg = anomalies.groupby('subgroup').agg(
                anomaly_count=('severity', 'count'),
                avg_severity=('severity', 'mean'),
                main_driver=('drivers', lambda x: pd.Series(
                    [d for ds in x.dropna() for d in ds.split(', ')]
                ).mode().iloc[0] if len(x.dropna()) > 0 else 'N/A')
            ).sort_values('anomaly_count', ascending=False).head(15).reset_index()

            st.subheader("Топ-15 аномальных подгрупп")
            st.dataframe(top_sg, use_container_width=True)

    # ── ЭКСПОРТ ──
    st.markdown("---")
    if len(anomalies) > 0:
        export_cols = [c for c in anomalies.columns
                       if not c.startswith('yoy_') and not c.startswith('peer_')
                       and not c.startswith('z_') and not c.startswith('flag_')]
        csv_data = anomalies[export_cols].to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "💾 Скачать аномалии (CSV)",
            data=csv_data,
            file_name="anomalies_export.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()

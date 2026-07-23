"""Deterministic market-data, ranking-model, and score-fusion stage for Auto Fin."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl

from .....components import R
from .....components.outbound_proxy import BaseOutboundProxy
from .....enumeration import ComponentEnum
from .....schema import (
    DimensionRanking,
    EtfScore,
    EventAnalysisOutput,
    ExtremeAnalysis,
    FusionRanking,
    RankingMetrics,
)
from .....utils.tushare import create_tushare_api
from ....base_step import Ref
from .._common import write_atomic
from ._base import AutoFinAnalysisStep

_DIMENSIONS = ("event", "backtest", "us_correlation")
_US_CODES = ("NVDA", "MU", "WDC", "STX", "SOXX", "SOXL", "QQQ", "SPY")
_EXCLUDED_ETF_WORDS = (
    "债",
    "货币",
    "黄金",
    "白银",
    "商品",
    "原油",
    "纳指",
    "标普",
    "恒生",
    "日经",
    "德国",
    "法国",
    "美国",
    "港股",
)


def _stock_ts_code(value: str) -> str:
    """Normalize a six-digit A-share holding code for TuShare daily queries."""
    code = value.strip()
    if "." in code or len(code) != 6:
        return code
    return f"{code}.SH" if code[0] in {"5", "6", "9"} else f"{code}.SZ"


def _records_to_frame(value: Any) -> pl.DataFrame:
    """Normalize injected records, pandas frames, and Polars frames."""
    if isinstance(value, pl.DataFrame):
        return value
    if value is None:
        return pl.DataFrame()
    if hasattr(value, "to_dict"):
        try:
            records = value.to_dict(orient="records")
            return pl.from_dicts(records, infer_schema_length=None) if records else pl.DataFrame()
        except TypeError:
            pass
    records = list(value)
    return pl.from_dicts(records, infer_schema_length=None) if records else pl.DataFrame()


def _date_expr(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.String).str.replace_all("-", "").str.strptime(pl.Date, "%Y%m%d", strict=False)


def _rank(values: np.ndarray) -> np.ndarray:
    """Average-free deterministic ranks; sufficient for cross-sectional diagnostics."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.nanstd(left) == 0 or np.nanstd(right) == 0:
        return math.nan
    return float(np.corrcoef(_rank(left), _rank(right))[0, 1])


def _ndcg(actual: np.ndarray, predicted: np.ndarray, k: int = 20) -> float:
    if len(actual) < 2:
        return math.nan
    relevance = _rank(actual) / max(len(actual) - 1, 1)
    predicted_order = np.argsort(-predicted, kind="stable")[:k]
    ideal_order = np.argsort(-actual, kind="stable")[:k]
    discounts = np.log2(np.arange(2, len(predicted_order) + 2))
    dcg = float(np.sum((2.0 ** relevance[predicted_order] - 1.0) / discounts))
    idcg = float(np.sum((2.0 ** relevance[ideal_order] - 1.0) / discounts))
    return dcg / idcg if idcg > 0 else math.nan


@dataclass
class _TreeNode:
    value: float
    feature: int = -1
    threshold: float = 0.0
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None


class _ExtraTreesRegressor:
    """Small dependency-free extremely-randomized tree ensemble for ETF ranking."""

    def __init__(self, *, tree_count: int = 31, max_depth: int = 5, min_leaf: int = 20, seed: int = 42):
        self.tree_count = tree_count
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.seed = seed
        self._medians: np.ndarray | None = None
        self._trees: list[_TreeNode] = []

    def _build(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray, depth: int, rng) -> _TreeNode:
        node = _TreeNode(value=float(np.mean(y[indices])))
        if depth >= self.max_depth or len(indices) < self.min_leaf * 2 or np.std(y[indices]) < 1e-12:
            return node
        feature_count = max(1, int(math.sqrt(x.shape[1])))
        features = rng.choice(x.shape[1], size=feature_count, replace=False)
        best: tuple[float, int, float, np.ndarray] | None = None
        best_gain = -math.inf
        parent_error = float(np.var(y[indices]) * len(indices))
        for feature in features:
            values = x[indices, feature]
            low, high = np.quantile(values, [0.1, 0.9])
            if not np.isfinite(low + high) or low >= high:
                continue
            for threshold in rng.uniform(low, high, size=4):
                mask = values <= threshold
                left_count = int(mask.sum())
                if left_count < self.min_leaf or len(indices) - left_count < self.min_leaf:
                    continue
                error = float(
                    np.var(y[indices[mask]]) * left_count + np.var(y[indices[~mask]]) * (len(indices) - left_count),
                )
                gain = parent_error - error
                if gain > best_gain:
                    best_gain = gain
                    best = (gain, int(feature), float(threshold), mask)
        if best is None:
            return node
        gain, node.feature, node.threshold, mask = best
        if gain <= 0:
            return node
        node.left = self._build(x, y, indices[mask], depth + 1, rng)
        node.right = self._build(x, y, indices[~mask], depth + 1, rng)
        return node

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_ExtraTreesRegressor":
        """Fit bootstrapped randomized regression trees."""
        if len(x) != len(y) or len(x) < self.min_leaf * 2:
            raise ValueError("insufficient samples for tree ensemble")
        self._medians = np.nanmedian(x, axis=0)
        clean = np.where(np.isfinite(x), x, self._medians)
        rng = np.random.default_rng(self.seed)
        self._trees = []
        for _ in range(self.tree_count):
            sample = rng.choice(len(clean), size=len(clean), replace=True)
            self._trees.append(self._build(clean, y, sample, 0, rng))
        return self

    @staticmethod
    def _predict_tree(tree: _TreeNode, row: np.ndarray) -> float:
        node = tree
        while node.feature >= 0:
            child = node.left if row[node.feature] <= node.threshold else node.right
            if child is None:
                break
            node = child
        return node.value

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return the mean prediction across fitted trees."""
        if self._medians is None or not self._trees:
            raise RuntimeError("tree ensemble is not fitted")
        clean = np.where(np.isfinite(x), x, self._medians)
        return np.mean(
            [[self._predict_tree(tree, row) for tree in self._trees] for row in clean],
            axis=1,
        )


class TushareResearchClient:
    """Concurrent, cutoff-bounded TuShare adapter returning Polars frames."""

    def __init__(self, token: str, *, concurrency: int = 6, proxy_url: str | None = None):
        self._pro = create_tushare_api(token, proxy_url=proxy_url)
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self.sources: list[dict[str, Any]] = []

    async def _call(self, endpoint: str, **kwargs) -> pl.DataFrame:
        started = datetime.now().astimezone()
        query = json.dumps(kwargs, sort_keys=True, default=str)
        async with self._semaphore:
            method: Callable[..., Any] = getattr(self._pro, endpoint)
            value = await asyncio.to_thread(method, **kwargs)
        frame = _records_to_frame(value)
        self.sources.append(
            {
                "endpoint": endpoint,
                "query_hash": hashlib.sha256(query.encode()).hexdigest(),
                "request_started_at": started.isoformat(),
                "fetched_at": datetime.now().astimezone().isoformat(),
                "row_count": frame.height,
            },
        )
        return frame

    async def _optional_call(self, endpoint: str, **kwargs) -> pl.DataFrame:
        """Return an empty frame for a non-critical endpoint and retain its warning."""
        try:
            return await self._call(endpoint, **kwargs)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            query = json.dumps(kwargs, sort_keys=True, default=str)
            self.sources.append(
                {
                    "endpoint": endpoint,
                    "query_hash": hashlib.sha256(query.encode()).hexdigest(),
                    "fetched_at": datetime.now().astimezone().isoformat(),
                    "row_count": 0,
                    "warnings": [f"{type(exc).__name__}: {exc}"],
                },
            )
            return pl.DataFrame()

    async def fetch_bundle(
        self,
        *,
        history_end: date,
        us_history_end: date | None = None,
        lookback_days: int,
        preselect_size: int,
        us_codes: tuple[str, ...] = _US_CODES,
        fetch_us: bool = True,
    ) -> dict[str, pl.DataFrame]:
        """Fetch universe, daily bars, holdings, valuations, and US bars."""
        end_text = history_end.strftime("%Y%m%d")
        start_text = (history_end - timedelta(days=lookback_days)).strftime("%Y%m%d")
        us_end = us_history_end or history_end
        us_end_text = us_end.strftime("%Y%m%d")
        us_start_text = (us_end - timedelta(days=lookback_days)).strftime("%Y%m%d")
        basic, latest = await asyncio.gather(
            self._optional_call(
                "fund_basic",
                market="E",
                status="L",
                fields="ts_code,name,fund_type,market,status",
            ),
            self._call(
                "fund_daily",
                trade_date=end_text,
                fields="ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount",
            ),
        )
        if latest.is_empty() or "ts_code" not in latest.columns:
            raise RuntimeError(f"TuShare fund_daily returned no ETF rows for {history_end}")
        selected = (
            latest.with_columns(pl.col("amount").cast(pl.Float64, strict=False).fill_null(0.0))
            .sort("amount", descending=True)
            .head(preselect_size)
            .get_column("ts_code")
            .cast(pl.String)
            .to_list()
        )
        daily_frames, fund_adj_frames, us_frames = await asyncio.gather(
            asyncio.gather(
                *[
                    self._optional_call(
                        "fund_daily",
                        ts_code=code,
                        start_date=start_text,
                        end_date=end_text,
                        fields="ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount",
                    )
                    for code in selected
                ],
            ),
            asyncio.gather(
                *[
                    self._optional_call(
                        "fund_adj",
                        ts_code=code,
                        start_date=start_text,
                        end_date=end_text,
                        fields="ts_code,trade_date,adj_factor",
                    )
                    for code in selected
                ],
            ),
            asyncio.gather(
                *(
                    [
                        self._optional_call(
                            "us_daily_adj",
                            ts_code=code,
                            start_date=us_start_text,
                            end_date=us_end_text,
                            fields="ts_code,trade_date,open,high,low,close,pre_close,pct_change,vol,amount,adj_factor",
                        )
                        for code in us_codes
                    ]
                    if fetch_us
                    else []
                ),
            ),
        )
        daily = pl.concat([frame for frame in daily_frames if not frame.is_empty()], how="diagonal_relaxed")
        fund_adj = (
            pl.concat([frame for frame in fund_adj_frames if not frame.is_empty()], how="diagonal_relaxed")
            if any(not frame.is_empty() for frame in fund_adj_frames)
            else pl.DataFrame()
        )
        us_daily = (
            pl.concat([frame for frame in us_frames if not frame.is_empty()], how="diagonal_relaxed")
            if any(not frame.is_empty() for frame in us_frames)
            else pl.DataFrame()
        )
        # Holdings are only used as confirmation. Limit them to the liquid shortlist.
        holding_frames = await asyncio.gather(
            *[
                self._optional_call(
                    "fund_portfolio",
                    ts_code=code,
                    start_date=(history_end - timedelta(days=550)).strftime("%Y%m%d"),
                    end_date=end_text,
                    fields="ts_code,ann_date,end_date,symbol,mkv,amount,stk_mkv_ratio",
                )
                for code in selected[:20]
            ],
        )
        holdings = (
            pl.concat([frame for frame in holding_frames if not frame.is_empty()], how="diagonal_relaxed")
            if any(not frame.is_empty() for frame in holding_frames)
            else pl.DataFrame()
        )
        stock_codes: list[str] = []
        if not holdings.is_empty() and {"symbol", "stk_mkv_ratio"}.issubset(holdings.columns):
            stock_codes = [
                _stock_ts_code(value)
                for value in (
                    holdings.with_columns(pl.col("stk_mkv_ratio").cast(pl.Float64, strict=False))
                    .sort(["ts_code", "end_date", "stk_mkv_ratio"], descending=[False, True, True])
                    .group_by("ts_code", maintain_order=True)
                    .head(3)
                    .get_column("symbol")
                    .cast(pl.String)
                    .unique()
                    .to_list()
                )
            ]
        stock_daily_frames, stock_adj_frames = await asyncio.gather(
            asyncio.gather(
                *[
                    self._optional_call(
                        "daily",
                        ts_code=code,
                        start_date=(history_end - timedelta(days=60)).strftime("%Y%m%d"),
                        end_date=end_text,
                        fields="ts_code,trade_date,close,pct_chg,amount",
                    )
                    for code in stock_codes
                ],
            ),
            asyncio.gather(
                *[
                    self._optional_call(
                        "adj_factor",
                        ts_code=code,
                        start_date=(history_end - timedelta(days=60)).strftime("%Y%m%d"),
                        end_date=end_text,
                        fields="ts_code,trade_date,adj_factor",
                    )
                    for code in stock_codes
                ],
            ),
        )
        valuation = await self._optional_call(
            "daily_basic",
            trade_date=end_text,
            fields="ts_code,trade_date,turnover_rate,pe_ttm,pb,total_mv",
        )
        stock_daily = (
            pl.concat([frame for frame in stock_daily_frames if not frame.is_empty()], how="diagonal_relaxed")
            if any(not frame.is_empty() for frame in stock_daily_frames)
            else pl.DataFrame()
        )
        stock_adj = (
            pl.concat([frame for frame in stock_adj_frames if not frame.is_empty()], how="diagonal_relaxed")
            if any(not frame.is_empty() for frame in stock_adj_frames)
            else pl.DataFrame()
        )
        return {
            "universe": basic,
            "etf_daily": daily,
            "fund_adj": fund_adj,
            "holdings": holdings,
            "stock_daily": stock_daily,
            "stock_adj": stock_adj,
            "stock_valuation": valuation,
            "us_daily": us_daily,
        }


class AutoFinQuantResearch:
    """Pure Polars/NumPy research engine used by the pipeline and unit tests."""

    feature_columns = ("ret_1", "ret_5", "ret_20", "intraday", "volatility_20", "amount_ratio")

    def __init__(self, *, tree_count: int = 31, seed: int = 42):
        self.tree_count = tree_count
        self.seed = seed

    @staticmethod
    def _adjust_prices(
        daily: pl.DataFrame,
        factors: pl.DataFrame,
        *,
        price_columns: tuple[str, ...],
        label: str,
    ) -> pl.DataFrame:
        """Apply required cumulative adjustment factors without a raw-price fallback."""
        if daily.is_empty():
            return daily
        required_daily = {"ts_code", "trade_date", *price_columns}
        if not required_daily.issubset(daily.columns):
            raise ValueError(f"{label} data is missing columns: {sorted(required_daily - set(daily.columns))}")
        required_factors = {"ts_code", "trade_date", "adj_factor"}
        if factors.is_empty() or not required_factors.issubset(factors.columns):
            raise ValueError(f"{label} adjustment factors are required")
        normalized_daily = daily.with_columns(
            pl.col("ts_code").cast(pl.String),
            _date_expr("trade_date").alias("trade_date"),
        )
        normalized_factors = (
            factors.with_columns(
                pl.col("ts_code").cast(pl.String),
                _date_expr("trade_date").alias("trade_date"),
                pl.col("adj_factor").cast(pl.Float64, strict=False),
            )
            .filter(pl.col("adj_factor").is_not_null() & (pl.col("adj_factor") > 0))
            .unique(["ts_code", "trade_date"], keep="last")
        )
        adjusted = normalized_daily.join(
            normalized_factors.select("ts_code", "trade_date", "adj_factor"),
            on=["ts_code", "trade_date"],
            how="left",
        )
        missing = adjusted.filter(pl.col("adj_factor").is_null()).height
        if missing:
            raise ValueError(f"{label} adjustment factors are incomplete: missing {missing} rows")
        return adjusted.with_columns(
            *[
                (pl.col(column).cast(pl.Float64, strict=False) * pl.col("adj_factor")).alias(column)
                for column in price_columns
            ],
        )

    @staticmethod
    def _prepare_universe(bundle: dict[str, pl.DataFrame], universe_size: int) -> tuple[pl.DataFrame, pl.DataFrame]:
        daily = AutoFinQuantResearch._adjust_prices(
            bundle.get("etf_daily", pl.DataFrame()),
            bundle.get("fund_adj", pl.DataFrame()),
            price_columns=tuple(
                column
                for column in ("open", "high", "low", "close", "pre_close")
                if column in bundle.get("etf_daily", pl.DataFrame()).columns
            ),
            label="ETF",
        )
        if daily.is_empty():
            return pl.DataFrame(), pl.DataFrame()
        required = {"ts_code", "trade_date", "open", "close", "amount"}
        if not required.issubset(daily.columns):
            raise ValueError(f"ETF daily data is missing columns: {sorted(required - set(daily.columns))}")
        daily = (
            daily.with_columns(
                pl.col("ts_code").cast(pl.String),
                *[
                    pl.col(column).cast(pl.Float64, strict=False)
                    for column in ("open", "high", "low", "close", "pre_close", "vol", "amount")
                    if column in daily.columns
                ],
            )
            .filter(pl.col("trade_date").is_not_null() & (pl.col("open") > 0) & (pl.col("close") > 0))
            .sort(["ts_code", "trade_date"])
        )
        basic = bundle.get("universe", pl.DataFrame())
        name_column = "name" if "name" in basic.columns else "fund_name" if "fund_name" in basic.columns else None
        if not basic.is_empty() and name_column:
            names = basic.select(
                pl.col("ts_code").cast(pl.String),
                pl.col(name_column).cast(pl.String).fill_null("").alias("name"),
            )
        else:
            names = daily.select("ts_code").unique().with_columns(pl.lit("").alias("name"))
        excluded = "|".join(_EXCLUDED_ETF_WORDS)
        names = names.filter(~pl.col("name").str.contains(excluded))
        liquidity = (
            daily.join(names, on="ts_code", how="inner")
            .group_by(["ts_code", "name"])
            .agg(pl.col("amount").tail(20).mean().alias("average_amount"))
            .sort("average_amount", descending=True)
            .head(universe_size)
        )
        return daily.join(liquidity.select("ts_code"), on="ts_code", how="inner"), liquidity

    @staticmethod
    def _features(daily: pl.DataFrame) -> pl.DataFrame:
        return daily.sort(["ts_code", "trade_date"]).with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0).over("ts_code").alias("ret_1"),
            (pl.col("close") / pl.col("close").shift(5) - 1.0).over("ts_code").alias("ret_5"),
            (pl.col("close") / pl.col("close").shift(20) - 1.0).over("ts_code").alias("ret_20"),
            (pl.col("close") / pl.col("open") - 1.0).alias("intraday"),
            ((pl.col("close") / pl.col("close").shift(1) - 1.0).rolling_std(20).over("ts_code")).alias("volatility_20"),
            (pl.col("amount") / pl.col("amount").rolling_mean(20).over("ts_code")).alias("amount_ratio"),
            (pl.col("close").shift(-1) / pl.col("close") - 1.0).over("ts_code").alias("future_return"),
            (pl.col("open").shift(-2) / pl.col("open").shift(-1) - 1.0).over("ts_code").alias("future_open_return"),
            pl.col("trade_date").shift(-1).over("ts_code").alias("prediction_date"),
        )

    def _fit_ranker(
        self,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str],
        label_column: str,
        names: dict[str, str],
        prediction_frame: pl.DataFrame | None = None,
    ) -> tuple[list[EtfScore], RankingMetrics, np.ndarray, pl.DataFrame]:
        available = frame.drop_nulls(feature_columns + [label_column])
        dates = sorted(available.get_column("trade_date").unique().to_list())
        if len(dates) < 40 or available.height < 500:
            raise ValueError(
                "at least 40 dates and 500 ETF-day samples are required "
                f"(got dates={len(dates)}, samples={available.height})",
            )
        split = dates[max(1, int(len(dates) * 0.75))]
        train = available.filter(pl.col("trade_date") < split)
        validation = available.filter(pl.col("trade_date") >= split)
        x_train = train.select(feature_columns).to_numpy()
        y_train = train.get_column(label_column).to_numpy()
        model = _ExtraTreesRegressor(tree_count=self.tree_count, seed=self.seed).fit(x_train, y_train)
        predictions = model.predict(validation.select(feature_columns).to_numpy())
        validation = validation.with_columns(pl.Series("prediction", predictions))
        daily_ic: list[float] = []
        daily_ndcg: list[float] = []
        for group in validation.partition_by("trade_date", maintain_order=True):
            actual = group.get_column(label_column).to_numpy()
            predicted = group.get_column("prediction").to_numpy()
            ic = _spearman(actual, predicted)
            ndcg = _ndcg(actual, predicted)
            if np.isfinite(ic):
                daily_ic.append(ic)
            if np.isfinite(ndcg):
                daily_ndcg.append(ndcg)
        rank_ic = float(np.mean(daily_ic)) if daily_ic else None
        rank_ic_ir = (
            float(np.mean(daily_ic) / np.std(daily_ic, ddof=1))
            if len(daily_ic) > 1 and np.std(daily_ic, ddof=1) > 0
            else None
        )
        metrics = RankingMetrics(
            rank_ic=rank_ic,
            rank_ic_ir=rank_ic_ir,
            ndcg_at_20=float(np.mean(daily_ndcg)) if daily_ndcg else None,
            train_sample_count=train.height,
            validation_sample_count=validation.height,
            validation_date_count=len(daily_ic),
        )
        labelled = frame.drop_nulls(feature_columns + [label_column])
        final_model = _ExtraTreesRegressor(tree_count=self.tree_count, seed=self.seed).fit(
            labelled.select(feature_columns).to_numpy(),
            labelled.get_column(label_column).to_numpy(),
        )
        latest = (
            (prediction_frame if prediction_frame is not None else frame)
            .drop_nulls(feature_columns)
            .sort(["ts_code", "trade_date"])
            .group_by("ts_code", maintain_order=True)
            .tail(1)
        )
        if latest.is_empty():
            raise ValueError("no complete feature row is available for current ranking")
        latest_predictions = final_model.predict(latest.select(feature_columns).to_numpy())
        order = np.argsort(-latest_predictions, kind="stable")
        percentiles = _rank(latest_predictions) / max(len(latest_predictions) - 1, 1) * 100.0
        scores = [
            EtfScore(
                code=str(latest["ts_code"][int(index)]),
                name=names.get(str(latest["ts_code"][int(index)]), ""),
                rank=rank,
                score=float(percentiles[index]),
                expected_return=float(latest_predictions[int(index)]),
                confidence=float(min(1.0, max(0.0, 0.5 + (rank_ic or 0.0)))),
                reasons=["极端随机树横截面预测", "得分为当期预测收益的横截面百分位"],
            )
            for rank, index in enumerate(order[:20], 1)
        ]
        return scores, metrics, latest_predictions, latest

    @staticmethod
    def _extremes(features: pl.DataFrame, names: dict[str, str]) -> list[ExtremeAnalysis]:
        if features.is_empty():
            return []
        latest = features.sort(["ts_code", "trade_date"]).group_by("ts_code").tail(1)
        extremes: list[ExtremeAnalysis] = []
        for row in latest.filter(pl.col("ret_1").abs() >= 0.07).iter_rows(named=True):
            code = str(row["ts_code"])
            observed = float(row["ret_1"])
            history = features.filter((pl.col("ts_code") == code) & (pl.col("ret_1").abs() >= 0.07))
            future = history.get_column("future_return").drop_nulls().to_numpy()
            extremes.append(
                ExtremeAnalysis(
                    code=code,
                    direction="UP" if observed > 0 else "DOWN",
                    observed_return=observed,
                    threshold=0.07,
                    historical_sample_count=len(future),
                    next_return_mean=float(np.mean(future)) if len(future) else None,
                    next_return_hit_rate=float(np.mean(future > 0)) if len(future) else None,
                    conclusion=f"{names.get(code, code)} 单日涨跌超过 7%，单列尾部样本，不能按常态模型外推。",
                ),
            )
        return extremes[:20]

    @staticmethod
    def _event_ranking(
        event_output: EventAnalysisOutput,
        features: pl.DataFrame,
        liquidity: pl.DataFrame,
        bundle: dict[str, pl.DataFrame],
    ) -> DimensionRanking:
        as_of = event_output.window.end_inclusive
        if features.is_empty():
            return DimensionRanking(
                dimension="event",
                as_of=as_of,
                status="INSUFFICIENT_DATA",
                methodology="事件方向×置信度×未price-in比例，并以ETF近期行情确认",
                limitations=["没有可用 ETF 日线，无法生成事件 Top20。"],
            )
        latest = features.sort(["ts_code", "trade_date"]).group_by("ts_code").tail(1)
        names = dict(zip(liquidity["ts_code"].to_list(), liquidity["name"].to_list()))
        if not event_output.events:
            candidates = [
                EtfScore(
                    code=str(row["ts_code"]),
                    name=str(row["name"]),
                    rank=rank,
                    score=50.0,
                    confidence=0.0,
                    price_in=0.0,
                    reasons=["当前事件窗口无新增可验证事件，维持中性分"],
                )
                for rank, row in enumerate(liquidity.head(20).iter_rows(named=True), 1)
            ]
            return DimensionRanking(
                dimension="event",
                as_of=as_of,
                status="COMPLETE",
                methodology="无新增事件时对流动性ETF候选池给中性分；不制造方向信号",
                candidates=candidates,
            )
        holdings = bundle.get("holdings", pl.DataFrame())
        stock_daily = bundle.get("stock_daily", pl.DataFrame())
        if not stock_daily.is_empty():
            stock_daily = AutoFinQuantResearch._adjust_prices(
                stock_daily,
                bundle.get("stock_adj", pl.DataFrame()),
                price_columns=("close",),
                label="A-share constituent",
            )
        valuation = bundle.get("stock_valuation", pl.DataFrame())
        stock_returns: dict[str, float] = {}
        if not stock_daily.is_empty() and {"ts_code", "close", "trade_date"}.issubset(stock_daily.columns):
            stock_summary = (
                stock_daily.with_columns(
                    pl.col("ts_code").cast(pl.String).str.slice(0, 6).alias("stock_code"),
                    pl.col("close").cast(pl.Float64, strict=False),
                    _date_expr("trade_date").alias("trade_date"),
                )
                .drop_nulls(["close", "trade_date"])
                .sort(["stock_code", "trade_date"])
                .group_by("stock_code")
                .agg((pl.col("close").last() / pl.col("close").first() - 1.0).alias("return"))
            )
            stock_returns = dict(zip(stock_summary["stock_code"].to_list(), stock_summary["return"].to_list()))
        valuation_metrics: dict[str, tuple[float | None, float | None]] = {}
        if not valuation.is_empty() and "ts_code" in valuation.columns:
            normalized_valuation = valuation.with_columns(
                pl.col("ts_code").cast(pl.String).str.slice(0, 6).alias("stock_code"),
                pl.col("pe_ttm").cast(pl.Float64, strict=False),
                pl.col("pb").cast(pl.Float64, strict=False),
            )
            valuation_metrics = {
                str(row["stock_code"]): (row.get("pe_ttm"), row.get("pb"))
                for row in normalized_valuation.iter_rows(named=True)
            }
        top_holdings: dict[str, list[str]] = {}
        if not holdings.is_empty() and {"ts_code", "symbol", "stk_mkv_ratio"}.issubset(holdings.columns):
            normalized = holdings.with_columns(
                pl.col("ts_code").cast(pl.String),
                pl.col("symbol").cast(pl.String).str.slice(0, 6).alias("stock_code"),
                pl.col("stk_mkv_ratio").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("end_date").cast(pl.String).alias("end_date"),
            )
            for code_group in normalized.partition_by("ts_code"):
                code = str(code_group["ts_code"][0])
                latest_end = code_group.get_column("end_date").max()
                top_holdings[code] = (
                    code_group.filter(pl.col("end_date") == latest_end)
                    .sort("stk_mkv_ratio", descending=True)
                    .head(3)
                    .get_column("stock_code")
                    .to_list()
                )
        rows: list[tuple[str, float, float, list[str]]] = []
        for row in latest.iter_rows(named=True):
            code = str(row["ts_code"])
            name = str(names.get(code, ""))
            matched = [
                event
                for event in event_output.events
                if code in event.codes
                or any(industry and industry.lower() in name.lower() for industry in event.industries)
            ]
            if not matched:
                continue
            signed_strength = sum(
                event.confidence * (1 if event.direction == "POSITIVE" else -1 if event.direction == "NEGATIVE" else 0)
                for event in matched
            )
            volatility = max(float(row.get("volatility_20") or 0.0), 1e-6)
            move = float(row.get("ret_5") or 0.0)
            aligned_move = max(float(np.sign(signed_strength) * move), 0.0)
            price_in = float(1.0 - math.exp(-aligned_move / (volatility * math.sqrt(5))))
            score = float(np.clip(50.0 + signed_strength * (1.0 - price_in) * 40.0, 0.0, 100.0))
            constituents = top_holdings.get(code, [])
            constituent_returns = [stock_returns[value] for value in constituents if value in stock_returns]
            reasons = [
                f"映射到 {len(matched)} 个截止日前事件",
                f"近5日收益 {move:.2%}，估计 price-in 比例 {price_in:.0%}",
            ]
            if constituent_returns:
                confirmation = float(np.mean(constituent_returns))
                score = float(
                    np.clip(
                        score + np.sign(signed_strength) * np.clip(confirmation / 0.05, -1.0, 1.0) * 10.0,
                        0.0,
                        100.0,
                    ),
                )
                constituent_valuations = [
                    valuation_metrics[value] for value in constituents if value in valuation_metrics
                ]
                reasons.append(
                    f"Top3成分股近60日均值 {confirmation:.2%}，"
                    f"获取到 {len(constituent_valuations)} 只成分股的截止日估值快照",
                )
                pe_values = [value[0] for value in constituent_valuations if value[0] is not None and value[0] > 0]
                pb_values = [value[1] for value in constituent_valuations if value[1] is not None and value[1] > 0]
                if pe_values or pb_values:
                    pe_text = f"{float(np.median(pe_values)):.1f}" if pe_values else "-"
                    pb_text = f"{float(np.median(pb_values)):.2f}" if pb_values else "-"
                    reasons.append(f"Top3截止日估值中位数 PE(TTM)={pe_text}、PB={pb_text}")
            rows.append(
                (
                    code,
                    score,
                    price_in,
                    reasons,
                ),
            )
        rows.sort(key=lambda value: (-value[1], value[0]))
        candidates = [
            EtfScore(
                code=code,
                name=names.get(code, ""),
                rank=rank,
                score=score,
                confidence=min(1.0, abs(score - 50.0) / 50.0),
                price_in=price_in,
                reasons=reasons,
            )
            for rank, (code, score, price_in, reasons) in enumerate(rows[:20], 1)
        ]
        return DimensionRanking(
            dimension="event",
            as_of=as_of,
            status="COMPLETE" if candidates else "INSUFFICIENT_DATA",
            methodology="事件方向×置信度×(1-price-in)，price-in由ETF事件后收益/20日波动估算",
            candidates=candidates,
            limitations=[] if candidates else ["事件行业/代码未能映射到流动性 ETF 候选池。"],
        )

    def _backtest_ranking(
        self,
        features: pl.DataFrame,
        liquidity: pl.DataFrame,
        as_of: datetime,
    ) -> DimensionRanking:
        names = dict(zip(liquidity["ts_code"].to_list(), liquidity["name"].to_list()))
        try:
            scores, metrics, _, _ = self._fit_ranker(
                features,
                feature_columns=list(self.feature_columns),
                label_column="future_open_return",
                names=names,
            )
        except ValueError as exc:
            return DimensionRanking(
                dimension="backtest",
                as_of=as_of,
                status="INSUFFICIENT_DATA",
                methodology="以前一交易日复权收盘特征预测下一交易日开盘到再下一交易日开盘收益",
                model_name="ExtraTreesRegressor(local)",
                limitations=[str(exc)],
            )
        return DimensionRanking(
            dimension="backtest",
            as_of=as_of,
            status="COMPLETE",
            methodology=("前75%特征日训练、后25%特征日样本外验证；" "D日复权收盘特征预测D+1开盘到D+2开盘收益"),
            model_name="ExtraTreesRegressor(local)",
            candidates=scores,
            metrics=metrics,
            extremes=self._extremes(features, names),
        )

    def _us_ranking(
        self,
        features: pl.DataFrame,
        liquidity: pl.DataFrame,
        us_daily: pl.DataFrame,
        as_of: datetime,
    ) -> DimensionRanking:
        if us_daily.is_empty():
            return DimensionRanking(
                dimension="us_correlation",
                as_of=as_of,
                status="INSUFFICIENT_DATA",
                methodology="已完成美股close-close映射下一A股交易日ETF open-open",
                model_name="ExtraTreesRegressor(local)",
                limitations=["TuShare us_daily_adj 没有返回可用复权数据。"],
            )
        required_us = {"ts_code", "trade_date", "close", "adj_factor"}
        if not required_us.issubset(us_daily.columns):
            return DimensionRanking(
                dimension="us_correlation",
                as_of=as_of,
                status="INSUFFICIENT_DATA",
                methodology="美股复权close-close特征预测A股下一交易日open-open",
                model_name="ExtraTreesRegressor(local)",
                limitations=["us_daily_adj 的 close 与 adj_factor 是必需字段；禁止回退未复权价格。"],
            )
        us = (
            us_daily.with_columns(
                _date_expr("trade_date").alias("us_date"),
                (
                    pl.col("close").cast(pl.Float64, strict=False) * pl.col("adj_factor").cast(pl.Float64, strict=False)
                ).alias("adjusted_close"),
                pl.col("ts_code").cast(pl.String),
            )
            .filter(pl.col("adjusted_close").is_not_null() & (pl.col("adjusted_close") > 0))
            .sort(["ts_code", "us_date"])
            .with_columns(
                (pl.col("adjusted_close") / pl.col("adjusted_close").shift(1) - 1.0).over("ts_code").alias("us_return"),
            )
        )
        pivot = (
            us.drop_nulls(["us_date", "us_return"])
            .pivot(on="ts_code", index="us_date", values="us_return", aggregate_function="last")
            .sort("us_date")
        )
        us_columns = [column for column in pivot.columns if column != "us_date"]
        if len(us_columns) < 2:
            return DimensionRanking(
                dimension="us_correlation",
                as_of=as_of,
                status="INSUFFICIENT_DATA",
                methodology="已完成美股close-close映射下一A股交易日ETF open-open",
                model_name="ExtraTreesRegressor(local)",
                limitations=["至少需要两个美股股票/ETF的历史收益序列。"],
            )
        # A feature row dated D predicts the next A-share session T. A US session
        # dated before T closes before T's A-share open, including the usual US
        # session dated D that closes around 04:00/05:00 China time on T.
        aligned = (
            features.drop_nulls("prediction_date")
            .sort("prediction_date")
            .join_asof(
                pivot.rename({"us_date": "matched_us_date"}).sort("matched_us_date"),
                left_on="prediction_date",
                right_on="matched_us_date",
                strategy="backward",
                allow_exact_matches=False,
            )
            .filter(pl.col("matched_us_date") < pl.col("prediction_date"))
        )
        current_features = (
            features.sort(["ts_code", "trade_date"])
            .group_by("ts_code", maintain_order=True)
            .tail(1)
            .with_columns(pl.lit(as_of.date()).alias("prediction_date"))
            .sort("prediction_date")
            .join_asof(
                pivot.rename({"us_date": "matched_us_date"}).sort("matched_us_date"),
                left_on="prediction_date",
                right_on="matched_us_date",
                strategy="backward",
                allow_exact_matches=False,
            )
            .filter(pl.col("matched_us_date") < pl.col("prediction_date"))
        )
        feature_columns = [*us_columns, "ret_1", "ret_5", "volatility_20"]
        names = dict(zip(liquidity["ts_code"].to_list(), liquidity["name"].to_list()))
        try:
            scores, metrics, _, _ = self._fit_ranker(
                aligned,
                feature_columns=feature_columns,
                label_column="future_open_return",
                names=names,
                prediction_frame=current_features,
            )
        except ValueError as exc:
            return DimensionRanking(
                dimension="us_correlation",
                as_of=as_of,
                status="INSUFFICIENT_DATA",
                methodology="前一A股复权收盘特征加开盘前已完成美股复权收益，预测下一A股open-open",
                model_name="ExtraTreesRegressor(local)",
                limitations=[str(exc)],
            )
        return DimensionRanking(
            dimension="us_correlation",
            as_of=as_of,
            status="COMPLETE",
            methodology=("D日A股复权收盘特征加D+1开盘前已完成的美股复权close-close；" "预测D+1开盘到D+2开盘收益"),
            model_name="ExtraTreesRegressor(local)",
            candidates=scores,
            metrics=metrics,
            extremes=self._extremes(features, names),
            limitations=["SOXL使用实际逐日复合收益，不做简单除以3处理。"],
        )

    @staticmethod
    def fuse(
        rankings: dict[str, DimensionRanking],
        *,
        as_of: datetime,
        weights: dict[str, float],
    ) -> FusionRanking:
        """Fuse available dimension scores with per-code weight normalization."""
        if any(not math.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("fusion weights must be finite and non-negative")
        available = {
            name: ranking
            for name, ranking in rankings.items()
            if ranking.status == "COMPLETE" and ranking.candidates and weights.get(name, 0) > 0
        }
        total = sum(weights[name] for name in available)
        normalized = {name: weights[name] / total for name in available} if total else {}
        by_code: dict[str, list[tuple[str, EtfScore]]] = {}
        for name, ranking in available.items():
            for candidate in ranking.candidates:
                by_code.setdefault(candidate.code, []).append((name, candidate))
        fused: list[tuple[str, str, float, float, list[str], list[str]]] = []
        for code, values in by_code.items():
            code_weight = sum(normalized[name] for name, _ in values)
            score = sum(normalized[name] * candidate.score for name, candidate in values) / code_weight
            expected = [
                (name, candidate.expected_return) for name, candidate in values if candidate.expected_return is not None
            ]
            expected_weight = sum(normalized[name] for name, _ in expected)
            flags = [flag for _, candidate in values for flag in candidate.flags]
            fused.append(
                (
                    code,
                    next((candidate.name for _, candidate in values if candidate.name), ""),
                    float(np.clip(score, 0.0, 100.0)),
                    (
                        float(sum(normalized[name] * value for name, value in expected) / expected_weight)
                        if expected_weight
                        else 0.0
                    ),
                    [f"{name}={candidate.score:.1f}" for name, candidate in values],
                    flags,
                ),
            )
        fused.sort(key=lambda value: (-value[2], value[0]))
        candidates = [
            EtfScore(
                code=code,
                name=name,
                rank=rank,
                score=score,
                expected_return=expected if expected else None,
                confidence=min(1.0, abs(score - 50.0) / 50.0),
                reasons=reasons,
                flags=flags,
            )
            for rank, (code, name, score, expected, reasons, flags) in enumerate(fused[:20], 1)
        ]
        return FusionRanking(
            as_of=as_of,
            weights=normalized,
            candidates=candidates,
            methodology="仅对可用维度重归一化权重；同一ETF按维度分数加权，缺失维度不以零分惩罚",
            limitations=[] if len(available) == 3 else [f"仅融合可用维度：{', '.join(available) or '无'}"],
        )

    def run(
        self,
        bundle: dict[str, pl.DataFrame],
        *,
        event_output: EventAnalysisOutput,
        as_of: datetime,
        universe_size: int,
        weights: dict[str, float],
        us_ranking_override: DimensionRanking | None = None,
    ) -> tuple[dict[str, DimensionRanking], FusionRanking]:
        """Calculate event, backtest, US, and fused ETF rankings."""
        daily, liquidity = self._prepare_universe(bundle, universe_size)
        features = self._features(daily) if not daily.is_empty() else pl.DataFrame()
        rankings = {
            "event": self._event_ranking(event_output, features, liquidity, bundle),
            "backtest": self._backtest_ranking(features, liquidity, as_of),
            "us_correlation": (
                us_ranking_override
                if us_ranking_override is not None
                else self._us_ranking(
                    features,
                    liquidity,
                    bundle.get("us_daily", pl.DataFrame()),
                    as_of,
                )
            ),
        }
        return rankings, self.fuse(rankings, as_of=as_of, weights=weights)


@R.register("auto_fin_quant_step")
class AutoFinQuantStep(AutoFinAnalysisStep):
    """Fetch traceable market data and calculate all deterministic rankings."""

    outbound_proxy: BaseOutboundProxy | None = Ref(
        BaseOutboundProxy,
        ComponentEnum.OUTBOUND_PROXY,
        optional=True,
    )

    @staticmethod
    def _empty_rankings(as_of: datetime, reason: str) -> dict[str, DimensionRanking]:
        return {
            name: DimensionRanking(
                dimension=name,
                as_of=as_of,
                status="INSUFFICIENT_DATA",
                methodology="deterministic TuShare/Polars research",
                limitations=[reason],
            )
            for name in _DIMENSIONS
        }

    async def _persist_bundle(
        self,
        root: Path,
        bundle: dict[str, pl.DataFrame],
        manifest: dict[str, Any],
    ) -> str:
        root.mkdir(parents=True, exist_ok=True)
        snapshots: dict[str, dict[str, Any]] = {}
        for name, frame in bundle.items():
            if frame.is_empty():
                continue
            path = root / f"{name}.parquet"
            temp = root / f".{name}.{os.getpid()}.parquet.tmp"
            frame.write_parquet(temp)
            os.replace(temp, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            snapshots[name] = {
                "path": path.relative_to(self.workspace_path).as_posix(),
                "rows": frame.height,
                "sha256": digest,
            }
        manifest["snapshots"] = snapshots
        path = root / "manifest.json"
        await write_atomic(path, json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n")
        return path.relative_to(self.workspace_path).as_posix()

    async def execute(self):
        run_context = self.state("run_context")
        event_output = self.state("event_output")
        if not isinstance(run_context, dict) or not isinstance(event_output, EventAnalysisOutput):
            raise RuntimeError("Auto Fin event output and run context are required before quantitative research")
        self.require_checkpoint_reached(run_context)
        as_of = datetime.fromisoformat(str(run_context["data_cutoff"]))
        weights = self.state("fusion_weights", {"event": 0.30, "backtest": 0.45, "us_correlation": 0.25})
        if not isinstance(weights, dict):
            raise ValueError("fusion_weights must be a dimension-to-weight object")
        weights = {name: float(weights.get(name, 0.0)) for name in _DIMENSIONS}
        bundle_value = self.state("quant_bundle")
        us_ranking_override = self.state("us_ranking_override")
        if us_ranking_override is not None and not isinstance(us_ranking_override, DimensionRanking):
            raise ValueError("us_ranking_override must be a DimensionRanking")
        bundle: dict[str, pl.DataFrame]
        sources: list[dict[str, Any]] = []
        error = ""
        try:
            if not bool(self.state("quant_enabled", True)):
                raise RuntimeError("deterministic quantitative research is disabled")
            if isinstance(bundle_value, dict):
                bundle = {name: _records_to_frame(value) for name, value in bundle_value.items()}
            else:
                token = os.getenv("TUSHARE_TOKEN", "").strip()
                if not token:
                    raise RuntimeError("TUSHARE_TOKEN is required for deterministic ETF research")
                client = TushareResearchClient(
                    token,
                    concurrency=int(self.state("quant_concurrency", 6)),
                    proxy_url=self.outbound_proxy.http_url if self.outbound_proxy is not None else None,
                )
                client_history_end = date.fromisoformat(str(run_context["previous_trade_date"]))
                bundle = await client.fetch_bundle(
                    history_end=client_history_end,
                    us_history_end=as_of.date() - timedelta(days=1),
                    lookback_days=int(self.state("quant_lookback_days", 540)),
                    preselect_size=max(int(self.state("quant_universe_size", 50)) * 2, 60),
                    fetch_us=us_ranking_override is None and not bool(self.state("skip_us_research", False)),
                )
                sources = client.sources
            engine = AutoFinQuantResearch(
                tree_count=int(self.state("quant_tree_count", 31)),
            )
            rankings, fusion = engine.run(
                bundle,
                event_output=event_output,
                as_of=as_of,
                universe_size=int(self.state("quant_universe_size", 50)),
                weights=weights,
                us_ranking_override=us_ranking_override,
            )
            incomplete = [name for name, ranking in rankings.items() if ranking.status != "COMPLETE"]
            if incomplete:
                error = f"quantitative dimensions incomplete: {', '.join(incomplete)}"
            root = (
                self.workspace_path
                / str(self.config_value("resource_dir"))
                / "auto-fin"
                / str(run_context["trade_date"])
                / "quant"
                / str(run_context["checkpoint"])
            )
            manifest_path = await self._persist_bundle(
                root,
                bundle,
                {
                    "schema_version": "auto-fin-quant/v1",
                    "run_id": run_context["run_id"],
                    "data_cutoff": run_context["data_cutoff"],
                    "market_cutoff": run_context["market_cutoff"],
                    "parameters": {
                        "universe_size": int(self.state("quant_universe_size", 50)),
                        "lookback_days": int(self.state("quant_lookback_days", 540)),
                        "tree_count": int(self.state("quant_tree_count", 31)),
                        "fusion_weights": weights,
                        "adjustment": (
                            "ETF OHLC×fund_adj; A-share close×adj_factor; " "US close×us_daily_adj.adj_factor"
                        ),
                        "target": "D features -> A-share open(D+1) to open(D+2)",
                    },
                    "code_version": "auto-fin-quant/v2",
                    "parameter_hash": hashlib.sha256(
                        json.dumps(
                            {
                                "universe_size": int(self.state("quant_universe_size", 50)),
                                "lookback_days": int(self.state("quant_lookback_days", 540)),
                                "tree_count": int(self.state("quant_tree_count", 31)),
                                "fusion_weights": weights,
                                "adjustment": (
                                    "ETF OHLC×fund_adj; A-share close×adj_factor; " "US close×us_daily_adj.adj_factor"
                                ),
                                "target": "D features -> A-share open(D+1) to open(D+2)",
                            },
                            sort_keys=True,
                        ).encode(),
                    ).hexdigest(),
                    "sources": sources,
                    "rankings": {name: ranking.model_dump(mode="json") for name, ranking in rankings.items()},
                    "fusion": fusion.model_dump(mode="json"),
                },
            )
            rankings = {
                name: ranking.model_copy(update={"manifest_path": manifest_path}) for name, ranking in rankings.items()
            }
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Provider permissions, rate limits, and network failures degrade research;
            # they must not mutate or invalidate the deterministic portfolio ledger.
            error = f"{type(exc).__name__}: {exc}"
            self.logger.warning(f"[{self.name}] quantitative research degraded: {error}")
            rankings = self._empty_rankings(as_of, error)
            fusion = AutoFinQuantResearch.fuse(rankings, as_of=as_of, weights=weights)
        self.set_state("quant_rankings", rankings)
        self.set_state("fusion_ranking", fusion)
        self.set_state("quant_error", error)
        self.set_state(
            "event_output",
            event_output.model_copy(update={"ranking": rankings["event"]}),
        )
        self.logger.info(
            f"[{self.name}] quantitative research done "
            f"event={len(rankings['event'].candidates)} "
            f"backtest={len(rankings['backtest'].candidates)} "
            f"us={len(rankings['us_correlation'].candidates)} fusion={len(fusion.candidates)}",
        )
        assert self.context is not None
        return self.context.response

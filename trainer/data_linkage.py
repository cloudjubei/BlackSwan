"""Asset-linkage graph: ties a traded asset to the related series that drive it.

Each edge points a traded asset (a stock, FX pair, crypto, commodity) at a MINEABLE proxy series already
in the catalog (a commodity, macro series, …) with a typed relationship + rationale. The proxy carries
its own point-in-time anchor (`barCloseTz`) so the fusion loader joins by TIMESTAMP, never by date string
(a same-date cross-asset join imports later-session information). Surfaced on the Data tab as per-asset
"related data" chips so a user can mine the drivers of an asset in one place.

Seed edges are verified relationships between instruments ALREADY in the catalog (so every edge is
immediately mineable). The graph grows via guided discovery (Phase D4) behind a human-approve gate.
"""

from dataclasses import dataclass

from trainer import data_catalog

# Edge relationship types.
INPUT_COST = "input-cost"
COMMODITY_INPUT = "commodity-input"
SECTOR_PEER = "sector-peer"
TERMS_OF_TRADE = "terms-of-trade"
RATE_DIFFERENTIAL = "rate-differential"
CURVE_SLOPE = "curve-slope"
REAL_RATE = "real-rate"
MACRO_DRIVER = "macro-driver"


@dataclass(frozen=True)
class Edge:
    asset: str          # local symbol of the traded asset
    asset_class: str    # its class (so a fundamentals ticker stays distinct from the same stock ticker)
    proxy: str          # local symbol of the mineable proxy series (resolved via the catalog)
    edge_type: str
    rationale: str


_EDGES = [
    Edge("JPM", data_catalog.STOCKS, "T10Y2Y", CURVE_SLOPE, "Bank net-interest-margin tracks the yield-curve slope."),
    Edge("USDJPY", data_catalog.FX, "DGS10", RATE_DIFFERENTIAL, "USD/JPY is driven by the US-Japan rate differential (10Y)."),
    Edge("AUDUSD", data_catalog.FX, "COPPER", TERMS_OF_TRADE, "AUD is a commodity currency that tracks industrial-metal demand."),
    Edge("USDCAD", data_catalog.FX, "WTI", TERMS_OF_TRADE, "CAD strengthens with oil, so USD/CAD falls as WTI rises."),
    Edge("GOLD", data_catalog.COMMODITIES, "DFII10", REAL_RATE, "Gold inversely tracks the 10Y real yield."),
    Edge("TSLA", data_catalog.STOCKS, "COPPER", INPUT_COST, "EV production is copper-intensive."),
    Edge("WMT", data_catalog.STOCKS, "CPIAUCNS", MACRO_DRIVER, "Retail demand + margins track consumer prices."),
    Edge("BTCUSDT", data_catalog.CRYPTO, "DFF", MACRO_DRIVER, "Crypto liquidity tracks the policy rate."),
    # ETF-proxy edges (Phase D3 completion): semiconductor + lithium supply chains and the sector proxies.
    Edge("NVDA", data_catalog.STOCKS, "SOXX", SECTOR_PEER, "NVIDIA moves with the semiconductor sector."),
    Edge("AVGO", data_catalog.STOCKS, "SOXX", SECTOR_PEER, "Broadcom moves with the semiconductor sector."),
    Edge("AAPL", data_catalog.STOCKS, "SMH", SECTOR_PEER, "Apple's supply + demand track the semiconductor sector."),
    Edge("TSLA", data_catalog.STOCKS, "LIT", INPUT_COST, "EV batteries are lithium-intensive."),
    Edge("XLE", data_catalog.ETFS, "WTI", COMMODITY_INPUT, "Energy-sector revenue tracks crude oil."),
    Edge("JETS", data_catalog.ETFS, "WTI", INPUT_COST, "Jet fuel (crude-linked) is the top airline cost."),
    Edge("GDX", data_catalog.ETFS, "GOLD", COMMODITY_INPUT, "Gold miners lever the bullion price."),
    Edge("XLF", data_catalog.ETFS, "T10Y2Y", CURVE_SLOPE, "Bank net-interest-margin tracks the yield-curve slope."),
]


def _edge_dict(edge: Edge) -> dict:
    proxy = data_catalog.instrument(edge.proxy)
    return {
        "asset": edge.asset,
        "assetClass": edge.asset_class,
        "proxy": edge.proxy,
        "edgeType": edge.edge_type,
        "rationale": edge.rationale,
        "proxySource": proxy.source if proxy else None,
        "proxyClass": proxy.asset_class if proxy else None,
        "proxyBarCloseTz": proxy.bar_close_tz if proxy else None,
    }


def edges() -> list:
    """Every linkage edge, each decorated with its proxy's source + point-in-time anchor."""
    return [_edge_dict(edge) for edge in _EDGES]


def edges_for(asset: str, asset_class=None) -> list:
    """The edges FROM ``asset`` (optionally scoped to ``asset_class``)."""
    return [
        edge
        for edge in edges()
        if edge["asset"] == asset and (asset_class is None or edge["assetClass"] == asset_class)
    ]


def linkage() -> dict:
    """The linkage graph as an emittable payload."""
    return {"edges": edges()}

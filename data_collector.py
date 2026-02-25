"""
Market Data Collector with robust error handling and multiple exchange support
"""
import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Structured market data container"""
    timestamp: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    symbol: str
    exchange: str
    
class DataCollector:
    """Collects market data from multiple exchanges with fallback support"""
    
    def __init__(self, config):
        self.config = config
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self._initialize_exchanges()
        self.data_cache = {}
        self.cache_lock = threading.Lock()
        
    def _initialize_exchanges(self) -> None:
        """Initialize exchange connections with error handling"""
        exchanges_to_try = ['binance', 'kraken', 'coinbase']
        
        for exchange_name in exchanges_to_try:
            try:
                exchange_class = getattr(ccxt, exchange_name)
                exchange = exchange_class({
                    'enableRateLimit': True,
                    'timeout': 30000,
                })
                
                # Test connection
                exchange.load_markets()
                self.exchanges[exchange_name] = exchange
                logger.info(f"Successfully connected to {exchange_name}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize {exchange_name}: {e}")
                continue
                
        if not self.exchanges:
            raise ConnectionError("No exchanges could be initialized")
            
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 1000,
        exchange_name: Optional[str] = None
    ) -> Optional[MarketData]:
        """
        Fetch OHLCV
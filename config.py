"""
ADLTS Configuration Manager
Centralized configuration with environment variables and Firebase fallback
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
import logging

# Load environment variables
load_dotenv()

@dataclass
class TradingConfig:
    """Configuration for trading parameters"""
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    initial_balance: float = 10000.0
    max_position_size: float = 0.1  # 10% of balance
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.05  # 5%
    
@dataclass
class ModelConfig:
    """Configuration for deep learning models"""
    lstm_units: int = 50
    dense_units: int = 25
    dropout_rate: float = 0.2
    sequence_length: int = 60
    batch_size: int = 32
    learning_rate: float = 0.001
    
@dataclass
class RLConfig:
    """Configuration for reinforcement learning"""
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    target_update_freq: int = 100
    
class ConfigManager:
    """Manages configuration with Firebase fallback"""
    
    def __init__(self):
        self._db = None
        self.trading = TradingConfig()
        self.model = ModelConfig()
        self.rl = RLConfig()
        self._initialize_firebase()
        
    def _initialize_firebase(self) -> None:
        """Initialize Firebase connection for config management"""
        try:
            # Check for Firebase credentials
            cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
            if cred_path and os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                self._db = firestore.client()
                logging.info("Firebase initialized successfully")
            else:
                logging.warning("Firebase credentials not found, using local config only")
        except Exception as e:
            logging.error(f"Failed to initialize Firebase: {e}")
            
    def get_live_config(self, key: str) -> Optional[Any]:
        """Retrieve configuration from Firebase if available"""
        if not self._db:
            return None
            
        try:
            doc_ref = self._db.collection("adlts_config").document(key)
            doc = doc_ref.get()
            return doc.to_dict() if doc.exists else None
        except Exception as e:
            logging.error(f"Failed to fetch config from Firebase: {e}")
            return None
            
    def update_config(self, key: str, data: Dict[str, Any]) -> bool:
        """Update configuration in Firebase"""
        if not self._db:
            return False
            
        try:
            doc_ref = self._db.collection("adlts_config").document(key)
            doc_ref.set(data, merge=True)
            return True
        except Exception as e:
            logging.error(f"Failed to update config in Firebase: {e}")
            return False

# Global configuration instance
config = ConfigManager()
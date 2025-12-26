# blofin_trader.py - Grok Elite Signal Bot v27.12.13 - Blofin Auto-Trading Integration
# -*- coding: utf-8 -*-
"""
Blofin Exchange Auto-Trading Module for Signal Execution.

v27.12.13 CRITICAL FIX:
- Made brokerId configurable via BLOFIN_BROKER_ID environment variable
- Default to empty string "" for Transaction API Keys
- Added session-based trade timing
- Added correlation-based position sizing
- Added drawdown protection
- Improved error handling with specific error codes

Features:
- Automatic order placement from bot signals
- Position management with TP/SL
- Risk management with position sizing
- WebSocket for real-time updates
- Comprehensive error handling

API Documentation: https://docs.blofin.com
"""
import os
import json
import hmac
import base64
import hashlib
import asyncio
import logging
import aiohttp
from uuid import uuid4
from datetime import datetime, timezone
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class BlofinConfig:
    """Blofin API Configuration"""
    api_key: str = field(default_factory=lambda: os.getenv("BLOFIN_API_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.getenv("BLOFIN_SECRET_KEY", ""))
    passphrase: str = field(default_factory=lambda: os.getenv("BLOFIN_PASSPHRASE", ""))
    demo_mode: bool = field(default_factory=lambda: os.getenv("BLOFIN_DEMO_MODE", "false").lower() == "true")
    
    # v27.12.13: CRITICAL FIX - Configurable broker ID
    # Set to empty string "" for Transaction API Keys (most common)
    # Set to your broker ID if using a Broker API Key
    broker_id: str = field(default_factory=lambda: os.getenv("BLOFIN_BROKER_ID", ""))
    
    # Trading settings
    auto_trade_enabled: bool = field(default_factory=lambda: os.getenv("AUTO_TRADE_ENABLED", "false").lower() == "true")
    risk_per_trade: float = field(default_factory=lambda: float(os.getenv("AUTO_TRADE_RISK_PCT", "0.015")))
    max_leverage: int = field(default_factory=lambda: int(os.getenv("AUTO_TRADE_MAX_LEVERAGE", "5")))
    default_leverage: int = field(default_factory=lambda: int(os.getenv("AUTO_TRADE_DEFAULT_LEVERAGE", "3")))
    margin_mode: str = field(default_factory=lambda: os.getenv("AUTO_TRADE_MARGIN_MODE", "isolated"))
    position_mode: str = field(default_factory=lambda: os.getenv("AUTO_TRADE_POSITION_MODE", "net_mode"))
    
    # API endpoints
    @property
    def base_url(self) -> str:
        if self.demo_mode:
            return "https://demo-trading-openapi.blofin.com"
        return "https://openapi.blofin.com"
    
    @property
    def ws_public_url(self) -> str:
        if self.demo_mode:
            return "wss://demo-trading-openapi.blofin.com/ws/public"
        return "wss://openapi.blofin.com/ws/public"
    
    @property
    def ws_private_url(self) -> str:
        if self.demo_mode:
            return "wss://demo-trading-openapi.blofin.com/ws/private"
        return "wss://openapi.blofin.com/ws/private"
    
    def is_configured(self) -> bool:
        """Check if API credentials are configured"""
        return bool(self.api_key and self.secret_key and self.passphrase)


# ============================================================================
# SESSION-BASED TRADING (v27.12.13)
# ============================================================================
SESSION_MULTIPLIERS = {
    'asia': 0.85,              # Lower probability during Asia
    'london': 1.0,             # Normal during London
    'new_york': 1.1,           # Highest volatility
    'london_ny_overlap': 1.2,  # Best for entries
    'asia_close': 0.75         # Avoid this session
}

def get_current_session() -> Tuple[str, float]:
    """Get current trading session and its multiplier."""
    now = datetime.now(timezone.utc)
    hour = now.hour
    
    if 13 <= hour <= 17:  # London/NY overlap (13:00-17:00 UTC)
        return 'london_ny_overlap', SESSION_MULTIPLIERS['london_ny_overlap']
    elif 8 <= hour <= 12:  # London (08:00-12:00 UTC)
        return 'london', SESSION_MULTIPLIERS['london']
    elif 14 <= hour <= 21:  # New York (14:00-21:00 UTC)
        return 'new_york', SESSION_MULTIPLIERS['new_york']
    elif 0 <= hour <= 7:  # Asia (00:00-07:00 UTC)
        return 'asia', SESSION_MULTIPLIERS['asia']
    else:
        return 'asia_close', SESSION_MULTIPLIERS['asia_close']


# ============================================================================
# CORRELATION MANAGEMENT (v27.12.13)
# ============================================================================
CORRELATION_MAP = {
    'ETH/USDT': {'BTC/USDT': 0.85},
    'SOL/USDT': {'BTC/USDT': 0.75, 'ETH/USDT': 0.70},
    'BNB/USDT': {'BTC/USDT': 0.80, 'ETH/USDT': 0.75},
    'XRP/USDT': {'BTC/USDT': 0.60},
    'ADA/USDT': {'BTC/USDT': 0.70, 'ETH/USDT': 0.65},
    'AVAX/USDT': {'BTC/USDT': 0.70, 'SOL/USDT': 0.75},
}

def get_correlation_reduction(symbol: str, open_positions: List[str]) -> float:
    """
    Calculate position size reduction based on correlated open positions.
    
    Returns a multiplier (0.4 - 1.0) to apply to position size.
    """
    corr_reduction = 0
    correlations = CORRELATION_MAP.get(symbol, {})
    
    for open_symbol in open_positions:
        if open_symbol in correlations:
            corr_reduction += correlations[open_symbol] * 0.3  # 30% reduction per correlated asset
    
    return max(0.4, 1 - corr_reduction)  # Min 40% of base size


# ============================================================================
# DRAWDOWN PROTECTION (v27.12.13)
# ============================================================================
DRAWDOWN_THRESHOLDS = {
    'warning': 0.03,    # 3% drawdown - reduce size to 50%
    'caution': 0.05,    # 5% drawdown - reduce size to 25%
    'stop': 0.08        # 8% drawdown - pause trading
}

def get_drawdown_multiplier(current_balance: float, peak_balance: float) -> Tuple[float, str]:
    """
    Get position size multiplier based on current drawdown.
    
    Returns:
        Tuple of (multiplier, status)
    """
    if peak_balance <= 0:
        return 1.0, 'normal'
    
    drawdown = (peak_balance - current_balance) / peak_balance
    
    if drawdown >= DRAWDOWN_THRESHOLDS['stop']:
        return 0.0, 'stopped'
    elif drawdown >= DRAWDOWN_THRESHOLDS['caution']:
        return 0.25, 'caution'
    elif drawdown >= DRAWDOWN_THRESHOLDS['warning']:
        return 0.50, 'warning'
    return 1.0, 'normal'


# ============================================================================
# BLOFIN API CLIENT
# ============================================================================
class BlofinClient:
    """
    Async Blofin API Client for trading operations.
    
    Usage:
        async with BlofinClient() as client:
            balance = await client.get_balance()
            order = await client.place_order(...)
    """
    
    def __init__(self, config: Optional[BlofinConfig] = None):
        self.config = config or BlofinConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger("BlofinClient")
        
        # Symbol mapping (Grok format -> Blofin format)
        self.symbol_map = {
            "BTC/USDT": "BTC-USDT",
            "ETH/USDT": "ETH-USDT",
            "SOL/USDT": "SOL-USDT",
            "BNB/USDT": "BNB-USDT",
            "XRP/USDT": "XRP-USDT",
            "ADA/USDT": "ADA-USDT",
            "AVAX/USDT": "AVAX-USDT",
            "DOGE/USDT": "DOGE-USDT",
            "LINK/USDT": "LINK-USDT",
            "DOT/USDT": "DOT-USDT",
        }
        
        # Instrument info cache
        self._instruments: Dict[str, Dict] = {}
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convert symbol from Grok format to Blofin format"""
        return self.symbol_map.get(symbol, symbol.replace("/", "-"))
    
    def _generate_signature(
        self,
        method: str,
        path: str,
        body: Optional[Dict] = None
    ) -> Tuple[str, str, str]:
        """
        Generate Blofin API signature.
        
        Returns:
            Tuple of (signature, timestamp, nonce)
        """
        timestamp = str(int(datetime.now().timestamp() * 1000))
        nonce = str(uuid4())
        
        # Build prehash string - body is "" for GET, json string for POST
        if body:
            body_str = json.dumps(body)
        else:
            body_str = ""
        
        prehash = f"{path}{method}{timestamp}{nonce}{body_str}"
        
        # Generate HMAC-SHA256 signature
        hex_signature = hmac.new(
            self.config.secret_key.encode(),
            prehash.encode(),
            hashlib.sha256
        ).hexdigest().encode()
        
        # Convert to base64
        signature = base64.b64encode(hex_signature).decode()
        
        return signature, timestamp, nonce
    
    def _get_headers(
        self,
        method: str,
        path: str,
        body: Optional[Dict] = None
    ) -> Dict[str, str]:
        """Generate authenticated request headers"""
        signature, timestamp, nonce = self._generate_signature(method, path, body)
        
        return {
            "ACCESS-KEY": self.config.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-NONCE": nonce,
            "ACCESS-PASSPHRASE": self.config.passphrase,
            "Content-Type": "application/json"
        }
    
    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        body: Optional[Dict] = None
    ) -> Dict:
        """Make authenticated API request"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Build URL with query params for GET
        url = f"{self.config.base_url}{path}"
        if params and method == "GET":
            query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            if query:
                path = f"{path}?{query}"
                url = f"{self.config.base_url}{path}"
        
        headers = self._get_headers(method, path, body)
        
        try:
            async with self.session.request(
                method,
                url,
                headers=headers,
                json=body if method == "POST" else None
            ) as response:
                data = await response.json()
                
                if data.get("code") != "0":
                    self.logger.error(f"API Error: {data}")
                    
                    # v27.12.13: Add specific error handling
                    error_code = data.get("code", "unknown")
                    error_msg = data.get("msg", "Unknown error")
                    
                    # Provide helpful messages for common errors
                    if error_code == "152013":
                        error_msg = (
                            f"{error_msg} - Your API key requires a specific brokerId. "
                            f"Set BLOFIN_BROKER_ID environment variable. "
                            f"For Transaction API Keys, leave it empty (''). "
                            f"For Broker API Keys, use your assigned broker ID."
                        )
                    elif error_code == "152011":
                        error_msg = (
                            f"{error_msg} - Your Transaction API Key does not support brokerId. "
                            f"Set BLOFIN_BROKER_ID='' (empty string) in environment variables."
                        )
                    
                    raise BlofinAPIError(code=error_code, message=error_msg)
                
                return data
                
        except aiohttp.ClientError as e:
            self.logger.error(f"Request failed: {e}")
            raise BlofinAPIError(code="network", message=str(e))
    
    # ========================================================================
    # PUBLIC ENDPOINTS
    # ========================================================================
    
    async def get_instruments(self, inst_id: Optional[str] = None) -> List[Dict]:
        """Get available trading instruments"""
        params = {"instId": inst_id} if inst_id else {}
        response = await self._request("GET", "/api/v1/market/instruments", params=params)
        
        # Cache instrument info
        for inst in response.get("data", []):
            self._instruments[inst["instId"]] = inst
        
        return response.get("data", [])
    
    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get current ticker for symbol"""
        inst_id = self._convert_symbol(symbol)
        response = await self._request(
            "GET",
            "/api/v1/market/tickers",
            params={"instId": inst_id}
        )
        data = response.get("data", [])
        return data[0] if data else None
    
    async def get_mark_price(self, symbol: str) -> Optional[Dict]:
        """Get mark price for symbol"""
        inst_id = self._convert_symbol(symbol)
        response = await self._request(
            "GET",
            "/api/v1/market/mark-price",
            params={"instId": inst_id}
        )
        data = response.get("data", [])
        return data[0] if data else None
    
    # ========================================================================
    # ACCOUNT ENDPOINTS
    # ========================================================================
    
    async def get_balance(self) -> Dict:
        """Get futures account balance"""
        response = await self._request(
            "GET",
            "/api/v1/account/balance",
            params={"productType": "USDT-FUTURES"}
        )
        return response.get("data", {})
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get current positions"""
        params = {}
        if symbol:
            params["instId"] = self._convert_symbol(symbol)
        
        response = await self._request(
            "GET",
            "/api/v1/account/positions",
            params=params
        )
        return response.get("data", [])
    
    async def get_margin_mode(self) -> str:
        """Get current margin mode"""
        response = await self._request("GET", "/api/v1/account/margin-mode")
        return response.get("data", {}).get("marginMode", "isolated")
    
    async def set_margin_mode(self, mode: str) -> bool:
        """Set margin mode (cross/isolated)"""
        response = await self._request(
            "POST",
            "/api/v1/account/set-margin-mode",
            body={"marginMode": mode}
        )
        return response.get("code") == "0"
    
    async def get_position_mode(self) -> str:
        """Get position mode (net_mode/long_short_mode)"""
        response = await self._request("GET", "/api/v1/account/position-mode")
        return response.get("data", {}).get("positionMode", "net_mode")
    
    async def set_position_mode(self, mode: str) -> bool:
        """Set position mode"""
        response = await self._request(
            "POST",
            "/api/v1/account/set-position-mode",
            body={"positionMode": mode}
        )
        return response.get("code") == "0"
    
    async def get_leverage(self, symbol: str, margin_mode: str = "isolated") -> Dict:
        """Get leverage for symbol"""
        inst_id = self._convert_symbol(symbol)
        response = await self._request(
            "GET",
            "/api/v1/account/batch-leverage-info",
            params={"instId": inst_id, "marginMode": margin_mode}
        )
        data = response.get("data", [])
        return data[0] if data else {}
    
    async def set_leverage(
        self,
        symbol: str,
        leverage: int,
        margin_mode: str = "isolated",
        position_side: str = "net"
    ) -> bool:
        """Set leverage for symbol"""
        inst_id = self._convert_symbol(symbol)
        body = {
            "instId": inst_id,
            "leverage": str(leverage),
            "marginMode": margin_mode
        }
        if margin_mode == "isolated" and position_side != "net":
            body["positionSide"] = position_side
        
        response = await self._request(
            "POST",
            "/api/v1/account/set-leverage",
            body=body
        )
        return response.get("code") == "0"
    
    # ========================================================================
    # TRADING ENDPOINTS
    # ========================================================================
    
    def _build_order_body(self, base_body: Dict) -> Dict:
        """
        Build order body with optional brokerId.
        
        v27.12.13: Only include brokerId if it's set (non-empty).
        This fixes the "Unmatched brokerId" error for Transaction API Keys.
        """
        body = base_body.copy()
        
        # Only add brokerId if configured and non-empty
        if self.config.broker_id:
            body["brokerId"] = self.config.broker_id
        # If broker_id is empty string, don't include it at all
        
        return body
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        size: str,
        order_type: str = "market",
        price: Optional[str] = None,
        margin_mode: str = "isolated",
        position_side: str = "net",
        reduce_only: bool = False,
        tp_trigger_price: Optional[str] = None,
        tp_order_price: Optional[str] = None,
        sl_trigger_price: Optional[str] = None,
        sl_order_price: Optional[str] = None,
        client_order_id: Optional[str] = None
    ) -> Dict:
        """
        Place a new order.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            side: "buy" or "sell"
            size: Order size in contracts
            order_type: "market", "limit", "post_only", "fok", "ioc"
            price: Order price (required for limit orders)
            margin_mode: "cross" or "isolated"
            position_side: "net", "long", or "short"
            reduce_only: Close position only
            tp_trigger_price: Take profit trigger
            tp_order_price: Take profit order price (-1 for market)
            sl_trigger_price: Stop loss trigger
            sl_order_price: Stop loss order price (-1 for market)
            client_order_id: Custom order ID
        
        Returns:
            Order response dict
        """
        inst_id = self._convert_symbol(symbol)
        
        base_body = {
            "instId": inst_id,
            "marginMode": margin_mode,
            "positionSide": position_side,
            "side": side,
            "orderType": order_type,
            "size": str(size),
        }
        
        if order_type != "market" and price:
            base_body["price"] = str(price)
        
        if reduce_only:
            base_body["reduceOnly"] = "true"
        
        if client_order_id:
            base_body["clientOrderId"] = client_order_id
        
        # Add TP/SL if provided
        if tp_trigger_price:
            base_body["tpTriggerPrice"] = str(tp_trigger_price)
            base_body["tpOrderPrice"] = str(tp_order_price) if tp_order_price else "-1"
        
        if sl_trigger_price:
            base_body["slTriggerPrice"] = str(sl_trigger_price)
            base_body["slOrderPrice"] = str(sl_order_price) if sl_order_price else "-1"
        
        # v27.12.13: Use helper to conditionally add brokerId
        body = self._build_order_body(base_body)
        
        response = await self._request("POST", "/api/v1/trade/order", body=body)
        return response.get("data", [{}])[0]
    
    async def place_tpsl_order(
        self,
        symbol: str,
        side: str,
        size: str,
        margin_mode: str = "isolated",
        position_side: str = "net",
        tp_trigger_price: Optional[str] = None,
        tp_order_price: Optional[str] = None,
        sl_trigger_price: Optional[str] = None,
        sl_order_price: Optional[str] = None,
        reduce_only: bool = True,
        client_order_id: Optional[str] = None
    ) -> Dict:
        """Place a TP/SL order"""
        inst_id = self._convert_symbol(symbol)
        
        base_body = {
            "instId": inst_id,
            "marginMode": margin_mode,
            "positionSide": position_side,
            "side": side,
            "size": str(size),
            "reduceOnly": "true" if reduce_only else "false",
        }
        
        if tp_trigger_price:
            base_body["tpTriggerPrice"] = str(tp_trigger_price)
            base_body["tpOrderPrice"] = str(tp_order_price) if tp_order_price else "-1"
        
        if sl_trigger_price:
            base_body["slTriggerPrice"] = str(sl_trigger_price)
            base_body["slOrderPrice"] = str(sl_order_price) if sl_order_price else "-1"
        
        if client_order_id:
            base_body["clientOrderId"] = client_order_id
        
        # v27.12.13: Use helper to conditionally add brokerId
        body = self._build_order_body(base_body)
        
        response = await self._request("POST", "/api/v1/trade/order-tpsl", body=body)
        return response.get("data", {})
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an active order"""
        inst_id = self._convert_symbol(symbol)
        base_body = {
            "instId": inst_id,
            "orderId": order_id,
        }
        body = self._build_order_body(base_body)
        
        response = await self._request("POST", "/api/v1/trade/cancel-order", body=body)
        return response.get("code") == "0"
    
    async def cancel_tpsl_order(self, symbol: str, tpsl_id: str) -> bool:
        """Cancel a TP/SL order"""
        inst_id = self._convert_symbol(symbol)
        base_body = {
            "instId": inst_id,
            "tpslId": tpsl_id,
        }
        body = self._build_order_body(base_body)
        
        response = await self._request("POST", "/api/v1/trade/cancel-tpsl", body=body)
        return response.get("code") == "0"
    
    async def close_position(
        self,
        symbol: str,
        margin_mode: str = "isolated",
        position_side: str = "net"
    ) -> Dict:
        """Close entire position"""
        inst_id = self._convert_symbol(symbol)
        base_body = {
            "instId": inst_id,
            "marginMode": margin_mode,
            "positionSide": position_side,
        }
        body = self._build_order_body(base_body)
        
        response = await self._request("POST", "/api/v1/trade/close-position", body=body)
        return response.get("data", {})
    
    async def get_active_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get active orders"""
        params = {}
        if symbol:
            params["instId"] = self._convert_symbol(symbol)
        
        response = await self._request(
            "GET",
            "/api/v1/trade/orders-pending",
            params=params
        )
        return response.get("data", [])
    
    async def get_order_detail(self, symbol: str, order_id: str) -> Dict:
        """Get order details"""
        inst_id = self._convert_symbol(symbol)
        response = await self._request(
            "GET",
            "/api/v1/trade/order",
            params={"instId": inst_id, "orderId": order_id}
        )
        data = response.get("data", [])
        return data[0] if data else {}


# ============================================================================
# AUTO-TRADER - SIGNAL EXECUTION
# ============================================================================
class BlofinAutoTrader:
    """
    Automatic trade execution from Grok Elite signals.
    
    v27.12.13 Features:
    - Session-based trade timing
    - Correlation-aware position sizing
    - Drawdown protection
    - Configurable broker ID
    """
    
    def __init__(self, config: Optional[BlofinConfig] = None):
        self.config = config or BlofinConfig()
        self.client: Optional[BlofinClient] = None
        self.logger = logging.getLogger("BlofinAutoTrader")
        
        # Trade tracking
        self.executed_trades: Dict[str, Dict] = {}
        self.pending_signals: List[Dict] = []
        
        # Risk parameters
        self.min_size = Decimal("0.1")  # Minimum contract size
        
        # v27.12.13: Track peak balance for drawdown protection
        self.peak_balance: float = 0.0
        self.current_balance: float = 0.0
    
    async def initialize(self) -> bool:
        """Initialize the auto-trader"""
        if not self.config.is_configured():
            self.logger.warning("Blofin API not configured - auto-trading disabled")
            return False
        
        if not self.config.auto_trade_enabled:
            self.logger.info("Auto-trading is disabled")
            return False
        
        try:
            self.client = BlofinClient(self.config)
            await self.client.__aenter__()
            
            # Verify connection
            balance = await self.client.get_balance()
            if balance:
                equity = balance.get("totalEquity", "0")
                self.current_balance = float(equity)
                self.peak_balance = self.current_balance
                
                self.logger.info(f"✅ Blofin connected - Account equity: ${equity}")
                
                # Log broker ID configuration
                if self.config.broker_id:
                    self.logger.info(f"📋 Using broker ID: {self.config.broker_id}")
                else:
                    self.logger.info("📋 No broker ID configured (Transaction API Key mode)")
                
                # Set margin mode and position mode
                await self._setup_account()
                return True
            else:
                self.logger.error("Failed to get account balance")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Blofin: {e}")
            return False
    
    async def _setup_account(self):
        """Setup account settings"""
        try:
            # Set margin mode
            current_margin = await self.client.get_margin_mode()
            if current_margin != self.config.margin_mode:
                await self.client.set_margin_mode(self.config.margin_mode)
                self.logger.info(f"Set margin mode: {self.config.margin_mode}")
            
            # Set position mode
            current_pos_mode = await self.client.get_position_mode()
            if current_pos_mode != self.config.position_mode:
                await self.client.set_position_mode(self.config.position_mode)
                self.logger.info(f"Set position mode: {self.config.position_mode}")
                
        except Exception as e:
            self.logger.warning(f"Account setup warning: {e}")
    
    async def close(self):
        """Cleanup resources"""
        if self.client:
            await self.client.__aexit__(None, None, None)
    
    async def _update_balance(self):
        """Update current balance and track peak for drawdown protection"""
        if not self.client:
            return
        
        try:
            balance = await self.client.get_balance()
            self.current_balance = float(balance.get("totalEquity", "0"))
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
        except Exception as e:
            self.logger.warning(f"Failed to update balance: {e}")
    
    async def _get_open_position_symbols(self) -> List[str]:
        """Get list of symbols with open positions"""
        if not self.client:
            return []
        
        try:
            positions = await self.client.get_positions()
            return [
                pos.get("instId", "").replace("-", "/")
                for pos in positions
                if float(pos.get("positions", 0)) != 0
            ]
        except Exception:
            return []
    
    def _calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float,
        contract_value: float = 0.001,
        leverage: int = 3,
        session_mult: float = 1.0,
        correlation_mult: float = 1.0,
        drawdown_mult: float = 1.0
    ) -> Decimal:
        """
        Calculate position size based on risk management.
        
        v27.12.13: Added session, correlation, and drawdown multipliers.
        """
        # Base risk amount
        base_risk = Decimal(str(balance)) * Decimal(str(self.config.risk_per_trade))
        
        # Apply multipliers
        adjusted_risk = base_risk * Decimal(str(session_mult)) * Decimal(str(correlation_mult)) * Decimal(str(drawdown_mult))
        
        # Distance to stop loss
        sl_distance = abs(Decimal(str(entry_price)) - Decimal(str(stop_loss)))
        sl_pct = sl_distance / Decimal(str(entry_price))
        
        if sl_pct <= 0:
            return self.min_size
        
        # Position value we can afford to lose risk_amount at sl_pct
        position_value = adjusted_risk / sl_pct
        
        # Apply leverage
        margin_required = position_value / Decimal(str(leverage))
        
        # Convert to contracts
        contract_value_dec = Decimal(str(contract_value)) * Decimal(str(entry_price))
        size = position_value / contract_value_dec
        
        # Round down to lot size (0.1)
        size = (size / Decimal("0.1")).quantize(Decimal("1"), rounding=ROUND_DOWN) * Decimal("0.1")
        
        return max(size, self.min_size)
    
    async def execute_signal(self, signal: Dict) -> Optional[Dict]:
        """
        Execute a trading signal.
        
        v27.12.13: Added session timing, correlation, and drawdown checks.
        """
        if not self.client or not self.config.auto_trade_enabled:
            return None
        
        symbol = signal.get("symbol", "")
        direction = signal.get("direction", "").upper()
        entry = signal.get("entry")
        sl = signal.get("sl")
        tp1 = signal.get("tp1")
        tp2 = signal.get("tp2")
        confidence = signal.get("confidence", 0)
        grade = signal.get("grade", "C")
        
        # Validation
        if not all([symbol, direction, entry, sl, tp1]):
            self.logger.error(f"Invalid signal: missing required fields")
            return None
        
        if direction not in ["LONG", "SHORT"]:
            self.logger.error(f"Invalid direction: {direction}")
            return None
        
        # Check grade threshold
        min_grade = os.getenv("AUTO_TRADE_MIN_GRADE", "B")
        valid_grades = {"A": 3, "B": 2, "C": 1, "D": 0}
        if valid_grades.get(grade, 0) < valid_grades.get(min_grade, 2):
            self.logger.info(f"Signal grade {grade} below minimum {min_grade} - skipping")
            return None
        
        try:
            # v27.12.13: Update balance for drawdown check
            await self._update_balance()
            
            # Check drawdown protection
            drawdown_mult, drawdown_status = get_drawdown_multiplier(self.current_balance, self.peak_balance)
            if drawdown_status == 'stopped':
                self.logger.warning(f"❌ Trading paused due to drawdown ({(1 - self.current_balance/self.peak_balance)*100:.1f}%)")
                return None
            elif drawdown_status != 'normal':
                self.logger.warning(f"⚠️ Drawdown {drawdown_status}: reducing position size to {drawdown_mult*100:.0f}%")
            
            # Get current session multiplier
            session_name, session_mult = get_current_session()
            self.logger.info(f"📍 Session: {session_name} (mult: {session_mult})")
            
            # Get correlation reduction
            open_positions = await self._get_open_position_symbols()
            correlation_mult = get_correlation_reduction(symbol, open_positions)
            if correlation_mult < 1.0:
                self.logger.info(f"📊 Correlation reduction: {correlation_mult*100:.0f}% (open positions: {open_positions})")
            
            # Get account balance
            available = self.current_balance
            
            if available < 10:
                self.logger.warning(f"Insufficient balance: ${available}")
                return None
            
            # Get instrument info
            inst_id = self.client._convert_symbol(symbol)
            if inst_id not in self.client._instruments:
                await self.client.get_instruments(inst_id)
            
            inst_info = self.client._instruments.get(inst_id, {})
            contract_value = float(inst_info.get("contractValue", "0.001"))
            min_size = float(inst_info.get("minSize", "0.1"))
            tick_size = float(inst_info.get("tickSize", "0.1"))
            
            # Set leverage
            leverage = min(self.config.default_leverage, self.config.max_leverage)
            await self.client.set_leverage(
                symbol,
                leverage,
                self.config.margin_mode
            )
            
            # Calculate position size with all multipliers
            size = self._calculate_position_size(
                available,
                float(entry),
                float(sl),
                contract_value,
                leverage,
                session_mult,
                correlation_mult,
                drawdown_mult
            )
            
            # Ensure minimum size
            if float(size) < min_size:
                size = Decimal(str(min_size))
            
            # Determine order side
            side = "buy" if direction == "LONG" else "sell"
            
            # Generate client order ID
            client_id = f"grok_{symbol.replace('/', '')}_{int(datetime.now().timestamp())}"
            
            self.logger.info(
                f"📊 Executing {direction} signal:\n"
                f"   Symbol: {symbol}\n"
                f"   Size: {size} contracts\n"
                f"   Entry: ${entry}\n"
                f"   SL: ${sl}\n"
                f"   TP1: ${tp1}\n"
                f"   Leverage: {leverage}x\n"
                f"   Session: {session_name} ({session_mult}x)\n"
                f"   Correlation: {correlation_mult:.2f}x\n"
                f"   Drawdown: {drawdown_status} ({drawdown_mult}x)"
            )
            
            # Place market order with TP/SL
            order_result = await self.client.place_order(
                symbol=symbol,
                side=side,
                size=str(size),
                order_type="market",
                margin_mode=self.config.margin_mode,
                position_side="net",
                tp_trigger_price=str(tp1),
                tp_order_price="-1",  # Market price for TP
                sl_trigger_price=str(sl),
                sl_order_price="-1",  # Market price for SL
                client_order_id=client_id
            )
            
            if order_result.get("code") == "0" or order_result.get("orderId"):
                order_id = order_result.get("orderId")
                self.logger.info(f"✅ Order placed successfully: {order_id}")
                
                # Track the trade
                trade_info = {
                    "order_id": order_id,
                    "client_id": client_id,
                    "symbol": symbol,
                    "direction": direction,
                    "size": str(size),
                    "entry": entry,
                    "sl": sl,
                    "tp1": tp1,
                    "tp2": tp2,
                    "leverage": leverage,
                    "confidence": confidence,
                    "grade": grade,
                    "session": session_name,
                    "session_mult": session_mult,
                    "correlation_mult": correlation_mult,
                    "drawdown_mult": drawdown_mult,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                self.executed_trades[order_id] = trade_info
                return trade_info
            else:
                self.logger.error(f"Order failed: {order_result}")
                return None
                
        except BlofinAPIError as e:
            self.logger.error(f"Signal execution failed: {e}")
            # Provide specific guidance for common errors
            if e.code == "152013":
                self.logger.error(
                    "💡 FIX: Set BLOFIN_BROKER_ID environment variable:\n"
                    "   - For Transaction API Keys: BLOFIN_BROKER_ID='' (empty)\n"
                    "   - For Broker API Keys: BLOFIN_BROKER_ID='your_broker_id'"
                )
            return None
        except Exception as e:
            self.logger.error(f"Signal execution failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    async def check_positions(self) -> List[Dict]:
        """Check current positions"""
        if not self.client:
            return []
        
        try:
            return await self.client.get_positions()
        except Exception as e:
            self.logger.error(f"Failed to check positions: {e}")
            return []
    
    async def close_all_positions(self) -> Dict[str, bool]:
        """Close all open positions"""
        if not self.client:
            return {}
        
        results = {}
        positions = await self.check_positions()
        
        for pos in positions:
            inst_id = pos.get("instId")
            if inst_id and float(pos.get("positions", 0)) != 0:
                try:
                    await self.client.close_position(
                        symbol=inst_id,
                        margin_mode=pos.get("marginMode", "isolated"),
                        position_side=pos.get("positionSide", "net")
                    )
                    results[inst_id] = True
                    self.logger.info(f"Closed position: {inst_id}")
                except Exception as e:
                    results[inst_id] = False
                    self.logger.error(f"Failed to close {inst_id}: {e}")
        
        return results


# ============================================================================
# EXCEPTIONS
# ============================================================================
class BlofinAPIError(Exception):
    """Blofin API Error"""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================
_auto_trader: Optional[BlofinAutoTrader] = None

async def get_auto_trader() -> BlofinAutoTrader:
    """Get or create the auto-trader singleton"""
    global _auto_trader
    if _auto_trader is None:
        _auto_trader = BlofinAutoTrader()
        await _auto_trader.initialize()
    return _auto_trader


async def execute_trade_signal(signal: Dict) -> Optional[Dict]:
    """
    Convenience function to execute a signal.
    
    Usage from main.py:
        from bot.blofin_trader import execute_trade_signal
        
        result = await execute_trade_signal({
            "symbol": "BTC/USDT",
            "direction": "LONG",
            "entry": 43000,
            "sl": 42000,
            "tp1": 45000,
            "confidence": 75,
            "grade": "B"
        })
    """
    trader = await get_auto_trader()
    return await trader.execute_signal(signal)


# ============================================================================
# CLI TESTING
# ============================================================================
async def test_connection():
    """Test Blofin connection"""
    config = BlofinConfig()
    
    print("=" * 60)
    print("Blofin Connection Test v27.12.13")
    print("=" * 60)
    
    if not config.is_configured():
        print("❌ Blofin API not configured")
        print("Required environment variables:")
        print("  - BLOFIN_API_KEY")
        print("  - BLOFIN_SECRET_KEY")
        print("  - BLOFIN_PASSPHRASE")
        print("\nOptional:")
        print("  - BLOFIN_BROKER_ID (leave empty for Transaction API Keys)")
        return
    
    print(f"Broker ID: {'(empty)' if not config.broker_id else config.broker_id}")
    print(f"Demo Mode: {config.demo_mode}")
    print(f"Base URL: {config.base_url}")
    print()
    
    async with BlofinClient(config) as client:
        print("Testing Blofin API connection...")
        
        # Test balance
        balance = await client.get_balance()
        print(f"✅ Account Balance: ${balance.get('totalEquity', '0')}")
        
        # Test ticker
        ticker = await client.get_ticker("BTC/USDT")
        if ticker:
            print(f"✅ BTC/USDT Price: ${ticker.get('last', '0')}")
        
        # Test positions
        positions = await client.get_positions()
        print(f"✅ Open Positions: {len(positions)}")
        
        # Test session
        session_name, session_mult = get_current_session()
        print(f"✅ Current Session: {session_name} (multiplier: {session_mult})")
        
        print("\n✅ All tests passed!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_connection())

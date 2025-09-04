import subprocess
import os
import time
import MetaTrader5 as mt5
from datetime import datetime
from typing import Optional, Dict, Any

def create_live_config(expert_name, expert_params_file, symbol, timeframe="M1", 
                      auto_trading=True, allow_dll=True, allow_live_trading=True):
    """
    Cria arquivo de configuração para trading ao vivo no MT5
    
    Args:
        expert_name: Nome do Expert Advisor (ex: "MyEA")
        expert_params_file: Caminho para arquivo .set com parâmetros
        symbol: Símbolo para trading (ex: "EURUSD", "WIN$N")
        timeframe: Timeframe do gráfico (ex: "M1", "M5", "H1")
        auto_trading: Habilitar auto trading
        allow_dll: Permitir DLLs
        allow_live_trading: Permitir trading ao vivo
    
    Returns:
        str: Caminho do arquivo de configuração criado
    """
    auto_trading_flag = "1" if auto_trading else "0"
    dll_flag = "1" if allow_dll else "0"
    live_trading_flag = "1" if allow_live_trading else "0"
    
    config_content = f"""[Common]
AutoTrading={auto_trading_flag}
AllowDllImports={dll_flag}
AllowLiveTrading={live_trading_flag}

[Experts]
AllowLiveTrading={live_trading_flag}
AllowDllImport={dll_flag}
Enabled=1
[StartUp]
Expert={expert_name}
ExpertParameters={expert_params_file}
Symbol={symbol}
Period={timeframe}
"""
    
    config_path = f"live_trading_config_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ini"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path

def run_mt5_live_ea(expert_name: str, expert_params_file: str, symbol: str, 
                   timeframe: str = "M1", mt5_path: Optional[str] = None,
                   auto_trading: bool = True, allow_dll: bool = True,
                   monitoring_duration: int = 0, check_interval: int = 30) -> Dict[str, Any]:
    """
    Executa um Expert Advisor no MT5 para trading ao vivo
    
    Args:
        expert_name: Nome do Expert Advisor
        expert_params_file: Caminho para arquivo .set com parâmetros
        symbol: Símbolo para trading
        timeframe: Timeframe do gráfico
        mt5_path: Caminho para o MT5 (None para usar padrão)
        auto_trading: Habilitar auto trading
        allow_dll: Permitir DLLs
        monitoring_duration: Duração de monitoramento em segundos (0 = infinito)
        check_interval: Intervalo entre verificações em segundos
    
    Returns:
        dict: Status da execução e informações do EA
    """
    
    if mt5_path is None:
        mt5_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    
    # Criar configuração para live trading
    config_path = create_live_config(
        expert_name=expert_name,
        expert_params_file=expert_params_file,
            symbol=symbol,
            timeframe=timeframe,
            auto_trading=auto_trading,
            allow_dll=allow_dll
        )
                
    print(f"🚀 Iniciando MT5 Live Trading...")
    print(f"   Expert: {expert_name}")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Parameters: {expert_params_file}")
    print(f"   Config: {config_path}")
        
    # Executar MT5 com configuração
    command = f'"{mt5_path}" /config:{os.path.abspath(config_path)}'
    subprocess.Popen(command, shell=True)
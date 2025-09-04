import subprocess
import os
from datetime import datetime

def create_backtest_config(expert_name, expert_params_file, symbol, start_date, end_date, 
                          timeframe="M1", model=2, deposit=100_000, optimization=False):
    if optimization:
        optimization = "2"
    else:
        optimization = "0"

    config_content = f"""[Tester]
Expert={expert_name}
ExpertParameters={expert_params_file}
Symbol={symbol}
Period={timeframe}
Model={model}
Deposit={deposit}
FromDate={start_date}
ToDate={end_date}
Report=backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}
Optimization={optimization}
ShutdownTerminal=1
OptimizationCriterion
"""
    
    config_path = f"backtest_config_{symbol}.ini"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path

def run_mt5_backtest(expert_name, expert_params_file, symbol, start_date, end_date, timeframe="M1", model=2, deposit=100_000, optimization=False):
    # Criar arquivo de configuração
    config_path = create_backtest_config(expert_name, expert_params_file, symbol, start_date, end_date, timeframe, model, deposit, optimization)
    
    # Executar MT5 com configuração
    mt5_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    command = f'"{mt5_path}" /config:{os.path.abspath(config_path)}'
    
    subprocess.run(command, shell=True)
    
    # Limpar arquivo temporário
    os.remove(config_path)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class AnaliseFinanciamentoVsAluguel:
    def __init__(self, parametros):
        self.param = parametros
        self.param.setdefault('amortizacao_extra_anual', 0)
        self.validar_parametros()
        self.taxa_financiamento_mensal = self.param['taxa_financiamento'] / 100 / 12
        self.taxa_investimentos_mensal = self.param['rendimento_investimentos'] / 100 / 12
        self.taxa_fgts_mensal = self.param['rendimento_fgts'] / 100 / 12
        self.taxa_inflacao_mensal = self.param['taxa_inflacao'] / 100 / 12
        self.taxa_reajuste_aluguel_mensal = (1 + self.param['taxa_reajuste_aluguel'] / 100) ** (1 / 12) - 1

    def validar_parametros(self):
        if self.param['valor_fgts'] > self.param['valor_entrada']:
            raise ValueError("O FGTS não pode ser maior que o valor da entrada.")

    def calcular_tabela_financiamento(self):
        saldo_devedor = self.param['valor_imovel'] - self.param['valor_entrada']
        num_parcelas = self.param['tempo_financiamento'] * 12
        amortizacao = saldo_devedor / num_parcelas
        dados = []
        amortizacao_extra = self.param.get('amortizacao_extra_anual', 0)

        for mes in range(1, num_parcelas + 1):
            juros = saldo_devedor * self.taxa_financiamento_mensal
            parcela = amortizacao + juros
            saldo_devedor -= amortizacao
            
            # Amortização extra a cada 12 meses
            if amortizacao_extra > 0 and mes % 12 == 0 and saldo_devedor > 0:
                amortizar = min(amortizacao_extra, saldo_devedor)
                saldo_devedor -= amortizar
            
            # Better handling of final balance
            if saldo_devedor < 1:  # Very small amount due to rounding
                saldo_devedor = 0
                
            dados.append([mes, parcela, juros, amortizacao, saldo_devedor])
            
            if saldo_devedor <= 0:
                break

        df = pd.DataFrame(dados, columns=['Mes', 'Parcela', 'Juros', 'Amortizacao', 'Saldo_Devedor'])
        return df

    def simular_aluguel(self):
        tempo = self.param['tempo_financiamento'] * 12
        aluguel = self.param['aluguel_inicial']
        dados = []

        for mes in range(1, tempo + 1):
            if mes > 1 and (mes - 1) % 12 == 0:
                aluguel *= (1 + self.param['taxa_reajuste_aluguel'] / 100)
            dados.append([mes, aluguel])

        df = pd.DataFrame(dados, columns=['Mes', 'Aluguel'])
        return df

    def calcular_investimento_aluguel_economizado(self, tabela_aluguel):
        """
        Calcula o investimento dos aluguéis economizados no cenário do financiamento
        """
        tempo = self.param['tempo_financiamento'] * 12
        saldo_investimento = 0
        dados = []
        amortizacao_extra = self.param.get('amortizacao_extra_anual', 0)

        for mes in range(1, tempo + 1):
            aluguel = tabela_aluguel.loc[tabela_aluguel['Mes'] == mes, 'Aluguel'].values[0]
            
            # Aplica rendimento ao saldo existente
            saldo_investimento = saldo_investimento * (1 + self.taxa_investimentos_mensal)
            
            # Adiciona o aluguel economizado ao investimento
            saldo_investimento += aluguel
            
            # Adiciona amortização extra anual equivalente
            if amortizacao_extra > 0 and mes % 12 == 0:
                saldo_investimento += amortizacao_extra
            
            dados.append([mes, saldo_investimento])

        df = pd.DataFrame(dados, columns=['Mes', 'Investimento_Aluguel_Economizado'])
        return df

    def calcular_investimento_alternativo(self, tabela_financiamento, tabela_aluguel):
        """
        CORRECTED: Proper investment logic for rent scenario.
        Person keeps down payment invested, pays rent, invests savings vs financing,
        AND invests the interest that would be paid in financing (opportunity cost).
        """
        tempo = self.param['tempo_financiamento'] * 12
        
        # Initial investment amounts
        investimento_fgts = self.param['valor_fgts']
        investimento_proprio = self.param['valor_entrada'] - self.param['valor_fgts'] + self.param.get('custo_financiamento', 0)
        
        saldo_fgts = investimento_fgts
        saldo_proprio = investimento_proprio
        dados = []
        amortizacao_extra = self.param.get('amortizacao_extra_anual', 0)

        # Find when financing would end
        if len(tabela_financiamento) > 0:
            meses_amortizacao = len(tabela_financiamento)
        else:
            meses_amortizacao = tempo

        for mes in range(1, tempo + 1):
            aluguel = tabela_aluguel.loc[tabela_aluguel['Mes'] == mes, 'Aluguel'].values[0]
            
            # Get financing payment and interest for comparison (0 if loan is paid off)
            if mes <= len(tabela_financiamento):
                parcela = tabela_financiamento.loc[tabela_financiamento['Mes'] == mes, 'Parcela'].values[0]
                juros = tabela_financiamento.loc[tabela_financiamento['Mes'] == mes, 'Juros'].values[0]
            else:
                parcela = 0
                juros = 0
            
            # Apply investment returns
            saldo_proprio = saldo_proprio * (1 + self.taxa_investimentos_mensal)
            saldo_fgts = saldo_fgts * (1 + self.taxa_fgts_mensal)
            
            # Pay rent
            saldo_proprio -= aluguel
            
            # CORRECTED: Invest the difference between financing payment and rent
            if parcela > aluguel:
                # Invest only the amortização portion of the difference.
                # (parcela - juros) is the amortização efetiva naquela prestação.
                saldo_proprio += (parcela - aluguel - juros)
            
            # Invest full interest that would ter sido pago (savings), independentemente do aluguel
            saldo_proprio += juros
            
            # Add extra investment equivalent to extra amortization
            if amortizacao_extra > 0 and mes % 12 == 0 and mes <= meses_amortizacao:
                saldo_proprio += amortizacao_extra
            
            dados.append([mes, saldo_proprio, saldo_fgts])

        df = pd.DataFrame(dados, columns=['Mes', 'Investimento_Proprio', 'Investimento_FGTS'])
        return df

    def calcular_valor_real(self, valor, mes):
        return valor / ((1 + self.taxa_inflacao_mensal) ** mes)

    def calcular_rendimento_real(self):
        return ((1 + self.taxa_investimentos_mensal) / (1 + self.taxa_inflacao_mensal)) - 1

    def calcular_valorizacao_real_imovel(self):
        return ((1 + self.param['valorizacao_anual_imovel'] / 100) / (1 + self.param['taxa_inflacao'] / 100)) - 1

    def _expandir_dataframe(self, df, total_meses, colunas_valores, valores_default):
        """Helper method to expand dataframes to match time periods"""
        if df['Mes'].max() >= total_meses:
            return df
            
        last_row = df.iloc[-1].copy()
        rows_to_add = total_meses - df['Mes'].max()
        new_rows = []
        
        for i in range(1, rows_to_add + 1):
            row = last_row.copy()
            row['Mes'] = df['Mes'].max() + i
            
            if valores_default is not None:
                for col, val in zip(colunas_valores, valores_default):
                    row[col] = val
            
            new_rows.append(row)
        
        return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    def gerar_grafico_comparativo(self, financiamento, aluguel, investimento, investimento_aluguel_economizado):
        plt.figure(figsize=(16, 12))

        # Align vectors
        total_meses = max(financiamento['Mes'].max(), aluguel['Mes'].max(), investimento['Mes'].max())
        
        # Expand dataframes if necessary
        financiamento = self._expandir_dataframe(financiamento, total_meses, 
                                               ['Parcela', 'Juros', 'Amortizacao', 'Saldo_Devedor'], 
                                               [0, 0, 0, 0])
        investimento = self._expandir_dataframe(investimento, total_meses, 
                                              ['Investimento_Proprio', 'Investimento_FGTS'], 
                                              None)  # Keep last values
        aluguel = self._expandir_dataframe(aluguel, total_meses, ['Aluguel'], None)
        investimento_aluguel_economizado = self._expandir_dataframe(investimento_aluguel_economizado, total_meses, 
                                                                   ['Investimento_Aluguel_Economizado'], None)

        tempo = financiamento['Mes']

        # CORRECTED: Property value minus remaining debt minus interest costs paid PLUS rent savings invested
        valor_imovel_atual = self.param['valor_imovel'] * (1 + self.param['valorizacao_anual_imovel'] / 100) ** (financiamento['Mes'] / 12)
        juros_pagos_acumulados = financiamento['Juros'].cumsum()
        patrimonio_liquido_financiamento = (valor_imovel_atual - financiamento['Saldo_Devedor'] - 
                                          juros_pagos_acumulados + investimento_aluguel_economizado['Investimento_Aluguel_Economizado'] - self.param.get('custo_financiamento', 0))

        plt.subplot(2, 3, 1)
        plt.plot(tempo, patrimonio_liquido_financiamento, label='Patrimônio Líquido Financiamento', linewidth=2)
        plt.plot(tempo, investimento['Investimento_Proprio'] + investimento['Investimento_FGTS'], 
                label='Investimento Aluguel', linewidth=2)
        plt.title('Evolução Patrimonial COMPLETA')
        plt.xlabel('Meses')
        plt.ylabel('Valor (R$)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Monthly payments
        plt.subplot(2, 3, 2)
        plt.plot(tempo, financiamento['Parcela'], label='Parcela Financiamento')
        plt.plot(tempo, aluguel['Aluguel'], label='Aluguel')
        plt.title('Pagamentos Mensais')
        plt.xlabel('Meses')
        plt.ylabel('Valor (R$)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Components breakdown for financing
        plt.subplot(2, 3, 3)
        plt.plot(tempo, valor_imovel_atual, label='Valor do Imóvel', alpha=0.8)
        plt.plot(tempo, investimento_aluguel_economizado['Investimento_Aluguel_Economizado'], 
                label='Aluguéis Economizados Investidos', alpha=0.8)
        plt.plot(tempo, -juros_pagos_acumulados, label='Juros Pagos (negativo)', alpha=0.8)
        plt.title('Componentes do Financiamento')
        plt.xlabel('Meses')
        plt.ylabel('Valor (R$)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Accumulated costs
        custo_inicial_financiamento = self.param['valor_entrada'] + self.param.get('custo_financiamento', 0)
        custo_acumulado_financiamento = financiamento['Parcela'].cumsum() + custo_inicial_financiamento
        custo_acumulado_aluguel = aluguel['Aluguel'].cumsum()
        
        plt.subplot(2, 3, 4)
        plt.plot(tempo, custo_acumulado_financiamento, label='Total Pago no Financiamento')
        plt.plot(tempo, custo_acumulado_aluguel, label='Total Pago no Aluguel')
        plt.title('Custos Acumulados')
        plt.xlabel('Meses')
        plt.ylabel('Valor (R$)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Investment components for rental scenario
        plt.subplot(2, 3, 5)
        plt.plot(tempo, investimento['Investimento_Proprio'], label='Investimento Próprio', alpha=0.8)
        plt.plot(tempo, investimento['Investimento_FGTS'], label='Investimento FGTS', alpha=0.8)
        plt.title('Componentes do Aluguel')
        plt.xlabel('Meses')
        plt.ylabel('Valor (R$)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Final comparison
        plt.subplot(2, 3, 6)
        patrimonio_final = [patrimonio_liquido_financiamento.iloc[-1],
                            investimento.iloc[-1]['Investimento_Proprio'] + investimento.iloc[-1]['Investimento_FGTS']]
        colors = ['green' if patrimonio_final[0] > patrimonio_final[1] else 'red', 
                 'red' if patrimonio_final[0] > patrimonio_final[1] else 'green']
        bars = plt.bar(['Financiamento', 'Aluguel'], patrimonio_final, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, patrimonio_final):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(patrimonio_final)*0.01,
                    f'R$ {value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Comparação Final de Patrimônio')
        plt.ylabel('Valor (R$)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def executar_analise_completa(self, plot=True):
        financiamento = self.calcular_tabela_financiamento()
        aluguel = self.simular_aluguel()
        investimento = self.calcular_investimento_alternativo(financiamento, aluguel)
        investimento_aluguel_economizado = self.calcular_investimento_aluguel_economizado(aluguel)

        tempo = self.param['tempo_financiamento'] * 12
        valor_imovel_final = self.param['valor_imovel'] * (1 + self.param['valorizacao_anual_imovel'] / 100) ** self.param['tempo_financiamento']
        valor_imovel_real = self.calcular_valor_real(valor_imovel_final, tempo)

        investimento_final = investimento.iloc[-1]['Investimento_Proprio'] + investimento.iloc[-1]['Investimento_FGTS']
        investimento_real = self.calcular_valor_real(investimento_final, tempo)

        # CORRECTED: After financing ends, property is fully owned but subtract total interest costs AND add rent savings
        total_juros_pagos = financiamento['Juros'].sum()
        total_aluguel_economizado_investido = investimento_aluguel_economizado.iloc[-1]['Investimento_Aluguel_Economizado']
        patrimonio_financiamento = valor_imovel_final - total_juros_pagos + total_aluguel_economizado_investido - self.param.get('custo_financiamento', 0)
        patrimonio_financiamento_real = self.calcular_valor_real(patrimonio_financiamento, tempo)

        patrimonio_aluguel = investimento_final
        patrimonio_aluguel_real = investimento_real

        recomendacao = "FINANCIE" if patrimonio_financiamento_real > patrimonio_aluguel_real else "CONTINUE NO ALUGUEL"

        if plot:
            self.gerar_grafico_comparativo(financiamento, aluguel, investimento, investimento_aluguel_economizado)

        # Additional metrics
        total_custo_financiamento = financiamento['Parcela'].sum() + self.param['valor_entrada'] + self.param.get('custo_financiamento', 0)
        total_custo_aluguel = aluguel['Aluguel'].sum()
        diferenca_patrimonio = patrimonio_financiamento_real - patrimonio_aluguel_real
        diferenca_percentual = (diferenca_patrimonio / max(patrimonio_financiamento_real, patrimonio_aluguel_real)) * 100

        print("=== RELATÓRIO ANÁLISE COMPLETA E SIMÉTRICA ===")
        print("LÓGICA APLICADA:")
        print("• Financiamento: Valor imóvel - Juros pagos + Aluguéis economizados investidos")
        print("• Aluguel: Investe entrada + diferenças + juros economizados - aluguéis pagos")
        print("• COMPARAÇÃO TOTALMENTE SIMÉTRICA E JUSTA")
        print("==========================================================")
        print(f"Patrimônio Final (Financiamento): R$ {patrimonio_financiamento:,.2f} | Real: R$ {patrimonio_financiamento_real:,.2f}")
        print(f"  └─ Valor do Imóvel: R$ {valor_imovel_final:,.2f}")
        print(f"  └─ Juros Pagos: -R$ {total_juros_pagos:,.2f}")
        print(f"  └─ Aluguéis Economizados Investidos: +R$ {total_aluguel_economizado_investido:,.2f}")
        print(f"  └─ Custos de Financiamento: -R$ {self.param.get('custo_financiamento', 0):,.2f}")
        print()
        print(f"Patrimônio Final (Aluguel): R$ {patrimonio_aluguel:,.2f} | Real: R$ {patrimonio_aluguel_real:,.2f}")
        print()
        print(f"Diferença: R$ {diferenca_patrimonio:,.2f} ({diferenca_percentual:.1f}%)")
        print(f"Total Gasto (Financiamento): R$ {total_custo_financiamento:,.2f}")
        print(f"Total Gasto (Aluguel): R$ {total_custo_aluguel:,.2f}")
        print(f"Aluguéis Economizados e Investidos: R$ {total_aluguel_economizado_investido:,.2f}")
        print(f"Recomendação: {recomendacao}")
        print("===============================================")

        return {
            'financiamento': financiamento,
            'aluguel': aluguel,
            'investimento': investimento,
            'investimento_aluguel_economizado': investimento_aluguel_economizado,
            'patrimonio_financiamento': patrimonio_financiamento,
            'patrimonio_financiamento_real': patrimonio_financiamento_real,
            'patrimonio_aluguel': patrimonio_aluguel,
            'patrimonio_aluguel_real': patrimonio_aluguel_real,
            'diferenca_patrimonio': diferenca_patrimonio,
            'diferenca_percentual': diferenca_percentual,
            'total_custo_financiamento': total_custo_financiamento,
            'total_custo_aluguel': total_custo_aluguel,
            'total_juros_pagos': total_juros_pagos,
            'total_aluguel_economizado_investido': total_aluguel_economizado_investido,
            'recomendacao': recomendacao
        }

    def analise_sensibilidade(self):
        import copy
        parametros_base = self.param.copy()
        ranges = {
            'valor_entrada': np.linspace(parametros_base['valor_fgts'], parametros_base['valor_imovel'], 50),
            'rendimento_investimentos': np.linspace(0, 20, 20),
            'taxa_reajuste_aluguel': np.linspace(0, 15, 15),
            'valorizacao_anual_imovel': np.linspace(0, 15, 15),
        }
        nomes = {
            'valor_entrada': 'Valor da Entrada (R$)',
            'rendimento_investimentos': 'Rendimento dos Investimentos (%)',
            'taxa_reajuste_aluguel': 'Reajuste Anual do Aluguel (%)',
            'valorizacao_anual_imovel': 'Valorização Anual do Imóvel (%)',
        }
        
        fig, axs = plt.subplots(2, 2, figsize=(22, 12))
        axs = axs.flatten()
        
        for i, (param, valores) in enumerate(ranges.items()):
            patrimonio_fin = []
            patrimonio_alug = []
            
            for v in valores:
                p = copy.deepcopy(parametros_base)
                if param == 'valor_entrada':
                    p[param] = v
                    if p['valor_fgts'] > p['valor_entrada']:
                        continue
                else:
                    p[param] = v
                
                try:
                    sim = AnaliseFinanciamentoVsAluguel(p)
                    res = sim.executar_analise_completa(plot=False)
                    patrimonio_fin.append(res['patrimonio_financiamento_real'])
                    patrimonio_alug.append(res['patrimonio_aluguel_real'])
                except Exception as e:
                    patrimonio_fin.append(np.nan)
                    patrimonio_alug.append(np.nan)
            
            patrimonio_fin = np.array(patrimonio_fin)
            patrimonio_alug = np.array(patrimonio_alug)
            diferenca = patrimonio_fin - patrimonio_alug
            
            if not np.all(np.isnan(diferenca)):
                idx_otimo = np.nanargmax(diferenca)
                
                axs[i].plot(valores, diferenca, label='Diferença Financiamento - Aluguel')
                axs[i].scatter(valores[idx_otimo], diferenca[idx_otimo], color='red', marker='o', s=80, label='Ponto Ótimo')
                axs[i].set_title(f'Sensibilidade: {nomes[param]}')
                axs[i].set_xlabel(nomes[param])
                axs[i].set_ylabel('Diferença Patrimônio Real (R$)')
                axs[i].axhline(0, color='gray', linestyle='--', linewidth=1)
                axs[i].legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parametros = {
        'valor_imovel': 700_000,
        'valor_entrada': 350_000,
        'valor_fgts': 250_000,
        'taxa_financiamento': 13.5,
        'tempo_financiamento': 35,
        'custo_financiamento': 50_000,
        'aluguel_inicial': 3_800,
        'taxa_reajuste_aluguel': 5,
        'valorizacao_anual_imovel': 6,
        'rendimento_investimentos': 8,
        'rendimento_fgts': 4,
        'taxa_inflacao': 5,
        'amortizacao_extra_anual': 0
    }

    print("=== ANÁLISE FINANCEIRA COMPLETA E SIMÉTRICA ===")
    print("METODOLOGIA:")
    print("• Financiamento: Valor imóvel - Juros pagos + Aluguéis economizados investidos")
    print("• Aluguel: Investe entrada + diferenças + juros economizados - aluguéis pagos")
    print("• COMPARAÇÃO TOTALMENTE SIMÉTRICA E JUSTA")
    print("======================================================")
    
    simulacao = AnaliseFinanciamentoVsAluguel(parametros)
    resultados = simulacao.executar_analise_completa()
    simulacao.analise_sensibilidade() 
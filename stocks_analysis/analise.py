import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import os, re, sys
from datetime import datetime, date
from sklearn.preprocessing import MinMaxScaler
import mysql.connector as mysql

def file_reader(*args, **kwargs):
    filetype = kwargs.get('filetype', None)
    columns = kwargs.get('columns', None)
    sep = kwargs.get('sep', ',')
    try:
        if filetype == None:
            if args[0].__class__ == list:
                return [pd.read_csv(x, names=columns, sep=sep) for x in args[0]]
            else:
                return [pd.read_csv(x, names=columns, sep=sep) for x in args]
        elif filetype == 'xlsx' | filetype == 'excel':
            if args[0].__class__ == list:
                return [pd.read_excel(x, names=columns, sep=sep) for x in args[0]]
            else:
                return [pd.read_excel(x, names=columns, sep=sep) for x in args]
    except UnicodeDecodeError:
        if filetype == None:
            if args[0].__class__ == list:
                return [pd.read_csv(x, names=columns, sep=sep, encoding='ISO-8859-1') for x in args[0]]
            else:
                return [pd.read_csv(x, names=columns, sep=sep, encoding='ISO-8859-1') for x in args]
        elif filetype == 'xlsx' | filetype == 'excel':
            if args[0].__class__ == list:
                return [pd.read_excel(x, names=columns, sep=sep, encoding='ISO-8859-1') for x in args[0]]
            else:
                return [pd.read_excel(x, names=columns, sep=sep, encoding='ISO-8859-1') for x in args]

def retorno_log(data, window=1):
    return (np.log(data)-np.log(data).shift(window)).dropna()

def rent_anual(data, *,start_date=None, scale=False):
    try:
        if start_date == None:
            start_date = data.iloc[0,0]
            end_date = start_date.replace(year=start_date.year+1)
            r = [start_date, data.iloc[np.where(data.iloc[:,0] == end_date)[0][0], 1] - data.iloc[0, 1]]
        else:
            if type(start_date) != date:
                start_date = date.fromisoformat(start_date)
            else:
                pass
            end_date = start_date.replace(year=start_date.year + 1)
            r = [start_date, data.iloc[np.where(data.iloc[:,0] == end_date)[0][0], 1] - data.iloc[np.where(data.iloc[:,0] == start_date)[0][0], 1]]
    except IndexError:
        if start_date not in data.iloc[:,0].values:
            try:
                start_date = start_date.replace(day=start_date.day+1)
                r = [start_date, data.iloc[np.where(data.iloc[:,0] == end_date)[0][0], 1] - data.iloc[np.where(data.iloc[:,0] == start_date)[0][0], 1]]
            except IndexError:
                start_date = start_date.replace(day=start_date.day-2)
                r = [start_date, data.iloc[np.where(data.iloc[:,0] == end_date)[0][0], 1] - data.iloc[np.where(data.iloc[:,0] == start_date)[0][0], 1]]
        elif end_date not in data.iloc[:,0].values:
            try:
                end_date = end_date.replace(day=end_date.day+1)
                r = [start_date, data.iloc[np.where(data.iloc[:,0] == end_date)[0][0], 1] - data.iloc[np.where(data.iloc[:,0] == start_date)[0][0], 1]]
            except IndexError:
                end_date = end_date.replace(day=end_date.day-2)
                r = [start_date, data.iloc[np.where(data.iloc[:,0] == end_date)[0][0], 1] - data.iloc[np.where(data.iloc[:,0] == start_date)[0][0], 1]]
    return r

def main():
    #------------------------------------Estilo dos gráficos------------------------------------
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = 12,9

    #------------------------------------Lendo os Arquivos------------------------------------
    path = 'D:\\Case GCB'
    files = [(path + '\\' + file) for file in os.listdir(path) if file.endswith('.csv') == True]
    all_data = file_reader(files)
    azul, brlusd, bzf, goll4, latam, petr4, ibov = all_data

    #Nomes dos ativos
    names = [(re.split('\.', x)[0]).lower() for x in [file for file in os.listdir(path) if file.endswith('.csv') == True]]

    #------------------------------------Formatação------------------------------------
    
    #Datas
    #Azul
    azul.loc[:, ['Date']] = azul.loc[:,['Date']].apply(lambda x: datetime.fromisoformat(x[0]).date(), axis=1)
    #brlusd
    brlusd = brlusd.dropna()
    brlusd.loc[:, ['Date']] = brlusd.loc[:,['Date']].apply(lambda x: datetime.fromisoformat(x[0]).date(), axis=1)
    #Brent
    bzf = bzf.dropna()
    bzf.loc[:, ['Date']] = bzf.loc[:,['Date']].apply(lambda x: datetime.fromisoformat(x[0]).date(), axis=1)
    #Gol
    goll4.loc[:, ['Date']] = goll4.loc[:,['Date']].apply(lambda x: datetime.fromisoformat(x[0]).date(), axis=1)
    #Latam
    latam.loc[:, ['Date']] = latam.loc[:,['Date']].apply(lambda x: datetime.fromisoformat(x[0]).date(), axis=1)
    #Petrobras
    petr4.loc[:, ['Date']] = petr4.loc[:,['Date']].apply(lambda x: datetime.fromisoformat(x[0]).date(), axis=1)
    #Ibovespa
    ibov = ibov.dropna()
    ibov.loc[:, ['Date']] = ibov.loc[:,['Date']].apply(lambda x: datetime.fromisoformat(x[0]).date(), axis=1)

    #------------------------------------Criando DataFrames------------------------------------

    #DataFrame Close/Adj
    ativos_close =  brlusd.loc[:,['Date', 'Close']].merge(bzf.loc[:,['Date', 'Close']], on='Date')
    for df in [goll4, petr4, ibov]:
        ativos_close = ativos_close.merge(df.loc[:, ['Date','Adj Close']], on='Date')
    ativos_close = ativos_close.dropna()
    ativos_close.loc[:,['Close_x']] = np.divide(ativos_close['Close_y'], ativos_close['Close_x'])
    ativos_close.drop('Close_y', axis=1, inplace=True)
    ativos_close.columns=['Date','bzf (R$)', *[name for name in names if name not in ['azul4', 'brlusd=x', 'bz=f','ltmay']]]

    #Retorno (BZ=F (R$), GOLL4, PETR4, IBOV)
    janela = 1
    retorno = pd.merge(ativos_close.iloc[janela:,0], retorno_log(ativos_close.iloc[:,1:], window=janela), left_index=True, right_index=True, how="outer")

    #Retorno (AZUL, GOLL4, LATAM)
    azul['Retorno'] = retorno_log(azul['Adj Close'], window=janela)
    latam['Retorno'] = retorno_log(np.divide(latam['Adj Close'], brlusd['Close']), window=janela)
    retorno_goll4 = retorno.loc[:,['Date','goll4']].merge(azul.loc[:,['Date','Retorno']], on='Date').dropna()
    retorno_goll4 = retorno_goll4.merge(latam.loc[:,['Date', 'Retorno']], on='Date').dropna()
    retorno_goll4.columns = ['Date', 'goll4', 'azul', 'latam']
    
    #------------------------------------Comparação dos Retornos------------------------------------

    #Todos (BZ=F (R$), GOLL4, PETR4, IBOV)
    for x in retorno.columns:
        if x != 'Date':
            plt.plot(retorno['Date'], retorno[x])
            plt.xticks(rotation=45)
            plt.xlabel('Ano')
    plt.title('Log-Retorno 126 dias úteis')
    plt.legend(names)
    #plt.savefig('comparação log-retorno 126 dias')
    #plt.show()

    #Cia. Aéreas (GOLL4, LATAM, AZUL)
    for x in retorno_goll4.columns:
        if x != 'Date':
            plt.plot(retorno_goll4['Date'], retorno_goll4[x])
            plt.xticks(rotation=45)
            plt.xlabel('Ano')
    plt.title('Log-Retorno 126 dias úteis Cia. Aéreas')
    plt.legend(retorno_goll4.columns[1:])
    #plt.savefig('comparação log-retorno 126 dias cia aereas')
    #plt.show()

    #------------------------------------Scatter Retorno/Retorno------------------------------------

    #Todos (BZ=F (R$), GOLL4, PETR4, IBOV)
    fig, axs = plt.subplots(len(retorno.columns)-1,len(retorno.columns)-1)
    i = 0
    for x in retorno.columns[1:]:
        j=0
        for y in retorno.columns[1:]:
            if j < i:
                axs[i,j].scatter(retorno[x],retorno[y])
                axs[i,j].set_title(f'{x.lower()}/{y.lower()}', fontsize=9)
            else:
                pass
            j+=1
        i+=1
    fig.suptitle('Correlação Log-Retorno x Log-Retorno', fontsize=16)
    fig.tight_layout()
    #fig.savefig('scatter log-retorno')
    #fig.show()

    #Cia. Aéreas (GOLL4, LATAM, AZUL)
    fig2, axs2 = plt.subplots(len(retorno_goll4.columns)-1,len(retorno_goll4.columns)-1)
    i = 0
    for x in retorno_goll4.columns[1:]:
        j=0
        for y in retorno_goll4.columns[1:]:
            if j < i:
                axs2[i,j].scatter(retorno_goll4[x],retorno_goll4[y])
                axs2[i,j].set_title(f'{x.lower()}/{y.lower()}', fontsize=9)
            else:
                pass
            j+=1
        i+=1
    fig2.suptitle('Correlação Log-Retorno x Log-Retorno', fontsize=16)
    fig2.tight_layout()
    #fig2.savefig('scatter log-retorno cia aereas')
    #fig2.show()

    #------------------------------------Heatmap Correlação------------------------------------

    #Todos (BZ=F (R$), GOLL4, PETR4, IBOV)
    corr = np.corrcoef([retorno[x] for x in retorno.columns if x != 'Date'])
    hmap, ax = plt.subplots()
    ax.imshow(corr)
    ax.set_xticks(np.arange(len(retorno.columns)-1), labels=retorno.columns[1:])
    plt.xticks(rotation=45)
    ax.set_yticks(np.arange(len(retorno.columns)-1), labels=retorno.columns[1:])
    for i in range(len(retorno.columns)-1):
        for j in range(len(retorno.columns)-1):
            text = ax.text(j, i, round(corr[i, j],3),
                        ha="center", va="center", color="w")
    plt.title('Correlação (126 dias úteis)')
    hmap.tight_layout()
    #hmap.show()
    #hmap.savefig('corr heatmap 126 dias')

    #Cia. Aéreas (GOLL4, LATAM, AZUL)
    corr_goll4 = np.corrcoef([retorno_goll4[x] for x in retorno_goll4.columns if x != 'Date'])
    hmap2, ax2 = plt.subplots()
    ax2.imshow(corr_goll4)
    ax2.set_xticks(np.arange(len(retorno_goll4.columns)-1), labels=retorno_goll4.columns[1:])
    plt.xticks(rotation=45)
    ax2.set_yticks(np.arange(len(retorno_goll4.columns)-1), labels=retorno_goll4.columns[1:])
    for i in range(len(retorno_goll4.columns)-1):
        for j in range(len(retorno_goll4.columns)-1):
            text = ax2.text(j, i, round(corr_goll4[i, j],3),
                        ha="center", va="center", color="w")
    plt.title('Correlação (126 dias úteis) Cia. Aéreas')
    hmap2.tight_layout()
    #hmap2.show()
    #hmap2.savefig('corr heatmap 126 dias cia aereas')

    #------------------------------------Comparação PETR4 e GOLL4------------------------------------

    #Colocar Valores em Escala
    mms = MinMaxScaler()
    petr_gol = ativos_close[['Date', 'bzf (R$)', 'goll4', 'petr4']].reset_index(drop=True)
    for name in petr_gol.columns[1:]:
        petr_gol[name] = mms.fit_transform(np.reshape(petr_gol[name].values, (-1,1)))
    
    #Gráfico PETR4 e GOLL4 em escala
    datas = [ativos_close.loc[0,'Date']]
    for _ in range(5):
        datas.append(datas[-1].replace(year=datas[-1].year+1))

    img_aereas, ax3 = plt.subplots()
    for x in petr_gol.columns[1:]:
        ax3.plot(petr_gol['Date'], petr_gol[x])
    img_aereas.legend(petr_gol.columns[1:])
    ax3.set_title('Comparação Preço de Fechamento Cia. Aéreas (em escala)')
    plt.xlabel('Data')
    plt.xticks(rotation=45)
    [plt.axvline(data, color='k', ls='--') for data in datas]
    plt.ylabel('Preço de Fechamento em Escala')
    plt.tight_layout()
    #img_aereas.savefig('comparacao preco de fechamento cia aereas')
    
    #Rentabilidade Anual
    rent_df = pd.DataFrame(columns=['Date', 'bzf (R$)', 'goll4', 'petr4'])
    for year in ['2017','2018','2019','2020','2021']:
        for col in petr_gol.columns[1:]:
            d, r = rent_anual(petr_gol[['Date', col]], start_date=f'{year}-11-09')
            if d not in rent_df['Date'].values:
                rent_df.loc[len(rent_df), ['Date']] = d
                rent_df.loc[len(rent_df)-1, [col]] = float(f'{r*100:.2f}')
            else:
                rent_df.loc[(np.where(rent_df['Date'] == d))[0][0], [col]] = float(f'{r*100:.2f}')

    w = 0.25
    fig3 = plt.subplots()
    br1 = np.arange(len(rent_df.index))
    br2 = [x + w for x in br1]
    br3 = [x + w for x in br2]
    pos = [br1,br2,br3]
    for i in range(len(rent_df.columns)-1):
        plt.bar(pos[i], rent_df.iloc[:,i+1], width=w)

    plt.xlabel('Ano (Início/Final do Período)', fontweight ='bold', fontsize = 15)
    plt.ylabel('Rentabilidade (em %)', fontweight ='bold', fontsize = 15)
    plt.xticks([r + w for r in range(len(br1))], ['2017/2018','2018/2019','2019/2020','2020/2021','2021/2022'])
    plt.title('Rentabilidade Anual (em %)')
    plt.legend(rent_df.columns[1:])
    plt.tight_layout()
    #plt.savefig('rentabilidade cia aerea barplot')

if __name__ == '__main__':
    main()
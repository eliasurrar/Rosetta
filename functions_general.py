# Definicion funciones basicas =====================================

import pandas as pd
import numpy as np
import os

def clean_dataset(dataframe, replace_near0_with=np.nan):
    # dataframe = dataframe.replace(r'^([A-Za-z]|[0-9]|_)+$', np.nan, regex=True)  # reemplaza cualquier string con NaN
    dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    for i in dataframe.columns:
        dataframe[i] = np.where(
            (((dataframe[i] >= -0.0005) & (dataframe[i] < 0)) | ((dataframe[i] > 0) & (dataframe[i] <= 0.0005))),
            replace_near0_with,
            dataframe[i])
    #dataframe = dataframe.dropna(how='all')  # elimino filas con NaN
    return dataframe


def range_check(dataframe, columnas, min, max):
    for c in columnas:
        dataframe.loc[(dataframe[c] < min) | (dataframe[c] > max), c] = np.nan
    return dataframe


def weighted_mean(df, values, weights, groupby):
    df = df.copy()
    grouped = df.resample(groupby, offset='5H').sum()
    df['weighted_average'] = df[values] / grouped[weights].transform('sum') * df[weights]
    return grouped['weighted_average'].sum(min_count=1)  # min_count is required for Grouper objects

def weighted_average(df, float_cols, weight_col, group_every, offset=None): #probar funcionamiento, agregarle offset...
    df1 = df.copy(deep=True)
    df1 = df1.apply(pd.to_numeric, errors='coerce')
    float_cols.remove(weight_col) if weight_col in float_cols else None
    df2 = pd.DataFrame(data=[], columns=float_cols)#, index=pd.DataFrame(df1.resample(group_every).sum()).index)
    for i in float_cols:
        try:
            dfaux = df1.copy()
            dfaux['mass_wt'] = np.where(dfaux[i].notnull(), dfaux[weight_col] * dfaux[i], np.nan)
            # dfaux = dfaux[dfaux['mass_wt'].notna()]
            op = dfaux[[i, 'mass_wt', weight_col]]. \
                query(f'({i} == {i})'). \
                groupby(pd.Grouper(freq=group_every, offset=offset)). \
                agg(weightcol_sum=(weight_col, np.sum), weightcol_mean=(weight_col, np.mean),
                    masswt_sum=('mass_wt', np.sum), weightcol_count=(weight_col, 'count')) # masswt_count=('mass_wt', 'count')
            op['op'] = op['masswt_sum'] / op['weightcol_sum']
            df2[i] = op['op']
        except:
            pass
    df2[f'{weight_col}_mean'] = dfaux[[i, 'mass_wt', weight_col]]. \
        groupby(pd.Grouper(freq=group_every, offset=offset)). \
        agg(weightcol_mean=(weight_col, np.mean))
    df2[f'{weight_col}_sum'] = dfaux[[i, 'mass_wt', weight_col]]. \
        groupby(pd.Grouper(freq=group_every, offset=offset)). \
        agg(weightcol_sum=(weight_col, np.sum))
    df2[f'{weight_col}_count'] = dfaux[[i, 'mass_wt', weight_col]]. \
        groupby(pd.Grouper(freq=group_every, offset=offset)). \
        agg(weightcol_count=(weight_col, 'count'))  # 'masswt_count'   # df2[f'{weight_col}_count'] = op['weightcol_count'].copy(deep=True)
    # df2 = df2[df2[weight_col].notna()]
    print(i)
    return df2

def weighted_average_mincount(df, float_cols, weight_col, group_every, min_count=1, offset=None): #probar funcionamiento, agregarle offset...
    df1 = df.copy(deep=True)
    df1 = df1.apply(pd.to_numeric, errors='coerce')
    float_cols.remove(weight_col) if weight_col in float_cols else None
    df2 = pd.DataFrame(data=[], columns=float_cols)#, index=pd.DataFrame(df1.resample(group_every).sum()).index)
    nan = np.nan
    for i in float_cols:
        dfaux = df1.copy()
        dfaux['mass_wt'] = np.where(dfaux[i].notnull(), dfaux[weight_col] * dfaux[i], np.nan)
        #dfaux = dfaux[dfaux['mass_wt'].notna()]
        op = dfaux[[i, 'mass_wt', weight_col]]. \
            query(f'({i} == {i})'). \
            groupby(pd.Grouper(freq=group_every, offset=offset)). \
            agg(weightcol_sum=(weight_col, lambda x: x.sum(min_count=min_count)),
                weightcol_mean=(weight_col, np.mean),
                masswt_sum=('mass_wt', lambda x: x.sum(min_count=min_count)),
                masswt_count=('mass_wt', 'count')
                )
        op['op'] = op['masswt_sum'] / op['weightcol_sum']
        df2[i] = op['op']
    df2[f'{weight_col}_mean'] = op['weightcol_mean'].copy(deep=True)
    df2[f'{weight_col}_sum'] = op['weightcol_sum'].copy(deep=True)
    df2[f'{weight_col}_count'] = op['masswt_count'].copy(deep=True)
    #df2 = df2[df2[weight_col].notna()]
    print(i)
    return df2

def sumaproducto(df, float_cols, weight_col, period, offset=0):
    df2 = pd.DataFrame(data=[], columns=float_cols, index=df.groupby(level=0).sum().index)
    df2[weight_col] = df[weight_col].groupby(level=0).sum()
    for i in float_cols:
        sumaproducto = df.assign(sumprod=df[weight_col].mul(df[i]))[['sumprod']].groupby(level=0).sum()  # SE PUEDE AGREGAR MAS COLUMNAS ANTES DE .SUM
        df2[i] = sumaproducto.pop('sumprod').div(df2[weight_col])
    df3 = df2.resample(period).fillna(method='ffill')
    df3 = df3.shift(periods=offset, freq=period, fill_value=np.nan)
    return df3

def delete_repeated_values(df, cols):
    for c in cols:
        df[c] = df[c].mask(df.duplicated([c]))
    return df


def Mask_Or_Drop_Successive_Identical_Values(df, cols, drop=False, keep_first=True, axis=0, how='all', n_decimals=3):
    '''
    #Function built with the help of:
    # 1) https://stackoverflow.com/questions/48428173/how-to-change-consecutive-repeating-values-in-pandas-dataframe-series-to-nan-or
    # 2) https://stackoverflow.com/questions/19463985/pandas-drop-consecutive-duplicates

    Input:
    df should be a pandas.DataFrame of a a pandas.Series
    Output:
    df of ts with masked or droped values
    '''

    # Mask keeping the first occurence
    if keep_first:
        df[cols] = df[cols].mask(round(df[cols].shift(1), n_decimals) == round(df[cols], n_decimals))
    # Mask including the first occurence
    else:
        df[cols] = df[cols].mask((round(df[cols].shift(1), n_decimals)) == round(df[cols], n_decimals) |
                                 (round(df[cols].shift(-1), n_decimals) == round(df[cols]), n_decimals))

    # Drop the values (e.g. rows are deleted)
    if drop:
        return df[cols].dropna(axis=axis, how=how)
        # Only mask the values (e.g. become 'NaN')
    else:
        return df

'''
def fill_nan_with_mean_from_prev_and_next(df, cols):
    s = df[cols].isnull().cumsum()
    t1 = df[cols][(s == 1).shift(-1).fillna(False)].stack().reset_index(level=0, drop=True)
    t2 = df[cols].lookup(s.idxmax() + 1, s.idxmax().index)
    df[cols].fillna(t1 / (2 ** s) + t2 * (1 - 0.5 ** s) * 2 / 2)
    return df
'''

def remove_outliers(dataframe, cols, z_score=3):
    from scipy.stats import zscore
    #z_scores = np.array([])
    for c in cols:
        z_scores = zscore(dataframe[c], nan_policy='omit')
        dataframe[c] = dataframe[c][np.abs(z_scores) < z_score]
        #dataframe[c] = dataframe[c].mask(dataframe[c].sub(dataframe[c].mean()).div(dataframe[c].std()).abs().gt(veces_std))
    return dataframe

def remove_outliers_STD(dataframe, cols, veces_std=2.5):
    for c in cols:
        dataframe[c] = dataframe[c].mask(dataframe[c].sub(dataframe[c].mean()).div(dataframe[c].std()).abs().gt(veces_std))
    return dataframe



# https://stackoverflow.com/questions/40839609/rename-unnamed-multiindex-columns-in-pandas-dataframe
def rename_unnamed(df):
    """Rename unamed columns name for Pandas DataFrame
    See https://stackoverflow.com/questions/41221079/rename-multiindex-columns-in-pandas
    Parameters
    ----------
    df : pd.DataFrame object
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Output dataframe
    """
    if df.columns.nlevels > 1:
        df = df.rename(columns={np.nan: 'type'})
        for i, columns in enumerate(df.columns.levels):
            columns_new = columns.astype('str').tolist()
            for j, row in enumerate(columns_new):
                matches = ['Unnamed', 'unnamed', 'nan', 'NaN', 'type']
                if any(x in row for x in matches):
                    columns_new[j] = ""
            if pd.__version__ < "0.21.0":  # https://stackoverflow.com/a/48186976/716469
                df.columns.set_levels(columns_new, level=i, inplace=True)
            else:
                df = df.rename(columns=dict(zip(columns.astype('str').tolist(), columns_new)), level=i)
    else:
        pass

    # para eliminar cualquier numero dentro de los encabezados o nombres de columna
    if df.columns.nlevels > 0:
        for i, columns in enumerate(df.columns.levels):
            columns_new = columns.astype('str').tolist()
            for j, row in enumerate(columns_new):
                if pd.isna(pd.to_numeric(row, errors='coerce')) == False:  # isinstance(row, (int, float))
                    columns_new[j] = ""
            if pd.__version__ < "0.21.0":  # https://stackoverflow.com/a/48186976/716469
                df.columns.set_levels(columns_new, level=i, inplace=True)
            else:
                df = df.rename(columns=dict(zip(columns.tolist(), columns_new)), level=i) # saco .astype para eviar conertir numero a string
    else:
        pass
    return df

def replace_column_names(df, startswith):
    if df.columns.nlevels > 1:
        df = df.rename(columns={np.nan: 'type'})
        for i, columns in enumerate(df.columns.levels):
            columns_new = columns.astype('str').tolist()
            for j, row in enumerate(columns_new):
                matches = startswith
                if any(x in row for x in matches):
                    columns_new[j] = ""
            if pd.__version__ < "0.21.0":  # https://stackoverflow.com/a/48186976/716469
                df.columns.set_levels(columns_new, level=i, inplace=True)
            else:
                df = df.rename(columns=dict(zip(columns.astype('str').tolist(), columns_new)), level=i)
    else:
        pass
    return df


def normalize_chars(df, rename_unnamed_cols=True, remove_duplicated_cols=False):
    # https://stackoverflow.com/questions/49695477/removing-specific-duplicated-characters-from-a-string-in-python
    import re
    to_remove = "_.- " #characters to remove
    pattern = "(?P<char>[" + re.escape(to_remove) + "])(?P=char)+"
    if rename_unnamed_cols:
        df = rename_unnamed(df)
    else:
        pass
    if df.columns.nlevels > 1:
        df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]  # multiindex to single index
        df.columns = df.columns.str.normalize('NFKD').str.lower().str.encode('ascii', errors='ignore').str.decode('utf-8')
        df.columns = df.columns.str.replace(r"[^a-zA-Z0-9()%-+#]", "_", regex=True)
        df.columns = [x.strip("_") for x in df.columns]
        df.columns = [re.sub(pattern, r"\1", x) for x in df.columns] # [re.sub(r'(\S)\1+', '\\1', x) for x in df.columns]
        df = df.loc[:, ~df.columns.duplicated()] if remove_duplicated_cols else df
    else:
        df.columns = df.columns.str.normalize('NFKD').str.lower().str.encode('ascii', errors='ignore').str.decode('utf-8')
        df.columns = df.columns.str.replace(r"[^a-zA-Z0-9()%-+#]", "_", regex=True)
        df.columns = [x.strip("_") for x in df.columns]
        df.columns = [re.sub(pattern, r"\1", x) for x in df.columns] # [re.sub(r'(\S)\1+', '\\1', x) for x in df.columns]
        df = df.loc[:, ~df.columns.duplicated()] if remove_duplicated_cols else df
    return df


#%% Funciones de compilacion para bases de datos

def compile_ControlFino(id, files, date, sheet, file_path):
    import datetime as dt

    for x in enumerate(sheet, start=0):
        locals()['df_final%s' % '{:0>2}'.format(x[0])] = pd.DataFrame()
        locals()['df_Aux%s' % '{:0>2}'.format(x[0])] = pd.DataFrame()

    df_final = pd.DataFrame()

    for f, d in zip(files, date):
        for s, (a, b) in zip(sheet, enumerate(id)):
            #print(a)
            try:
                df_Aux = pd.read_excel(file_path + f,
                                       # usecols=list(range(col_iProgConc[0], col_iProgConc[0] + 31, 1)),
                                       # header=[4, 5],
                                       # index_col=[col_iDRT[0]+7],
                                       # dtype=np.float64,
                                       # nrows=1,
                                       sheet_name=s,
                                       engine=None
                                       )
                indexi, coli = np.where(df_Aux == b)
                try:
                    locals()['df_Aux%s' % '{:0>2}'.format(a)] = pd.read_excel(file_path + f,
                                                                                 # usecols=list(range(col_iProgConc[0], col_iProgConc[0] + 31, 1)),
                                                                                 header=[indexi[0] + 1, indexi[0] + 2],
                                                                                 # index_col=[col_iDRT[0]+7],
                                                                                 # dtype=np.float64,
                                                                                 # nrows=1,
                                                                                 sheet_name=s,
                                                                                 engine=None
                                                                                 )
                    locals()['df_Aux%s' % '{:0>2}'.format(a)].drop(locals()['df_Aux%s' % '{:0>2}'.format(a)].index[pd.Period(d, freq='M').end_time.date().day:], inplace=True)
                    locals()['df_Aux%s' % '{:0>2}'.format(a)] = locals()['df_Aux%s' % '{:0>2}'.format(a)].iloc[:, coli[0]:]
                    locals()['df_Aux%s' % '{:0>2}'.format(a)] = clean_dataset(locals()['df_Aux%s' % '{:0>2}'.format(a)], replace_near0_with=0)
                    locals()['df_Aux%s' % '{:0>2}'.format(a)] = normalize_chars(locals()['df_Aux%s' % '{:0>2}'.format(a)], rename_unnamed_cols=True)
                    locals()['df_Aux%s' % '{:0>2}'.format(a)].index = [d + dt.timedelta(days=x) for x in range(pd.Period(d, freq='M').end_time.date().day)]
                    locals()['df_Aux%s' % '{:0>2}'.format(a)] = locals()['df_Aux%s' % '{:0>2}'.format(a)].add_prefix(s.replace(' ', '_') + '_')
                    locals()['df_Aux%s' % '{:0>2}'.format(a)] = normalize_chars(locals()['df_Aux%s' % '{:0>2}'.format(a)], rename_unnamed_cols=False)
                    locals()['df_final%s' % '{:0>2}'.format(a)] = pd.concat([locals()['df_final%s' % '{:0>2}'.format(a)], locals()['df_Aux%s' % '{:0>2}'.format(a)]], axis='index', join='outer')
                    locals()['df_final%s' % '{:0>2}'.format(a)].dropna(axis=1, how='all', subset=None, inplace=True)
                    locals()['df_final%s' % '{:0>2}'.format(a)].sort_index(inplace=True)
                except:
                    pass
                print(f, d, s, a, b)
            except:
                pass
    for z in enumerate(sheet, start=0):
        df_final = pd.concat([df_final, locals()['df_final%s' % '{:0>2}'.format(z[0])]], axis='columns', join='outer')

    return df_final



def compile_ControlFinoTurno(id, files, date, sheet, file_path):
    import datetime as dt

    for x in enumerate(sheet, start=0):
        locals()['df_final%s' % '{:0>2}'.format(x[0])] = pd.DataFrame()
        locals()['df_Aux%s' % '{:0>2}'.format(x[0])] = pd.DataFrame()

    df_final = pd.DataFrame()

    for f, d in zip(files, date):
        for s, (a, b) in zip(sheet, enumerate(id)):
            #print(a)
            try:
                df_Aux = pd.read_excel(file_path + f,
                                       # usecols=list(range(col_iProgConc[0], col_iProgConc[0] + 31, 1)),
                                       # header=[4, 5],
                                       # index_col=[col_iDRT[0]+7],
                                       # dtype=np.float64,
                                       # nrows=1,
                                       sheet_name=s,
                                       engine=None
                                       )
                indexi, coli = np.where(df_Aux == b)
                try:
                    locals()['df_Aux%s' % '{:0>2}'.format(a)] = pd.read_excel(file_path + f,
                                                                                 # usecols=list(range(col_iProgConc[0], col_iProgConc[0] + 31, 1)),
                                                                                 header=[indexi[0] + 1, indexi[0] + 2],
                                                                                 # index_col=[col_iDRT[0]+7],
                                                                                 # dtype=np.float64,
                                                                                 # nrows=1,
                                                                                 sheet_name=s,
                                                                                 engine=None
                                                                                 )
                    locals()['df_Aux%s' % '{:0>2}'.format(a)].drop(locals()['df_Aux%s' % '{:0>2}'.format(a)].index[pd.Period(d, freq='M').end_time.date().day * 3:], inplace=True)
                    locals()['df_Aux%s' % '{:0>2}'.format(a)] = locals()['df_Aux%s' % '{:0>2}'.format(a)].iloc[:, coli[0]:]
                    locals()['df_Aux%s' % '{:0>2}'.format(a)] = clean_dataset(locals()['df_Aux%s' % '{:0>2}'.format(a)], replace_near0_with=0)
                    locals()['df_Aux%s' % '{:0>2}'.format(a)] = normalize_chars(locals()['df_Aux%s' % '{:0>2}'.format(a)], rename_unnamed_cols=True)
                    locals()['df_Aux%s' % '{:0>2}'.format(a)].index = sorted([d + dt.timedelta(days=x) for x in range(pd.Period(d, freq='M').end_time.date().day)] * 3)
                    locals()['df_Aux%s' % '{:0>2}'.format(a)].index = pd.Series(locals()['df_Aux%s' % '{:0>2}'.format(a)].index) + pd.Series([dt.timedelta(hours=5), dt.timedelta(hours=13), dt.timedelta(hours=21)] * pd.Period(d, freq='M').end_time.date().day)
                    locals()['df_Aux%s' % '{:0>2}'.format(a)] = locals()['df_Aux%s' % '{:0>2}'.format(a)].add_prefix(s.replace(' ', '_') + '_')
                    locals()['df_Aux%s' % '{:0>2}'.format(a)] = normalize_chars(locals()['df_Aux%s' % '{:0>2}'.format(a)], rename_unnamed_cols=False)
                    locals()['df_final%s' % '{:0>2}'.format(a)] = pd.concat([locals()['df_final%s' % '{:0>2}'.format(a)], locals()['df_Aux%s' % '{:0>2}'.format(a)]], axis='index', join='outer')
                    locals()['df_final%s' % '{:0>2}'.format(a)].dropna(axis=1, how='all', subset=None, inplace=True)
                    locals()['df_final%s' % '{:0>2}'.format(a)].sort_index(inplace=True)
                except:
                    pass
                print(f, d, s, a, b)
            except:
                pass
    for z in enumerate(sheet, start=0):
        df_final = pd.concat([df_final, locals()['df_final%s' % '{:0>2}'.format(z[0])]], axis='columns', join='outer')

    return df_final

#%%
def df_findsubstring(df, val, case=False, na=False):
    df = df.astype(str)
    mask = np.column_stack([df[col].str.lower().str.contains(val, na=na, case=case) for col in df])
    find_result = np.where(mask==True)
    return find_result


#%% Normalize, lower and remove spaces by CHATGPT
import unicodedata
import re

def normalize_and_replace(string, chars_to_keep=r'[^a-z0-9_+\-%#\(\)]', replace_nan_unnamed=True, remove_numbers=False):
    if not isinstance(string, str):
        return string  # pass through NaN / non-string values unchanged
    normalized_string = unicodedata.normalize('NFKD', string)
    lowercase_string = normalized_string.lower()
    if replace_nan_unnamed:
        lowercase_string = lowercase_string.replace('nan', '').replace('unnamed', '').replace('units', '').replace('unit', '')
    if remove_numbers:
        lowercase_string = re.sub(r'\d', '', lowercase_string)
    replaced_special_chars = re.sub(chars_to_keep, '_', lowercase_string)
    cleaned_string = re.sub(r'_+', '_', replaced_special_chars)  # Remove consecutive underscores
    return cleaned_string.strip('_')

'''
def normalize_dataframe_columns(df):
    def flatten_multiindex(levels):
        flattened_levels = [normalize_and_replace(str(level)) for level in levels]
        return '_'.join(flattened_levels)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [flatten_multiindex(cols) for cols in df.columns]
    else:
        df.columns = [normalize_and_replace(col) for col in df.columns]
    return df



def normalize_and_replace(string):
        normalized_string = string.lower()
        if replace_unnamed:
            if any(normalized_string.startswith(prefix) for prefix in ['unnamed', 'nan', 'type']):
                parts = string.split('_', maxsplit=1)
                if len(parts) > 1:
                    return parts[1]
            return string
'''

def normalize_dataframe_columns(df, remove_numbers=False):
    def flatten_multiindex(levels):
        flattened_levels = [normalize_and_replace(str(level), remove_numbers=remove_numbers) for level in levels]
        return '_'.join(flattened_levels).rstrip('_')

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [flatten_multiindex(cols) for cols in df.columns]
    else:
        df.columns = [normalize_and_replace(col, remove_numbers=remove_numbers) for col in df.columns]

    return df


def normalize_series_values(series):
    return series.apply(normalize_and_replace)


def normalize_dataframe_values(df):
    return df.apply(normalize_and_replace)


def rename_duplicates(df):
    new_columns = []
    seen_columns = set()

    for col in df.columns:
        new_col = col
        count = 1

        while new_col in seen_columns:
            new_col = f"{col}_{count}"
            count += 1

        seen_columns.add(new_col)
        new_columns.append(new_col)

    df.columns = new_columns
    return df

#%% fill nans in duplicated columns

def fill_duplicates(df):
    duplicates = df.columns[df.columns.duplicated(keep=False)].tolist()
    filled_columns = set()

    for duplicate in duplicates:
        indices = df.columns.get_loc(duplicate)

        if duplicate in filled_columns:
            new_column_name = duplicate + '_' + str(indices[0] - indices[1] + 1)
            df.rename(columns={duplicate: new_column_name}, inplace=True)
        else:
            if duplicate not in filled_columns:
                df[duplicate].fillna(df.iloc[:, indices[1:]], inplace=True)
                filled_columns.add(duplicate)
                continue

            first_occurrence = df[duplicate]
            subsequent_occurrences = df.iloc[:, indices[1:]].copy()

            for i in range(len(df)):
                if pd.isnull(first_occurrence[i]):
                    non_null_values = subsequent_occurrences.loc[i, :].dropna()

                    if pd.isnull(non_null_values).all() or subsequent_occurrences.loc[i, :].isnull().sum() > 1:
                        break

                    first_occurrence[i] = non_null_values.iloc[0]
                    subsequent_occurrences.loc[i, non_null_values.index[1:]] = pd.NA

            df[duplicate] = first_occurrence
            df.iloc[:, indices[1:]] = subsequent_occurrences

    return df



#%%

def dataframe_to_python_code(df, folder, filename):
    # Round only the float values to 2 decimal places
    df_rounded = df.copy()
    float_columns = df_rounded.select_dtypes(include=['float']).columns
    df_rounded[float_columns] = df_rounded[float_columns].round(1)
    
    index_code = f"index={df_rounded.index.tolist()}"
    columns_code = f"columns={df_rounded.columns.tolist()}"
    values_code = f"data={df_rounded.values.tolist()}"
    
    code = f"""
import pandas as pd

index = {index_code}
columns = {columns_code}
data = {values_code}

df = pd.DataFrame(data, index=index, columns=columns)
print(df)
"""

    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Define the full path to the file
    file_path = os.path.join(folder, filename)
    
    # Write the code to the file
    with open(file_path, 'w') as file:
        file.write(code)
    
    print(f"Code has been saved to {file_path}")







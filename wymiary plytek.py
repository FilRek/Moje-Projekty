import pandas as pd
import re
import math

def normalizuj_wymiar(nazwa_produktu: str) -> str:

    if not isinstance(nazwa_produktu, str):
        return ""
    pattern = r'(\d+[\.,]?\d*)\s*[xX]\s*(\d+[\.,]?\d*)'
    match = re.search(pattern, nazwa_produktu)
    if not match:
        return ""
    try:
        dim1_str = match.group(1).replace(',', '.')
        dim2_str = match.group(2).replace(',', '.')
        dim1 = float(dim1_str)
        dim2 = float(dim2_str)
        if dim1 > 200 or dim2 > 200:
            dim1_cm = dim1 / 10
            dim2_cm = dim2 / 10
        else:
            dim1_cm = dim1
            dim2_cm = dim2
        dim1_rounded = int(math.ceil(dim1_cm))
        dim2_rounded = int(math.ceil(dim2_cm))
        mniejszy = min(dim1_rounded, dim2_rounded)
        wiekszy = max(dim1_rounded, dim2_rounded)
        return f"{mniejszy}x{wiekszy}"
    except (ValueError, IndexError):
        return ""


#Wczytywanie i zapis pliku 
try:
    input_file = 'sciezka_do_pliku/plik.xlsx' 
    df_z_pliku = pd.read_excel(input_file)
    
    if 'NAZWA_CALA' in df_z_pliku.columns:
        df_z_pliku['WYMIAR_ZNORMALIZOWANY'] = df_z_pliku['NAZWA_CALA'].apply(normalizuj_wymiar)
        
        output_file = 'C:/Users/Filip Rek/Desktop/wszystkie_plytki.xlsx' 
        df_z_pliku.to_excel(output_file, index=False)
        
        print(f"\nPrzetwarzanie zakończone. Wyniki zapisano w pliku: {output_file}")
    else:
        print(f"Błąd: W pliku '{input_file}' nie znaleziono kolumny 'NAZWA_CALA'.")

except FileNotFoundError:

    print(f"Błąd: Plik '{input_file}' nie został znaleziony.")

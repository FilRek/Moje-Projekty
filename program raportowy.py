import tkinter as tk
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import locale

try:
    locale.setlocale(locale.LC_TIME, 'Polish')
except:
    print("Nie udało się ustawić lokalizacji.")

dane = "dane.csv"
# --- Funkcja przeliczania i zapisu danych ---
def zapisz_i_pokaz():
    try:
        PZ = float(entry_PZ.get() or 0)
        PZk = float(entry_PZk.get() or 0)
        MP = float(entry_MP.get() or 0)
        ZU = float(entry_ZU.get() or 0)
        ZUk = float(entry_ZUk.get() or 0)
        PW = float(entry_PW.get() or 0)

        WZ = float(entry_WZ.get() or 0)
        WZk = float(entry_WZk.get() or 0)
        MW = float(entry_MW.get() or 0)
        SU = float(entry_SU.get() or 0)
        SUk = float(entry_SUk.get() or 0)
        RW = float(entry_RW.get() or 0)

        # Obliczenia
        zakupy = PZ + PZk + MP + ZU + ZUk + PW
        sprzedaz = WZ + WZk + MW + SU + SUk + RW
        miesiac = datetime.now().strftime("%B %Y")

        if not os.path.exists(dane):
            df = pd.DataFrame(columns=["Miesiąc", "Zakupy netto", "Sprzedaż netto"])
            df.to_csv(dane, index=False)

        # Wczytanie i dopisanie danych
        df = pd.read_csv(dane)
        nowy_wiersz = {"Miesiąc": miesiac, "Zakupy netto": zakupy, "Sprzedaż netto": sprzedaz}
        df = pd.concat([df, pd.DataFrame([nowy_wiersz])], ignore_index=True)
        df.to_csv(dane, index=False)

        messagebox.showinfo("Sukces", f"Dane zapisane dla {miesiac}.\n"
                                      f"Zakupy: {zakupy:.2f} zł\nSprzedaż: {sprzedaz:.2f} zł")

        # Wykres
        kolor = "green" if sprzedaz > zakupy else "red"
        plt.figure(figsize=(4, 6))
        plt.bar(["Zakupy netto", "Sprzedaż netto"], [zakupy, sprzedaz], color=["gray", kolor])
        plt.title(f"Wyniki za {miesiac}")
        plt.ylabel("Wartość netto [PLN]")
        plt.tight_layout()
        plt.show()

    except ValueError:
        messagebox.showerror("Błąd", "Upewnij się, że wszystkie pola zawierają liczby.")

# --- Tworzenie GUI ---
root = tk.Tk()
root.title("Zestawienie miesięczne - Zakupy i Sprzedaż")
root.geometry("400x700")
root.resizable(False, False)


frame_zakupy = tk.LabelFrame(root, text="Zakupy netto", padx=10, pady=10)
frame_zakupy.pack(padx=10, pady=10, fill="both")

frame_sprzedaz = tk.LabelFrame(root, text="Sprzedaż netto", padx=10, pady=10)
frame_sprzedaz.pack(padx=10, pady=10, fill="both")

# --- Pola wejściowe ---
def dodaj_pole(frame, label):
    tk.Label(frame, text=label).pack()
    entry = tk.Entry(frame)
    entry.pack()
    return entry

entry_PZ = dodaj_pole(frame_zakupy, "PZ")
entry_PZk = dodaj_pole(frame_zakupy, "PZk")
entry_MP = dodaj_pole(frame_zakupy, "MP")
entry_ZU = dodaj_pole(frame_zakupy, "ZU")
entry_ZUk = dodaj_pole(frame_zakupy, "ZUk")
entry_PW = dodaj_pole(frame_zakupy, "PW")

entry_WZ = dodaj_pole(frame_sprzedaz, "WZ")
entry_WZk = dodaj_pole(frame_sprzedaz, "WZk")
entry_MW = dodaj_pole(frame_sprzedaz, "MW")
entry_SU = dodaj_pole(frame_sprzedaz, "SU")
entry_SUk = dodaj_pole(frame_sprzedaz, "SUk")
entry_RW = dodaj_pole(frame_sprzedaz, "RW")

tk.Button(root, text="Zapisz i pokaż wykres", command=zapisz_i_pokaz, bg="lightgreen").pack(pady=20)

root.mainloop()
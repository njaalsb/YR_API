import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

def MLR(x, y_vektor):
    """
    Beregnar resultata frå ein multippel lineær regresjon ved hjelp av minste kvadrats metode.
    
    Input:
    - x: Ei matrise (pandas DataFrame eller numpy array) som inneheld forklaringsvariablane. Kvar kolonne representerar ein forklaringsvariabel.
    - y_vektor: Ein vektor (pandas Series eller numpy array) som inneheld responsvariablene.

    Output:
    - Ei ordbok med følgande nøklar:
      - "n": Antall rader i forklaringsvariablene (antall datapunkt).
      - "b_verdiar": Beta-koeffisientene som er beregna gjennom minste kvadraters metode.
      - "designmatrise": Designmatrisen som inneheld forklaringsvariablane og ei kolonne med 1-era for konstantleddet.
      - "hatt_matrise": Hatte-matrisen, som er brukt til å projisere y-verdiane i modellen.
      - "y_predikert": Predikerte verdiar for den avhengige variabelen basert på modellen.
      - "residualar": Residualene, dvs. forskjellen mellom dei observerte og predikerte y-verdiane.
      - "TSS": Total Sum of Squares – summen av kvadrerte avvik mellom observasjonane og gjennomsnittet av y.
      - "RSS": Residual Sum of Squares – summen av kvadrerte avvik mellom observasjonane og prediksjonane.
      - "ESS": Explained Sum of Squares – summen av kvadrerte avvik mellom prediksjonane og gjennomsnittet av y.
      - "R2": Determinasjonskoeffisienten, angir kor godt modellen forklarer variasjonen i den avhengige variabelen (R²-verdi).
      - "R2a": Justert R²-verdi, som tek omsyn til antall forklaringsvariablar og antall observasjonar.
      - "varians": Variansen av residualene (S²).
      - "stdavvik": Standardavviket til residualene (S).
      - "p": Antall forklaringsvariabler i modellen.
      - "stdresidualar": Standardiserte residualer, som er residualene delt på deira standardfeil.
      """

    n = x.shape[0]
    
    # Lag en kolonne med 1-ere
    enera = np.ones((n, 1))
    
    # Kombiner 1-ere kolonnen med målepunktene, lager designmatrise
    X = np.column_stack((enera, x))
    
    # Beregn minste kvadraters estimat for beta
    x_trans = X.T
    beta = np.array(np.linalg.inv(x_trans @ X) @ x_trans @ y_vektor)
    p = (len(beta)-1)
    
    #  Hatte-matrisen
    hat_matrise = X @ np.linalg.inv(x_trans @ X) @ x_trans
    y_predikert = hat_matrise @ y_vektor
    
    
    residualar = y_vektor - y_predikert
    
    TSS = np.sum((y_vektor - np.mean(y_vektor))**2, axis = 0)
    RSS = np.sum(residualar**2, axis = 0)
    ESS = np.sum((y_predikert - np.mean(y_vektor))**2, axis = 0)
    
    R2 = ESS / TSS
    R2a = 1 - ((RSS / (n - p - 1)) / (TSS / (n - 1)))
    
    S2 = RSS / (n - 1 - p)
    S = np.sqrt(S2)
    
    stdresidualar_liste = []
    diagonal = np.diag(hat_matrise)
    for j in range(0, len(hat_matrise)):
        stdfeil = np.sqrt(S2 * (1 - diagonal[j]))
        stdresi = residualar[j] / stdfeil
        stdresidualar_liste.append(stdresi)
    stdresidualar = np.array(stdresidualar_liste) 
    
    

    return { "n": n, "b_verdiar": beta, "designmatrise": X, "hatt_matrise": hat_matrise, "y_predikert": y_predikert,
             "residualar": residualar, "TSS": TSS, "RSS": RSS, "ESS": ESS, "R2": R2, "R2a": R2a, "varians": S2,
             "stdavvik": S, "p": p, "stdresidualar": stdresidualar}

def konf_interval_beta(b_verdi, stdavvik, X):
    """
    Reknar ut 95%-konfidensintervallet til alle beta-verdiane i vektoren.
    Input:
    - b_verdi: En vektor (liste eller numpy array) som inneheld beta-koeffisientene frå ei regresjonsanalyse.
    - stdavvik: En float som representerer standardavviket til residualene i modellen.
    - X: Designmatrisen (numpy array eller pandas DataFrame) som inneheld forklaringsvariablene med ei kolonne for konstantleddet.

    Output:
    - Ei liste med tupler, der kvart element representerar konfidensintervallet for den tilsvarende beta-verdien. Kvar tuppel består av:
      - min_sum: Nedre grense for konfidensintervallet for den aktuelle beta-verdien.
      - max_sum: Øvre grense for konfidensintervallet for den aktuelle beta-verdien.
    """
    konf_int = []
    b_array = np.array(b_verdi)
    for j in range(len(b_array)):
        beta_j = b_array[j]
        baksum = stdavvik * np.sqrt(np.linalg.inv(X.T@X)[j, j])  
        min_sum = float(np.round((beta_j - 1.96 * baksum), 5))
        max_sum = float(np.round((beta_j + 1.96 * baksum), 5))
        konf_int.append((min_sum, max_sum))
    return konf_int

def H0_hypotesetest(b_verdi, S2, X, friheitsgrader, signifikansniva):
    """
    Utførar hypotesetesting av nullhypotesen H0 for kvar beta-koeffisientene i en lineær regresjonsmodell.
    H0: β_k = 0, for hver beta-verdi, med et angitt signifikansnivå α.

    Input:
    - b_verdi: Ein vektor (liste eller numpy array) som inneheld beta-koeffisientene frå ei regresjonsanalyse.
    - S2: Ein float som representerar variansen av residualene frå modellen.
    - X: Designmatrisen (numpy array eller pandas DataFrame) som inneheld forklaringsvariablene med ei kolonne for konstantleddet.
    - friheitsgrader: Ein int som angir frihetsgradene til modellen, typisk lik antall forklaringsvariablar i modellen.
    - signifikansniva: Ein float som representerar signifikansnivået for hypotesetesten.

    Output:
    - Ein liste med hypotesetestresultater for kvar beta-koeffisient:
      - Kvart element i lista gir informasjon om nullhypotesen avvisast eller ikkje, og den tilhøyrande p-verdien.
    """
    
    # Beregn kritisk verdi basert på signifikansnivå
    kritisk_verdi = stats.norm.ppf(1 - signifikansniva / 2)
    resultat = []
    for k in range(1, friheitsgrader+1):
        testverdi = np.array((abs(b_verdi[k])) / (np.sqrt(S2 * (np.linalg.inv(X.T@X)[k, k]))))
        p_verdi = 2*(1 - stats.norm.cdf(testverdi))
        if testverdi >= kritisk_verdi:
            resultat.append(f"Beta_{k} = 0 avvis, P-verdi = {p_verdi:.5f}.")
        else:
            resultat.append(f"Beta_{k} = 0 ikkje avvis, P-verdi = {p_verdi:.5f}.")
        
    return resultat

def alle0_hypotesetest(ESS, RSS, antal, friheitsgrader, signifikans):
    """
    Utfører ein F-test for å teste nullhypotesen H0: Alle beta-koeffisienter = 0.
    
    Input:
    - ESS: Explained Sum of Squares (ESS). Summen av kvadrerte avvik mellom dei predikerte y-verdiene og gjennomsnittet av y.
    - RSS: Residual Sum of Squares (RSS). Summen av kvadrerte avvik mellom observerte og predikerte y-verdier.
    - antal: Antall observasjonar i datasettet (n).
    - friheitsgrader: Antall frihetsgrader for modellen, som er lik antall forklaringsvariabler (p).
    - signifikans: Signifikansnivået for testen, vanligvis satt til 0.05 (5%).

    Output:
    - Ein streng som forteller om nullhypotesen (alle beta-koeffisienter = 0) kan avvisast eller ikkje, samt p-verdien for testen.
    """
    
    testverdi = np.array((ESS / friheitsgrader) / (RSS / (antal-(friheitsgrader+1))))
    f_verdi = stats.f.ppf(1-signifikans, friheitsgrader, antal-(friheitsgrader+1))
    p_verdi = 1 - stats.f.cdf(testverdi, friheitsgrader, antal-(friheitsgrader+1))
    if testverdi >= f_verdi:
        return (f"Avviser H0. P-verdi = {p_verdi:.5f}.")
    else:
        return (f"Ikkje grunnlag for å forkaste H0. P-verdi = {p_verdi:.5f}.")
 
def korrelasjonsmatrise(x):
    """
    Beregnar korrelasjonsmatrisen for ei gitt matrise med forklaringsvariablar, samt den standardiserte matrisen og kondisjonstallet.
    
    Input:
    - x: Ei matrise (numpy array) som inneheld forklaringsvariablane. Kvar kolonne representerar ein forklaringsvariabel.

    Output:
    - Ei ordbok med tre nøklar:
      - "korrmatrise": Eit numpy array som representerar korrelasjonsmatrisen for variablene i `x`.
      - "Z": Eit numpy array/Pandas Data Frame som representerar standardiserte verdier av variablene i `x` (også kalt Z-verdi-matrisen).
      - "kondisjonstal": En float som angir kondisjonstallet til den standardiserte matrisen `Z`, som brukes til å vurdere multikollinearitet.
      """
    gjennomsnitt_x = np.mean(x, axis=0)
    stdavvik_x = np.std(x, axis=0, ddof=1)
    Z = (x - gjennomsnitt_x) / stdavvik_x
    korrelasjonsmatrisen = (1 / (len(x)-1))* Z.T@Z
    kondisjonstal = np.linalg.cond(Z)
    return { "korrmatrise": korrelasjonsmatrisen, "Z": Z, "kondisjonstal": kondisjonstal }

def qq(y):
    """
    Returnerar ei liste med sorterte stigande tall frå standard-normalfordelinga med same lengd som x og dei standardiserte residualane sortert stigande.
    Input:
    - x: Ei liste eller array som brukast til å bestemme antall elementer i den genererte standard normalfordelingslisten.
    - y: Ei liste eller array av standardiserte residualer.

    Output:
    - En ordbok med to nøkler:
      - "x_stig": Verdier fra en standard normalfordeling sortert stigende.
      - "stdres_stig": Standardiserte residualer sortert stigende.
    """
    norm = stats.norm.rvs(size = len(y))
    norm_stig = np.sort(norm)
    stdres_stig = np.sort(y)
    
    return {"norm_stig": norm_stig, "stdres_stig": stdres_stig }

def punktplot(x, y, farge, label, tittel):
    """
    Lagar scatterplots for kvar variabel i x mot den avhengige variabelen y.
    
    Input:
    - x: Pandas DataFrame med forklaringsvariablane. Kvar kolonne representerar ein forklaringsvariabel.
    - y: Pandas Series som representerar den avhengige variabelen.
    - farge: Ein streng som angir fargen som skal brukast for datapunkta i scatterplotta.
    - label: Ein streng som angir etiketten som skal brukast i forklaringa (legend) for datapunkta.
    - tittel: Ein streng som angir tittelen som skal brukast for scatterplotta.

    Output:
    - Funksjonen har ingen eksplisitt returverdi, men viser scatterplots for kvar variabel i x mot y.
    """

    for variab in range(len(x.columns)):
        plt.scatter(x.iloc[:, variab], y, color = farge, label = f'{label}', zorder = 5)
        plt.xlabel(f'{x.columns[variab]}')
        plt.ylabel(f'{y.name}')
        plt.legend()
        plt.title(f'{tittel}')
        plt.grid(True, zorder = 1)
        plt.show()
 
def VIF(z):
    """
    Beregnar Variance Inflation Factor (VIF) for kvar variabel i matrise z.
    
    Input:
    - z: Ei Pandas DataFrame som inneheld dei standardiserte forklaringsvariablene fra regresjonsmodellen. Hver kolonne representerer en forklaringsvariabel.

    Output:
    - Ei liste med VIF-verdier for kvar forklaringsvariabel.
    """
    vif_verdi = []
    for t in range(len(z.columns)):
        j_variabel = z.iloc[:, t]
        rest_variabel = z.drop(z.columns[t], axis=1)
    
        R2j = MLR(rest_variabel, j_variabel)['R2']
    
        vif = float(np.round(1 / (1-R2j), 5))
        vif_verdi.append(f"VIF til {z.columns[t]} = {vif}.")
    return vif_verdi

#  Lastar inn datasettet
datasett = pd.read_csv("dataset2023.csv")
#  Tildelar electricity variabelen y. 
y = (datasett['electricity'])
#  Tildelar airtemp, dewtemp, precipitation, pressure, relhum, vapor, windspeed, cumprecip til variabelen x.
x = (datasett.drop(columns=['time', 'electricity']))

#  Tilordnar variabel til funksjonen MLR med input x og y.
resultat = MLR(x, y)

#  Oppgåve 1
#  Hentar ut verdiane for betakoeffsientane og printar dei.
beta = np.round(resultat.get("b_verdiar"), 5)
for variabel in range(len(beta)):
    print(f"Beta{variabel} = {beta[variabel]} \n")
#  Lagar punktplot med x og y.
punktplot(x, y, 'red', 'Opprinneleg data', 'Punktplot forklaringsvariablar')
 

#  Oppgåve 2
#  Hentar ut verdiane for R2 og R2-adjusted og printar dei.
R2 = resultat.get("R2")
R2a = resultat.get("R2a")
print(f"R2 = {R2:.5f}, R2a = {R2a:.5f}")

#  Hentar ut y-verdiar predikert av regresjonsmodellen og plottar dei mot opprinneleg data.
y_predikert = resultat.get("y_predikert")
for variab in range(len(x.columns)):
    plt.scatter(x.iloc[:, variab], y, color = 'red', label = 'Opprinneleg data', zorder=5)
    plt.scatter(x.iloc[:, variab], y_predikert, color = 'blue', label = 'Data frå regmodell', zorder=5)
    plt.xlabel(f'{x.columns[variab]}')
    plt.ylabel(f'{y.name}')
    plt.legend()
    plt.grid(True, zorder=1)
    plt.title(f"Punktplot for reell/predikert straumpris mot {x.columns[variab]}.")
    plt.show()
    
#  Oppgåve 3

#  Hentar ut nødvendige variablar og finn og printar konfidensintervallet til B1 ved funksjonen konf_interval.
stdavvik = resultat.get("stdavvik")
designmatrise = resultat.get("designmatrise")
konf_interval = konf_interval_beta(beta, stdavvik, designmatrise)
print(f"Konfidensintervalet for Beta1 = {konf_interval[1]}")

#  Oppgåve 4

#  Hentar ut nødvendige variablar og finn og printar svar frå hypotesetesten for "alle Beta = 0".
ESS = resultat.get("ESS")
RSS = resultat.get("RSS")
n = resultat.get("n")
p = resultat.get("p")
print(alle0_hypotesetest(ESS, RSS, n, p, 0.05))

#  Oppgåve 5

#  Hentar ut nødvendige variablar og finn og printar svar for hypotesetest om dei ulike Beta = 0.
S2 = resultat.get("varians")
H0_test = H0_hypotesetest(beta, S2, designmatrise, p, 0.05)
for u in H0_test:
    print(f"{u} \n")

#  Oppgåve 6

#  Hentar ut korrelasjonsmatrise frå funksjonen og tilordnar den variabel.
korrmattrise = korrelasjonsmatrise(x)

#  Lagar heatmap for korrelasjonsmatrisen.
plt.figure(figsize=(10, 10))
sns.heatmap(korrmattrise.get("korrmatrise"), annot=True, cmap='coolwarm', linewidths=1)
plt.title('Korrelasjonsmatrise for forklaringsvariablane')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()

#  Oppgåve 7

#  Hentar og printar ut kondisjonstalet frå funksjon.
kondisjon = korrmattrise.get("kondisjonstal")
print(f"Kondisjonstalet er {kondisjon:.5f}.")

#  Oppgåve 8

#  Hentar ut standardiserte verdiar og tilordnar dei til variabel.
z = korrmattrise.get("Z")
#  Kjører funksjonen VIF(VIF-analyse) med input z og printar resultat.
vif = VIF(z)
for variabel1 in vif:
    print(f" {variabel1} \n")

#  Fjernar kolonna med "dewtemp" frå det standardiserte datasettet.
z_uten_dew = z.drop(columns='dewtemp')
#  Kjører funksjonen VIF(VIF-analyse) med input z_uten_dew og printar resultat.
vif1 = VIF(z_uten_dew)
for variabel2 in vif1:
    print(f" {variabel2} \n")

#  Fjernar kolonna med "vapor" frå datasettet "z_uten_dew".
z_uten_dewvap = z_uten_dew.drop(columns='vapor')
#  Kjører funksjonen VIF(VIF-analyse) med input z_uten_dewvap og printar resultat.
vif2 = VIF(z_uten_dewvap)
for variabel3 in vif2:
    print(f" {variabel3} \n")


#  Oppgåve 9

#  Tildelar airtemp, precipitation, pressure, relhum, windspeed, cumprecip til variabelen x.
x_ny = (datasett.drop(columns=['time', 'electricity', 'dewtemp', 'vapor']))
#  Hentar ut korrelasjonsmatrise frå funksjonen og tilordnar den variabel.
korrelasjonsmatrise_ny = korrelasjonsmatrise(x_ny)

#  Lagar heatmap for den nye korrelasjonsmatrisen.
plt.figure(figsize=(10, 10))
sns.heatmap(korrelasjonsmatrise_ny.get("korrmatrise"), annot=True, cmap='coolwarm', linewidths=1)
plt.title('Korrelasjonmatrise for oppdaterte forklaringsvariablar')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()

#  Hentar ut det nye kondisjonstalet og printar det.
kondisjon = korrelasjonsmatrise_ny.get("kondisjonstal")
print(f"Kondisjonstalet er {kondisjon:.5f}.")

#  Oppgåve 10

#  Tilordnar variabel til funksjonen MLR med input x_y og y.
resultat_ny = MLR(x_ny, y)

#  Hentar ut verdiane for R2 og R2-adjusted og printar dei.
R2 = resultat_ny.get("R2")
R2a = resultat_ny.get("R2a")
print(f"R2 = {R2:.5f}, R2a = {R2a:.5f}")

#  Hentar ut y-verdiar predikert av regresjonsmodellen og plottar dei mot opprinneleg data.
y_predikert_ny = resultat_ny.get("y_predikert")
for variab in range(len(x_ny.columns)):
    plt.scatter(x_ny.iloc[:, variab], y, color = 'red', label = 'Opprinneleg data')
    plt.scatter(x_ny.iloc[:, variab], y_predikert_ny, color = 'blue', label = 'Data frå regmodell')
    plt.xlabel(f"{x_ny.columns[variab]}")
    plt.ylabel(f"{y.name}")
    plt.grid(True)
    plt.legend()
    plt.title(f"Punktplot for reell/predikert straumpris mot {x_ny.columns[variab]}.")
    plt.show()

#  Hentar ut nødvendige variablar og finn og printar konfidensintervallet til B1 ved funksjonen konf_interval.
beta_ny = np.round(resultat_ny.get("b_verdiar"), 5)
stdavvik = resultat_ny.get("stdavvik")
designmatrise = resultat_ny.get("designmatrise")
konf_interval = konf_interval_beta(beta_ny, stdavvik, designmatrise)
print(konf_interval[1])
     
#  Oppgåve 11

#  Hentar ut standardiserte residualar og plottar dei mot observasjonsnummera til x.
stdresidualar = resultat_ny.get("stdresidualar")
plt.scatter(range(len(stdresidualar)), stdresidualar, color = 'red', label = 'Standardiserte residualar')
plt.xlabel('Observasjonsnummer')
plt.ylabel('Standardiserte residualar')
plt.title('Punktplot standardiserte residualar')
plt.axhline(y=0, color="green", linestyle="--", label = "y = 0")
plt.legend()
plt.grid(True)
plt.show()

#  Plottar dei standardiserte residualane mot alle forklaringsvariablane.
for variab in range(len(x_ny.columns)):
    plt.scatter(x_ny.iloc[:, variab], stdresidualar, color = 'red', label = 'Standardiserte residualar')
    plt.xlabel(f'{x_ny.columns[variab]}')
    plt.ylabel('Standardiserte residualar')
    plt.axhline(y=0, color="green", linestyle="--", label = "y = 0")
    plt.grid(True)
    plt.title(f'Punktplot standardiserte residualar vs {x_ny.columns[variab]}')
    plt.legend()
    plt.show()


#  Oppgåve 12

#  Hentar ut sorterte verdiar for standardiserte residualar og standard-normalfordelinga frå funksjon.
qq_plot = qq(stdresidualar)
norm_sort = qq_plot.get("norm_stig")
stdresi_sort = qq_plot.get("stdres_stig")

#  Lagar QQ-plott med verdiar frå standard normalfordeling og standardiserte residualar sortert stigande.
plt.scatter(qq_plot.get("norm_stig"), qq_plot.get("stdres_stig"))
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(qq_plot.get("norm_stig"), qq_plot.get("norm_stig"), color = "red", label = "Sortert standard-normalfordeling")
plt.legend()
plt.title("QQ-plot")
plt.grid(True)
plt.show()

#  Oppgåve 13

#  Hentar ut konfidensintervalla for alle forklaringsvariablane frå funksjon og printar dei.
konf_interval = konf_interval_beta(beta_ny, stdavvik, designmatrise)
for s in range(len(konf_interval)):
    print(f"Konfidensintervalet for beta{s} er {konf_interval[s]}.")


    

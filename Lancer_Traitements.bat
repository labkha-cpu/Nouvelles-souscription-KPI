@echo off
chcp 65001 >nul
color 0A
title KPI Data Quality - Automate

:: --- 1. SE PLACER DANS LE BON DOSSIER ---
:: Adaptez ce chemin si vous déplacez le dossier un jour
cd /d "C:\Users\klabadi\Downloads\KPI"

echo ========================================================
echo      DEMARRAGE DU PROCESSUS DATA QUALITY
echo ========================================================
echo.

:: --- 2. LANCEMENT SCRIPT 1 (Preparation) ---
echo [ETAPE 1/4] Generation IEHE et Requetes SQL...
python Scripts/01_generation_donnees.py
if %ERRORLEVEL% NEQ 0 goto :Erreur

echo.
echo ========================================================
echo  STOP ! ACTION MANUELLE REQUISE
echo ========================================================
echo 1. Allez dans le dossier 'Output' et ouvrez les fichiers .sql
echo 2. Executez ces requetes sur votre base CIAM.
echo 3. Enregistrez les resultats dans 'Input_Data' sous les noms :
echo    - JJMMAAAA_CM.csv
echo    - JJMMAAAA_CK.csv
echo.
echo Une fois que les fichiers sont deposes, appuyez sur une touche...
echo ========================================================
pause

:: --- 3. LANCEMENT SCRIPT 2 (KPI) ---
echo.
echo [ETAPE 2/4] Calcul des KPIs...
python Scripts/02_calcul_kpi.py
if %ERRORLEVEL% NEQ 0 goto :Erreur

:: --- 4. LANCEMENT SCRIPT 3 (Details) ---
echo.
echo [ETAPE 3/4] Generation fichiers detailles (NS_CIAM/IEHE)...
python Scripts/03_generation_fichiers_detail.py
if %ERRORLEVEL% NEQ 0 goto :Erreur

:: --- 5. LANCEMENT SCRIPT 4 (BDD) ---
echo.
echo [ETAPE 4/4] Historisation en Base de Donnees...
python Scripts/04_chargement_bdd.py
if %ERRORLEVEL% NEQ 0 goto :Erreur

echo.
echo ========================================================
echo      TRAITEMENT TERMINE AVEC SUCCES ! 
echo ========================================================
pause
exit

:Erreur
color 0C
echo.
echo ========================================================
echo      UNE ERREUR EST SURVENUE
echo      Verifiez les messages ci-dessus.
echo ========================================================
pause
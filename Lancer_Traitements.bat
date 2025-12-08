@echo off
chcp 65001 >nul
color 0A
title KPI Data Quality - Automate

:: --- 1. SE PLACER DANS LE BON DOSSIER ---
:: Force le repertoire courant a etre celui ou se trouve ce script
cd /d "%~dp0"

echo ========================================================
echo      DEMARRAGE DU PROCESSUS DATA QUALITY
echo ========================================================
echo.

:: --- 2. LANCEMENT SCRIPT 1 (Phase 1 : Email) ---
echo [ETAPE 1/4] Generation IEHE et Requetes SQL (Phase 1 : Email)...
python Scripts/01_generation_donnees.py
if %ERRORLEVEL% NEQ 0 goto :Erreur

echo.
echo ========================================================
echo [PAUSE 1] ACTION MANUELLE REQUISE : EMAIL
echo ========================================================
echo 1. Allez dans 'Output', recuperez 'EMAIL_Global.sql'.
echo 2. Executez sur BDD et enregistrez sous '..._CM.csv' dans 'Input_Data'.
echo.
pause

:: --- 3. LANCEMENT SCRIPT 1 (Phase 2 : KPEP) ---
echo.
echo [ETAPE 2/4] Generation Requetes SQL (Phase 2 : KPEP)...
python Scripts/01_generation_donnees.py
if %ERRORLEVEL% NEQ 0 goto :Erreur

echo.
echo ========================================================
echo [PAUSE 2] ACTION MANUELLE REQUISE : KPEP
echo ========================================================
echo 1. Allez dans 'Output', recuperez 'KPEP_Global.sql'.
echo 2. Executez sur BDD et enregistrez sous '..._CK.csv' dans 'Input_Data'.
echo.
pause

:: --- 4. LANCEMENT SCRIPT 1 (Phase 3 : Reliquat) ---
echo.
echo [ETAPE 2bis/4] Generation Requetes SQL (Phase 3 : Reliquat)...
python Scripts/01_generation_donnees.py
if %ERRORLEVEL% NEQ 0 goto :Erreur

echo.
echo ========================================================
echo [PAUSE 3] ACTION MANUELLE REQUISE : RELIQUAT
echo ========================================================
echo 1. Allez dans 'Output', recuperez les requetes 'LastName' et 'MiddleName'.
echo 2. Executez sur BDD.
echo 3. Enregistrez sous '..._Rech_Nom.csv' et '..._Rech_Middle.csv' dans 'Input_Data'.
echo.
pause

:: --- 5. LANCEMENT CALCULS FINAUX ---
echo.
echo [ETAPE 3/4] Calcul des KPIs...
python Scripts/02_calcul_kpi.py
if %ERRORLEVEL% NEQ 0 goto :Erreur

echo.
echo [ETAPE 4/4] Generation fichiers detailles...
python Scripts/03_generation_fichiers_detail.py
if %ERRORLEVEL% NEQ 0 goto :Erreur

echo.
echo [ETAPE FINALE] Historisation en Base de Donnees...
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

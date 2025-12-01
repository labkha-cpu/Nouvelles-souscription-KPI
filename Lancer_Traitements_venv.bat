@echo off
chcp 65001 >nul
color 0A
title KPI Data Quality - Automate

:: --- 1. SE PLACER DANS LE DOSSIER DU SCRIPT ---
cd /d "%~dp0"

echo ========================================================
echo      DEMARRAGE DU PROCESSUS DATA QUALITY
echo ========================================================

:: --- VERIFICATION DU VENV ---
:: On vérifie d'abord si le fichier d'activation existe
if not exist "venv\Scripts\activate.bat" goto :ErreurVenv

:: Si le fichier existe, on l'active et on saute l'erreur
call venv\Scripts\activate.bat
echo (VENV active : Environnement isole detecte)
echo.
goto :Etape1

:ErreurVenv
color 0C
echo.
echo [ERREUR] Le dossier 'venv' est introuvable ou incomplet.
echo Le fichier "venv\Scripts\activate.bat" n'existe pas.
echo.
echo Solution :
echo 1. Ouvrez une invite de commande ici.
echo 2. Tapez : python -m venv venv
echo 3. Tapez : venv\Scripts\pip install pandas numpy sqlalchemy psycopg
echo.
pause
exit

:: --- 2. LANCEMENT DES SCRIPTS ---
:Etape1
echo [ETAPE 1/4] Generation IEHE et Requetes SQL...
if not exist "Scripts\01_generation_donnees.py" goto :ErreurFichier
python Scripts/01_generation_donnees.py
if %ERRORLEVEL% NEQ 0 goto :ErreurPython

echo.
echo ========================================================
echo  STOP ! ACTION MANUELLE REQUISE
echo ========================================================
echo 1. Allez dans le dossier 'Output', recuperez les requetes SQL.
echo 2. Jouez-les et enregistrez les resultats dans 'Input_Data'.
echo    (Noms attendus : JJMMAAAA_CM.csv et JJMMAAAA_CK.csv)
echo.
echo Une fois fait, appuyez sur une touche pour continuer...
echo ========================================================
pause

echo.
echo [ETAPE 2/4] Calcul des KPIs...
if not exist "Scripts\02_calcul_kpi.py" goto :ErreurFichier
python Scripts/02_calcul_kpi.py
if %ERRORLEVEL% NEQ 0 goto :ErreurPython

echo.
echo [ETAPE 3/4] Generation fichiers detailles...
if not exist "Scripts\03_generation_fichiers_detail.py" goto :ErreurFichier
python Scripts/03_generation_fichiers_detail.py
if %ERRORLEVEL% NEQ 0 goto :ErreurPython

echo.
echo [ETAPE 4/4] Historisation en BDD...
if not exist "Scripts\04_chargement_bdd.py" goto :ErreurFichier
python Scripts/04_chargement_bdd.py
if %ERRORLEVEL% NEQ 0 goto :ErreurPython

echo.
echo ========================================================
echo      TRAITEMENT TERMINE AVEC SUCCES ! 
echo ========================================================
call deactivate
pause
exit

:: --- GESTION DES ERREURS ---
:ErreurFichier
color 0C
echo.
echo [ERREUR CRITIQUE] Script Python introuvable.
echo Verifiez que le dossier "Scripts" existe et contient les fichiers .py.
pause
exit

:ErreurPython
color 0C
echo.
echo [ERREUR CRITIQUE] Le script Python a echoue.
echo Regardez le message d'erreur ci-dessus.
call deactivate
pause
exit
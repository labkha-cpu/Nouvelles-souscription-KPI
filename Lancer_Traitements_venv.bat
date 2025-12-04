@echo off
chcp 65001 >nul
color 0A
title KPI Data Quality - Automate (VENV)

:: =================================================================================
:: SCRIPT : Lancer_Traitements_venv.bat
:: DESCRIPTION : Orchestrateur avec gestion d'environnement virtuel (venv)
::               et détection automatique de l'emplacement des scripts Python.
::               Logique complète : Email -> KPEP -> Reliquat (Nom/Middle).
::
:: --- HISTORIQUE ---
:: 1.0 : Version Initiale
:: 1.1 : Ajout détection dynamique dossier
:: 1.2 : Alignement Flux Global (2 pauses pour generation SQL progressive)
:: 1.3 : Ajout Etape 2bis (Reliquat Nom/Middle)
:: =================================================================================

:: --- 1. SE PLACER DANS LE DOSSIER DU SCRIPT ---
cd /d "%~dp0"

echo ========================================================
echo      DEMARRAGE DU PROCESSUS DATA QUALITY (VENV)
echo ========================================================

:: --- DETECTION DES SCRIPTS (Racine ou dossier Scripts) ---
if exist "01_generation_donnees.py" (
    set "P="
    echo [INFO] Scripts detectes a la racine.
) else (
    if exist "Scripts\01_generation_donnees.py" (
        set "P=Scripts\"
        echo [INFO] Scripts detectes dans le dossier 'Scripts'.
    ) else (
        goto :ErreurFichier
    )
)

:: --- VERIFICATION DU VENV ---
:: On vérifie d'abord si le fichier d'activation existe
if not exist "venv\Scripts\activate.bat" goto :ErreurVenv

:: Si le fichier existe, on l'active et on saute l'erreur
call venv\Scripts\activate.bat
echo (VENV active : Environnement isole detecte)
echo.

:: ==========================================================
:: [ETAPE 1] GENERATION REQUETE EMAIL (CM)
:: ==========================================================
echo.
echo [ETAPE 1/4] Generation IEHE et Requetes SQL (Phase 1 : Email)...
:: Lancement 01 : Va generer IEHE + SQL CM et s'arreter (car CM.csv manquant)
python "%P%01_generation_donnees.py"
if %ERRORLEVEL% NEQ 0 goto :ErreurPython

echo.
echo ----------------------------------------------------------
echo [PAUSE 1] ACTION MANUELLE REQUISE
echo ----------------------------------------------------------
echo.
echo 1. Allez dans le dossier 'Output'.
echo 2. Recuperez la requete SQL terminant par 'EMAIL_Global.sql'.
echo 3. Executez-la sur votre base de donnees.
echo 4. Enregistrez le resultat sous '..._CM.csv' dans le dossier 'Input_Data'.
echo.
echo Une fois le fichier CM depose, appuyez sur une touche pour continuer.
echo.
pause >nul

:: ==========================================================
:: [ETAPE 2] GENERATION REQUETE KPEP (CK)
:: ==========================================================
echo.
echo [ETAPE 2/4] Generation Requetes SQL (Phase 2 : KPEP)...
:: Lancement 01 (2eme fois) : Va detecter CM.csv et generer SQL CK
python "%P%01_generation_donnees.py"
if %ERRORLEVEL% NEQ 0 goto :ErreurPython

echo.
echo ----------------------------------------------------------
echo [PAUSE 2] ACTION MANUELLE REQUISE
echo ----------------------------------------------------------
echo.
echo 1. Allez dans le dossier 'Output'.
echo 2. Recuperez la requete SQL terminant par 'KPEP_Global.sql'.
echo 3. Executez-la sur votre base de donnees.
echo 4. Enregistrez le resultat sous '..._CK.csv' dans le dossier 'Input_Data'.
echo.
echo Une fois le fichier CK depose, appuyez sur une touche pour la SUITE.
echo.
pause >nul

:: ==========================================================
:: [ETAPE 2bis] GENERATION REQUETES RELIQUAT (NOM/MIDDLE)
:: ==========================================================
echo.
echo [ETAPE 2bis/4] Generation Requetes SQL (Phase 3 : Reliquat)...
:: Lancement 01 (3eme fois) : Va generer les requetes pour les cas complexes (Nom/Prenom/Middle)
python "%P%01_generation_donnees.py"
if %ERRORLEVEL% NEQ 0 goto :ErreurPython

echo.
echo ----------------------------------------------------------
echo [PAUSE 2bis] ACTION MANUELLE REQUISE
echo ----------------------------------------------------------
echo.
echo 1. Allez dans le dossier 'Output'.
echo 2. Recuperez les requetes SQL complementaires (MiddleName / LastName).
echo 3. Executez-les sur votre base de donnees.
echo 4. Enregistrez les resultats sous '..._NM.csv' (ou _Reliquat.csv) dans 'Input_Data'.
echo.
echo Une fois les fichiers complementaires deposes (si applicable), appuyez sur une touche.
echo.
pause >nul

:: ==========================================================
:: [ETAPE 3] CALCULS FINAUX ET CHARGEMENT
:: ==========================================================
echo.
echo [ETAPE 3/4] Calculs finaux et Chargement...

echo ... Calcul des KPIs ...
python "%P%02_calcul_kpi.py"
if %ERRORLEVEL% NEQ 0 goto :ErreurPython

echo ... Generation fichiers detailles ...
python "%P%03_generation_fichiers_detail.py"
if %ERRORLEVEL% NEQ 0 goto :ErreurPython

echo ... Historisation en BDD ...
python "%P%04_chargement_bdd.py"
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
echo Verifiez que le dossier "Scripts" existe (ou que les .py sont a la racine).
pause
exit

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

:ErreurPython
color 0C
echo.
echo [ERREUR CRITIQUE] Le script Python a echoue.
echo Regardez le message d'erreur ci-dessus.
call deactivate
pause
exit

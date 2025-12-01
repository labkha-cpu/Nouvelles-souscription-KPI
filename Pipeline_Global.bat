@echo off
chcp 65001 >nul
:: Force le repertoire courant a etre celui ou se trouve ce script .bat
cd /d "%~dp0"

title PIPELINE DATA QUALITY - EXECUTION COMPLETE
cls

echo ==========================================================
echo    PIPELINE DATA QUALITY : EXECUTION COMPLETE
echo ==========================================================
echo.
echo Ce script va enchainer toutes les etapes.
echo Gardez cette fenetre ouverte tant que le traitement n'est pas fini.
echo.

:: --- DETECTION DU CHEMIN DES SCRIPTS ---
if exist "01_generation_donnees.py" (
    set "P="
) else (
    if exist "Scripts\01_generation_donnees.py" (
        set "P=Scripts\"
    ) else (
        echo.
        echo ❌ ERREUR CRITIQUE : Impossible de trouver les scripts Python.
        echo Verifiez que le dossier 'Scripts' est present.
        echo.
        pause
        exit /b 1
    )
)

echo.
echo ==========================================================
echo [ETAPE 1] GENERATION REQUETE EMAIL (CM)
echo ==========================================================
echo.
:: Lancement 01 : Va generer IEHE + SQL CM et s'arreter (car CM.csv manquant)
python "%P%01_generation_donnees.py"

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

echo.
echo ==========================================================
echo [ETAPE 2] GENERATION REQUETE KPEP (CK)
echo ==========================================================
echo.
:: Lancement 01 (2eme fois) : Va detecter CM.csv et generer SQL CK
python "%P%01_generation_donnees.py"

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Erreur lors de la generation CK.
    pause
    exit /b
)

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
echo Une fois le fichier CK depose, appuyez sur une touche pour FINALISER.
echo.
pause >nul

echo.
echo ==========================================================
echo [ETAPE 3] CALCULS FINAUX ET CHARGEMENT
echo ==========================================================
echo.

echo ... Calcul des KPI ...
python "%P%02_calcul_kpi.py"

echo ... Generation des fichiers details ...
python "%P%03_generation_fichiers_detail.py"

echo ... Chargement en Base de Donnees ...
python "%P%04_chargement_bdd.py"

echo.
echo ==========================================================
echo    PIPELINE TERMINE AVEC SUCCES !
echo ==========================================================
echo.
pause
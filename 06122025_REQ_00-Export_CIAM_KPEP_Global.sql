/* GENERATED 2025-12-09 16:39:02.665573 | SOURCE: 06122025 | NB: 57 */
/* OPTIMIZED QUERY FOR KPEP
   Logic: Filter Users by KPEP -> Find Latest Event -> Join Attributes ONCE
*/

WITH TargetUsers AS (
    -- 1. Récupération des utilisateurs cibles via KPEP ID (Filtre primaire)
    -- On identifie d'abord les users qui possèdent un des KPEP recherchés
    SELECT
        usr.id,
        usr.email,
        usr.first_name,
        usr.last_name,
        usr.created_timestamp,
        att_kpep.value as kpep_searched -- On garde le KPEP qui a matché pour la ref
    FROM rcia.user_entity usr
    JOIN rcia.user_attribute att_kpep ON usr.id = att_kpep.user_id 
    WHERE att_kpep.name = 'kpepId'
      AND att_kpep.value IN (
          'KPEP00001038708018','KPEP00000667382020','KPEP00001048000438','KPEP00001050728216','KPEP00001042923505','KPEP00000910200214','KPEP00000788465436','KPEP00000471687725','KPEP00001045612010','KPEP00001051742638','KPEP00001039537741','KPEP00001041589824','KPEP00001047597701','KPEP00001040646339','KPEP00001000300533','KPEP00001045345735','KPEP00001039164606','KPEP00001044674400','KPEP00001042484929','KPEP00001045607406','KPEP00001051031424','KPEP00001049965331','KPEP00001037556410','KPEP00000243886408','KPEP00000071007624','KPEP00001054001632','KPEP00000813055729','KPEP00001053520004','KPEP00001038847224','KPEP00001036630836','KPEP00000978706919','KPEP00001036022400','KPEP00001039455517','KPEP00001049820410','KPEP00001046037927','KPEP00001054126016','KPEP00001045106638','KPEP00001044536305','KPEP00000897137002','KPEP00001049481624','KPEP00001054607400','KPEP00001036033729','KPEP00001051028133','KPEP00001049786406','KPEP00001046963038','KPEP00001046465317','KPEP00001052166737','KPEP00001038861440','KPEP00001050987808','KPEP00001048000539','KPEP00000930369927','KPEP00001052928309','KPEP00001041168713','KPEP00001050981226','KPEP00001037240602','KPEP00000332833200','KPEP00001048714741'
      )
      AND usr.realm_id != 'master'
),

LatestEvents AS (
    -- 2. Récupération du DERNIER événement pertinent par utilisateur identifié
    SELECT DISTINCT ON (evt.user_id)
        evt.user_id,
        evt.type,
        evt.client_id,
        evt.event_time
    FROM rcia.event_entity evt
    JOIN TargetUsers tu ON evt.user_id = tu.id
    WHERE evt.type IN ('LOGIN', 'UPDATE_PROFILE')
      AND TO_TIMESTAMP(evt.event_time / 1000)::date BETWEEN TO_TIMESTAMP('2000-03-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS') 
                                                        AND TO_TIMESTAMP('2025-12-09 00:00:00', 'YYYY-MM-DD HH24:MI:SS')
    ORDER BY evt.user_id, evt.event_time DESC
),

UserActivity AS (
    -- 3. Comparaison Création vs Dernier Événement pour choisir la ligne maître
    SELECT
        tu.id,
        tu.email,
        tu.first_name,
        tu.last_name,
        tu.kpep_searched,
        -- Timestamp final
        CASE 
            WHEN le.event_time IS NOT NULL AND le.event_time >= tu.created_timestamp THEN le.event_time
            ELSE tu.created_timestamp
        END as final_timestamp,
        -- Type final
        CASE 
            WHEN le.event_time IS NOT NULL AND le.event_time >= tu.created_timestamp THEN le.type
            ELSE 'CREATION'
        END as final_type,
        -- Client final
        CASE 
            WHEN le.event_time IS NOT NULL AND le.event_time >= tu.created_timestamp THEN le.client_id
            ELSE NULL
        END as final_client,
        -- Filtre de validité date global
        CASE
            WHEN le.event_time IS NOT NULL AND le.event_time >= tu.created_timestamp THEN 1
            WHEN TO_TIMESTAMP(tu.created_timestamp / 1000)::date BETWEEN TO_TIMESTAMP('2000-03-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS') 
                                                                     AND TO_TIMESTAMP('2025-12-09 00:00:00', 'YYYY-MM-DD HH24:MI:SS') THEN 1
            ELSE 0
        END as is_valid
    FROM TargetUsers tu
    LEFT JOIN LatestEvents le ON tu.id = le.user_id
)

-- 4. Selection Finale et Jointure des Attributs (1 seule fois par KPEP trouvé)
SELECT DISTINCT ON (ua.kpep_searched)
    ua.id,
    COALESCE(attrealm.value, '') AS realm_id,
    ua.kpep_searched AS idkpep, -- On utilise la valeur du filtre initial
    ua.email,
    COALESCE(attother.value, '') AS email_other,
    COALESCE(attchannel.value, '') AS channel,
    ua.first_name,
    ua.last_name,
    COALESCE(attmiddle.value, '') AS middleName,
    COALESCE(attphone.value, '') AS phoneNumber,
    COALESCE(attbirth.value, '') AS birthDate,
    ua.final_type AS type,
    COALESCE(attorigin.value, '') AS originCreation,
    ua.final_client AS client,
    TO_TIMESTAMP(ua.final_timestamp/1000)::date AS date_evt,
    TO_TIMESTAMP(ua.final_timestamp/1000)::time AS heure_evt
FROM UserActivity ua
-- Jointures des attributs optimisées (uniquement sur les lignes gagnantes)
LEFT JOIN rcia.user_attribute attrealm ON ua.id = attrealm.user_id AND attrealm.name='societe-codeGestionnaire'
LEFT JOIN rcia.user_attribute attbirth ON ua.id = attbirth.user_id AND attbirth.name = 'birthDate'
LEFT JOIN rcia.user_attribute attother ON ua.id = attother.user_id AND attother.name = 'email_other'
LEFT JOIN rcia.user_attribute attchannel ON ua.id = attchannel.user_id AND attchannel.name = 'ActivationData-DeepLink-Chanel'
LEFT JOIN rcia.user_attribute attmiddle ON ua.id = attmiddle.user_id AND attmiddle.name = 'middleName'
LEFT JOIN rcia.user_attribute attphone ON ua.id = attphone.user_id AND attphone.name = 'phoneNumber'
LEFT JOIN rcia.user_attribute attorigin ON ua.id = attorigin.user_id AND attorigin.name = 'originCreation'
WHERE ua.is_valid = 1
ORDER BY ua.kpep_searched, date_evt DESC;
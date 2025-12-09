/* GENERATED 2025-12-09 16:39:02.668888 | SOURCE: 06122025 | NB: 21 */
/* REQUETE RECHERCHE NOM / DATE NAISSANCE */
SELECT DISTINCT ON (first_name, last_name, birthDate)
  id,
  realm_id,
  idkpep,
  email,
  email_other,
  channel,
  first_name,
  last_name,
  middleName,
  phoneNumber,
  birthDate,
  type,
  originCreation,
  client,
  date_evt,
  heure_evt
FROM (
  -- CREATION
  SELECT
    usr.id,
    COALESCE(attrealm.value, '') AS realm_id,
    COALESCE(att.value, '') AS idkpep,
    usr.email,
    COALESCE(attother.value, '') AS email_other,
    COALESCE(attchannel.value, '') AS channel,
    usr.first_name,
    usr.last_name,
    COALESCE(attmiddle.value, '') AS middleName,
    COALESCE(attphone.value, '') AS phoneNumber,
    COALESCE(att2.value, '') AS birthDate,
    'CREATION' AS type,
    COALESCE(attorigin.value, '') AS originCreation,
    NULL AS client,
    TO_TIMESTAMP(usr.created_timestamp/1000)::date AS date_evt,
    TO_TIMESTAMP(usr.created_timestamp/1000)::time AS heure_evt
  FROM rcia.user_entity usr
  LEFT JOIN rcia.user_attribute attrealm ON usr.id = attrealm.user_id AND attrealm.name='societe-codeGestionnaire'
  LEFT JOIN rcia.user_attribute att ON usr.id = att.user_id AND att.name = 'kpepId'
  LEFT JOIN rcia.user_attribute att2 ON usr.id = att2.user_id AND att2.name = 'birthDate'
  LEFT JOIN rcia.user_attribute attother ON usr.id = attother.user_id AND attother.name = 'email_other'
  LEFT JOIN rcia.user_attribute attchannel ON usr.id = attchannel.user_id AND attchannel.name = 'ActivationData-DeepLink-Chanel'
  LEFT JOIN rcia.user_attribute attmiddle ON usr.id = attmiddle.user_id AND attmiddle.name = 'middleName'
  LEFT JOIN rcia.user_attribute attphone ON usr.id = attphone.user_id AND attphone.name = 'phoneNumber'
  LEFT JOIN rcia.user_attribute attorigin ON usr.id = attorigin.user_id AND attorigin.name = 'originCreation'
  WHERE usr.realm_id != 'master'
    AND TO_TIMESTAMP(usr.created_timestamp / 1000)::date BETWEEN TO_TIMESTAMP('2000-03-01', 'YYYY-MM-DD') AND TO_TIMESTAMP('2025-12-09', 'YYYY-MM-DD')
    AND (
      (usr.last_name ILIKE 'BARMOY' AND att2.value = '2000-08-05')
      OR (usr.last_name ILIKE 'BOIVIN' AND att2.value = '1987-05-05')
      OR (usr.last_name ILIKE 'BOULEZ' AND att2.value = '1982-06-02')
      OR (usr.last_name ILIKE 'CAPIZZI' AND att2.value = '1973-08-04')
      OR (usr.last_name ILIKE 'CHAPU' AND att2.value = '2002-04-08')
      OR (usr.last_name ILIKE 'CUVILLIER' AND att2.value = '1974-03-04')
      OR (usr.last_name ILIKE 'DASSUNCAO' AND att2.value = '2000-07-04')
      OR (usr.last_name ILIKE 'DEL-BONTA' AND att2.value = '2001-10-08')
      OR (usr.last_name ILIKE 'DEMARET' AND att2.value = '1970-01-03')
      OR (usr.last_name ILIKE 'DUBOIS' AND att2.value = '1973-05-01')
      OR (usr.last_name ILIKE 'FELLRATH' AND att2.value = '1973-12-02')
      OR (usr.last_name ILIKE 'GALIA' AND att2.value = '2002-10-07')
      OR (usr.last_name ILIKE 'HUSSON' AND att2.value = '1982-10-12')
      OR (usr.last_name ILIKE 'LAUDON' AND att2.value = '1970-01-05')
      OR (usr.last_name ILIKE 'LOMBARDO' AND att2.value = '1966-01-05')
      OR (usr.last_name ILIKE 'MAKOBA' AND att2.value = '1989-10-05')
      OR (usr.last_name ILIKE 'NADEAU' AND att2.value = '1979-09-05')
      OR (usr.last_name ILIKE 'PETITGENET' AND att2.value = '2000-02-01')
      OR (usr.last_name ILIKE 'RYCKEBUSCH' AND att2.value = '1979-05-08')
      OR (usr.last_name ILIKE 'SAID MOHAMED' AND att2.value = '1981-10-07')
      OR (usr.last_name ILIKE 'SOURTY' AND att2.value = '1970-02-01')
    )

  UNION ALL

  -- LOGIN / UPDATE_PROFILE
  SELECT
    usr.id,
    COALESCE(attrealm.value, '') AS realm_id,
    COALESCE(att.value, '') AS idkpep,
    usr.email,
    COALESCE(attother.value, '') AS email_other,
    COALESCE(attchannel.value, '') AS channel,
    usr.first_name,
    usr.last_name,
    COALESCE(attmiddle.value, '') AS middleName,
    COALESCE(attphone.value, '') AS phoneNumber,
    COALESCE(att2.value, '') AS birthDate,
    evt.type AS type,
    COALESCE(attorigin.value, '') AS originCreation,
    evt.client_id AS client,
    TO_TIMESTAMP(evt.event_time/1000)::date AS date_evt,
    TO_TIMESTAMP(evt.event_time/1000)::time AS heure_evt
  FROM rcia.user_entity usr
  LEFT JOIN rcia.user_attribute attrealm ON usr.id = attrealm.user_id AND attrealm.name='societe-codeGestionnaire'
  LEFT JOIN rcia.user_attribute att ON usr.id = att.user_id AND att.name = 'kpepId'
  LEFT JOIN rcia.user_attribute att2 ON usr.id = att2.user_id AND att2.name = 'birthDate'
  LEFT JOIN rcia.event_entity evt ON evt.user_id = usr.id
  LEFT JOIN rcia.user_attribute attother ON usr.id = attother.user_id AND attother.name = 'email_other'
  LEFT JOIN rcia.user_attribute attchannel ON usr.id = attchannel.user_id AND attchannel.name = 'ActivationData-DeepLink-Chanel'
  LEFT JOIN rcia.user_attribute attmiddle ON usr.id = attmiddle.user_id AND attmiddle.name = 'middleName'
  LEFT JOIN rcia.user_attribute attphone ON usr.id = attphone.user_id AND attphone.name = 'phoneNumber'
  LEFT JOIN rcia.user_attribute attorigin ON usr.id = attorigin.user_id AND attorigin.name = 'originCreation'
  WHERE usr.realm_id != 'master'
    AND evt.type IN ('LOGIN', 'UPDATE_PROFILE')
    AND TO_TIMESTAMP(evt.event_time / 1000)::date BETWEEN TO_TIMESTAMP('2000-03-01', 'YYYY-MM-DD') AND TO_TIMESTAMP('2025-12-09', 'YYYY-MM-DD')
    AND (
      (usr.last_name ILIKE 'BARMOY' AND att2.value = '2000-08-05')
      OR (usr.last_name ILIKE 'BOIVIN' AND att2.value = '1987-05-05')
      OR (usr.last_name ILIKE 'BOULEZ' AND att2.value = '1982-06-02')
      OR (usr.last_name ILIKE 'CAPIZZI' AND att2.value = '1973-08-04')
      OR (usr.last_name ILIKE 'CHAPU' AND att2.value = '2002-04-08')
      OR (usr.last_name ILIKE 'CUVILLIER' AND att2.value = '1974-03-04')
      OR (usr.last_name ILIKE 'DASSUNCAO' AND att2.value = '2000-07-04')
      OR (usr.last_name ILIKE 'DEL-BONTA' AND att2.value = '2001-10-08')
      OR (usr.last_name ILIKE 'DEMARET' AND att2.value = '1970-01-03')
      OR (usr.last_name ILIKE 'DUBOIS' AND att2.value = '1973-05-01')
      OR (usr.last_name ILIKE 'FELLRATH' AND att2.value = '1973-12-02')
      OR (usr.last_name ILIKE 'GALIA' AND att2.value = '2002-10-07')
      OR (usr.last_name ILIKE 'HUSSON' AND att2.value = '1982-10-12')
      OR (usr.last_name ILIKE 'LAUDON' AND att2.value = '1970-01-05')
      OR (usr.last_name ILIKE 'LOMBARDO' AND att2.value = '1966-01-05')
      OR (usr.last_name ILIKE 'MAKOBA' AND att2.value = '1989-10-05')
      OR (usr.last_name ILIKE 'NADEAU' AND att2.value = '1979-09-05')
      OR (usr.last_name ILIKE 'PETITGENET' AND att2.value = '2000-02-01')
      OR (usr.last_name ILIKE 'RYCKEBUSCH' AND att2.value = '1979-05-08')
      OR (usr.last_name ILIKE 'SAID MOHAMED' AND att2.value = '1981-10-07')
      OR (usr.last_name ILIKE 'SOURTY' AND att2.value = '1970-02-01')
    )
) AS unioned
ORDER BY first_name, last_name, birthDate, date_evt DESC;
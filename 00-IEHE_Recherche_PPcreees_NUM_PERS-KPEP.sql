SELECT *
FROM iehe.refkpep r1
JOIN iehe.refkpep r2 ON r2.idrpp = r1.idrpp
WHERE r1.refperboccn IN (
'num_personne1',
'num_personne2
)
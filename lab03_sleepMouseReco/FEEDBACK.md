C'est bien, vous avez été cherché des manières de régulariser le modèle.

Si ça ne marche pas avec mse pour la second expérience, c'est que vous avez trois sorties et 0, 1, 2 comme classes.

On
 s'attend a avoir en sortie un encodage one-hot ([0,0,1], [0,1,0], 
[1,0,0]), SparseCategoricalCrossentropy fait cette conversion 0,1,2 
-> [0,0,1], [0,1,0], [1,0,0]... pour vous. Par-contre au lieu d'avoir
 aucune fonction d'activation sur la dernière couche, on aimerait avoir 
softmax.

Vous n'avez pas clairement séparé la seconde 
expérience du changement pour la compétition, mais vous avez fait des 
modifications (équilibrage des classes par exemple) donc j'ai compté ça 
comme la dernière expérience.

C'est intéressant d'avoir calculé la proportions des classes de vos prédictions.

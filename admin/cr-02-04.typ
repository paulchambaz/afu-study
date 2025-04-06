#import "template.typ": *
#import "@preview/cetz:0.3.1": canvas, draw

#show: report.with(
title: [Compte Rendu de Réunion 8],
course: [Projet #smallcaps[Ai2d]],
authors: ("Paul Chambaz", "Frédéric Li Combeau"),
university: [Sorbonne Université],
reference: [Master #smallcaps[Ai2d] M1 -- 2 avril 2025],
)

== Points clés

- Question sur l'entrainement offline, comment passer de offline à online: plusieurs façons: RLPD: 50% online, 50% offline, Cal-QL: garder le replay buffer, autres: refaire un nouveau replay buffer.
- Replay buffer à taille finie, pop les données les plus anciennes.
- Commencer par Mountaincar et Pendulum plutôt que Lunar Lander comme nous ne sommes pas sûrs d'avoir la meilleure politique.
- Pour le dataset, il faut être en off-policy. Un bon dataset doit couvrir suffisamment l'espace d'états-actions et doit avoir des récompenses, essayer de bricoler un peu. Faire le dataset avec un uniform sampling ? Voir le filtrer après.
- Si l'on a peu de neurones, le réseau va avoir tendance à généraliser beaucoup donc il n'y a pas besoin d'avoir des points qui soient très denses.
- Les articles scientifiques suivent toujours le même plan:
    - Abstract
    - Introduction: Problème ouvert et comment on le résout
    - Related Work: Voir comment se situe notre contribution par rapport aux contributions proches
    - Souvent, Background: L'ensemble des informations que le lecteur doit savoir pour la suite, explication des bases
    - Méthodes: Description de ce que l'on a fait, de l'algorithme
    - Experimental study ou Experimental setup et une partie résultats
    - Conclusion: Qu'est-ce qu'on a apporté de nouveau par rapport à la problématique de l'introduction
- Ennoncer ses contributions sous forme assez précise.
- Faire l'abstract à la fin.
- Généralement il y a une limite de pages, il faut donc être concis, éviter le temps futur. On met généralement le reste dans les annexes.
- Ne pas écrire dans un style historique.

== Travail à faire

- Obtenir les résultats de l'impact du nombre de neurones dans les couches cachées sur Lunar Lander
- Modifier AFU pour obtenir AFU-beta et observer la convergence
- Tester IQL et Cal-QL sur Mountaincar avec l'experience offline et offline to online
- Commencer le rapport sur la partie méthode, related work
- Tester différentes tailles de datasets si possible
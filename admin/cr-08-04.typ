#import "template.typ": *
#import "@preview/cetz:0.3.1": canvas, draw

#show: report.with(
title: [Compte Rendu de Réunion 9],
course: [Projet #smallcaps[Ai2d]],
authors: ("Paul Chambaz", "Frédéric Li Combeau"),
university: [Sorbonne Université],
reference: [Master #smallcaps[Ai2d] M1 -- 8 avril 2025],
)

== Points clés

- Explications sur Cal-QL, en particulier le Monte Carlo return et comment l'implémenter (max(Q, V)).
- Hybrid policy: Voir s'il y a un bug comme les résultats sont étonnants.
- Discussion sur les résultats de nos expériences.
- Stable Baselines pour bien tuner nos algorithmes, site rlbaselinezoo avec tous les environnements et hyperparamètres.
- AFU Beta: reprendre le code d'AFU Perrin et adapter son code, tester avec AFU Perrin si l'on arrive pas à finir. 

== Travail à faire

- Finir AFU Beta
- Finir et tester IQL et Cal-QL
- Fixer l'expérience hybride
- Tester les expériences
- Analyser Mountaincar

23 avril point en visio
5 mai point 9h-10h
8 mai point
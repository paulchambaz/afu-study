#import "template.typ": *
#import "@preview/cetz:0.3.1": canvas, draw

#show: report.with(
title: [Compte Rendu de Réunion 6],
course: [Projet #smallcaps[Ai2d]],
authors: ("Paul Chambaz", "Frédéric Li Combeau"),
university: [Sorbonne Université],
reference: [Master #smallcaps[Ai2d] M1 -- 12 mars 2025],
)

== Points clés

- Discussion sur les environnements, set_state et correction des wrapper customisés
- Discussion sur les résultats d'AFU Perrin, wrapper custimisé et réglage des hyperparamètres
- Discussion sur l'adaptation Off to On-Policy, CalQL et la comparaison avec nos algorithmes

== Axes de travail

- Tester AFU Perrin sans wrapper customisé/corriger le wrapper et régler les hyperparamètres
- Finir les expériences Off to On-Policy
- Continuer AFU BBRL
- Bonus: Comparer un algorithme Offline comme CQL ou IQL sur Cartpole et Pendulum
